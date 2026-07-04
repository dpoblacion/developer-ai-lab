import unittest
from scripts.lib.bench_report import agent_record, build_report, dev_record

SLO = {"max_ttft": 2.0, "min_tps": 20.0}

# A level's full metrics dict: medians + p90 tail + server-side tokens + level duration.
LAT_FULL = {"median_ttft": 0.1, "median_tps": 30.0, "p90_ttft": 2.5, "p90_tps": 22.0,
            "server_tokens": {"prompt": 900000, "generation": 180000}, "duration_s": 1800.0}


def _hard(*pairs):
    return {"gates": {"results": [{"id": i, "passed": p, "exit_code": 0 if p else 1}
                                  for i, p in pairs]}}


def _gen(total=300000, wall=200.0):
    return {"phases": [{"prompt_tokens": total // 2, "completion_tokens": total // 2,
                        "total_tokens": total, "wall_time": wall}]}


def _agents(*specs):
    # specs: (all_gates_pass: bool, total_tokens, wall)
    return [agent_record(i, _hard(("g", ok)), _gen(tot, wall))
            for i, (ok, tot, wall) in enumerate(specs)]


class DevRecordTest(unittest.TestCase):
    def test_holds_slo_cost_and_validity(self):
        agents = _agents((True, 300000, 200.0), (False, 280000, 190.0))
        r = dev_record(8, {"median_ttft": 0.1, "median_tps": 30.0}, agents, SLO, cost_per_hour=0.79)
        self.assertEqual(r["devs"], 8)
        self.assertTrue(r["holds_slo"])
        self.assertAlmostEqual(r["cost_per_dev_hour"], 0.79 / 8)
        self.assertAlmostEqual(r["cost_per_dev_month"], 0.79 / 8 * 730)
        self.assertTrue(r["valid"])                 # median wall 195s >= 30
        self.assertEqual(r["agents_ok"], 1)         # only the first passed gates
        self.assertEqual(r["tokens_per_dev"], 290000)   # median of 300k/280k

    def test_fails_slo_on_tps(self):
        r = dev_record(8, {"median_ttft": 0.1, "median_tps": 5.0},
                       _agents((True, 300000, 200.0)), SLO, cost_per_hour=0.79)
        self.assertFalse(r["holds_slo"])

    def test_fails_slo_on_ttft(self):
        r = dev_record(8, {"median_ttft": 3.0, "median_tps": 30.0},
                       _agents((True, 300000, 200.0)), SLO, cost_per_hour=0.79)
        self.assertFalse(r["holds_slo"])

    def test_effective_cost_per_mtok_from_server_tokens(self):
        # Concurrency-aware costing (arXiv:2606.11690): Ceff = price/hour over the token
        # throughput the level actually achieved, from server-side counters.
        lat = dict(LAT_FULL)
        r = dev_record(8, lat, _agents((True, 300000, 200.0)), SLO, cost_per_hour=7.2)
        self.assertEqual(r["server_tokens"], {"prompt": 900000, "generation": 180000})
        self.assertEqual(r["tokens_source"], "server")
        # 1.08M tokens in 0.5h -> 2.16M tok/h
        self.assertAlmostEqual(r["tokens_per_hour"], 2_160_000.0)
        self.assertAlmostEqual(r["cost_per_mtok"], 7.2 / 2.16, places=3)      # blended
        # all cost on generated tokens (the API-comparable figure): 0.36M tok/h
        self.assertAlmostEqual(r["cost_per_mtok_output"], 20.0, places=3)

    def test_slo_evaluated_at_configured_percentile(self):
        # arXiv:2606.11690 expresses its SLA at p99; the scenario picks the percentile
        # the gate is judged at (default: median, the harness's historical behavior).
        lat = dict(LAT_FULL, p99_ttft=3.0, p99_tps=21.0)
        r = dev_record(8, lat, _agents((True, 300000, 200.0)),
                       dict(SLO, percentile="p99"), cost_per_hour=7.2)
        self.assertFalse(r["holds_slo"])            # p99_ttft 3.0 > max_ttft 2.0
        self.assertEqual(r["slo_percentile"], "p99")

    def test_record_carries_p99_and_e2e_when_measured(self):
        lat = dict(LAT_FULL, p99_ttft=2.6, p99_tps=21.0, e2e_p50=30.0, e2e_p99=90.0)
        r = dev_record(8, lat, _agents((True, 300000, 200.0)), SLO, cost_per_hour=7.2)
        self.assertEqual(r["p99_ttft"], 2.6)
        self.assertEqual(r["p99_tps"], 21.0)
        self.assertEqual(r["e2e_p50"], 30.0)
        self.assertEqual(r["e2e_p99"], 90.0)

    def test_record_carries_prefix_cache_hit_rate(self):
        lat = dict(LAT_FULL, prefix_cache={"queries": 20000, "hits": 15000, "hit_rate": 0.75})
        r = dev_record(8, lat, _agents((True, 300000, 200.0)), SLO, cost_per_hour=7.2)
        self.assertEqual(r["prefix_cache_hit_rate"], 0.75)

    def test_computed_throughput_discounts_cache_hits(self):
        # Cache-hit prompt tokens are served nearly for free: counting them against a
        # no-cache Θmax probe made U exceed 100% (live 2026-07-04, 96% hit rate).
        # computed_tokens_per_hour discounts the hits so utilization compares compute
        # with compute; the $/MTok economics stay on SERVED tokens (what you receive).
        lat = dict(LAT_FULL, prefix_cache={"queries": 900000, "hits": 450000,
                                           "hit_rate": 0.5})
        r = dev_record(8, lat, _agents((True, 300000, 200.0)), SLO, cost_per_hour=7.2)
        # (900k prompt - 450k cached + 180k generated) / 0.5h
        self.assertAlmostEqual(r["computed_tokens_per_hour"], 1_260_000.0)
        self.assertAlmostEqual(r["tokens_per_hour"], 2_160_000.0)   # served, unchanged

    def test_utilization_prefers_computed_throughput(self):
        lat = dict(LAT_FULL, prefix_cache={"queries": 900000, "hits": 450000,
                                           "hit_rate": 0.5})
        rec = dev_record(8, lat, _agents((True, 300000, 200.0)), SLO, cost_per_hour=7.2)
        theta = {"tokens_per_hour": 2_520_000.0, "output_tokens_per_hour": 500_000.0}
        rep = build_report("q", "fp8", "m", "l40s", "dev-load", 1, 0.79,
                           "20260704-000000", [rec], SLO, theta_max=theta)
        self.assertAlmostEqual(rep["by_devs"][0]["utilization"], 0.5)   # 1.26M / 2.52M

    def test_p90_slo_reported_alongside_median(self):
        # Median passes but the p90 tail violates TTFT: holds_slo stays the (median) gate,
        # holds_slo_p90 exposes what the median hides.
        r = dev_record(8, dict(LAT_FULL), _agents((True, 300000, 200.0)), SLO, cost_per_hour=7.2)
        self.assertTrue(r["holds_slo"])
        self.assertFalse(r["holds_slo_p90"])            # p90_ttft 2.5 > max_ttft 2.0
        self.assertEqual(r["p90_ttft"], 2.5)
        self.assertEqual(r["p90_tps"], 22.0)

    def test_falls_back_to_client_reported_tokens(self):
        # Without server counters the economics come from the agents' client-reported
        # usage (what the server processed) over the slowest agent's wall time — marked
        # tokens_source="client" so consumers can tell the provenance apart.
        r = dev_record(8, {"median_ttft": 0.1, "median_tps": 30.0},
                       _agents((True, 300000, 200.0), (False, 280000, 190.0)),
                       SLO, cost_per_hour=0.79)
        self.assertEqual(r["tokens_source"], "client")
        # _gen splits totals half prompt / half completion: 150k+140k per side
        self.assertEqual(r["server_tokens"], {"prompt": 290000, "generation": 290000})
        self.assertAlmostEqual(r["tokens_per_hour"], 580000 * 3600 / 200.0)
        self.assertAlmostEqual(r["cost_per_mtok"], 0.79e6 / (580000 * 3600 / 200.0))
        self.assertNotIn("p90_ttft", r)     # tails exist only with server histograms

    def test_no_tokens_anywhere_omits_economics(self):
        r = dev_record(8, {"median_ttft": 0.0, "median_tps": 0.0}, [], SLO,
                       cost_per_hour=0.79)
        for k in ("server_tokens", "tokens_per_hour", "cost_per_mtok",
                  "cost_per_mtok_output", "tokens_source", "holds_slo_p90", "p90_ttft"):
            self.assertNotIn(k, r)

    def test_invalid_when_agents_did_no_real_work(self):
        r = dev_record(8, {"median_ttft": 0.0, "median_tps": 0.0},
                       _agents((False, 0, 3.0)), SLO, cost_per_hour=0.79)   # 3s wall = garbage
        self.assertFalse(r["valid"])


class AgentRecordTest(unittest.TestCase):
    def test_gate_timeout_is_marked_and_fails(self):
        # An environmental gate timeout must be distinguishable in the report from a
        # genuine gate failure (it says nothing about the agent's work).
        rec = agent_record(0, {}, _gen(total=15), gate_timed_out=True)
        self.assertFalse(rec["passed"])
        self.assertTrue(rec["gate_timeout"])

    def test_no_gate_timeout_key_by_default(self):
        rec = agent_record(0, _hard(("g", True)), _gen(total=15))
        self.assertNotIn("gate_timeout", rec)

    def test_passed_when_all_gates_pass(self):
        hs = {"gates": {"results": [{"id": "typecheck", "passed": True, "exit_code": 0},
                                    {"id": "acceptance", "passed": True, "exit_code": 0}]}}
        rec = agent_record(0, hs, _gen(total=15))
        self.assertTrue(rec["passed"])
        self.assertEqual(rec["tokens"]["total"], 15)

    def test_output_kept_only_for_failing_gate(self):
        hs = {"gates": {"results": [
            {"id": "typecheck", "passed": True, "exit_code": 0},
            {"id": "acceptance", "passed": False, "exit_code": 1, "output": "addTask is not a function"}]}}
        rec = agent_record(1, hs, _gen())
        gates = {g["id"]: g for g in rec["gates"]}
        self.assertFalse(rec["passed"])
        self.assertNotIn("output", gates["typecheck"])
        self.assertIn("addTask", gates["acceptance"]["output"])


class BuildReportTest(unittest.TestCase):
    def _dev(self):
        return dev_record(8, {"median_ttft": 0.1, "median_tps": 30.0},
                          _agents((True, 300000, 200.0)), SLO, cost_per_hour=0.79)

    def test_flat_shape(self):
        segs = [{"label": "generation", "seconds": 3600}]
        rep = build_report("qwen3-coder", "fp8", "qwen3-coder-30b-fp8", "l40s", "dev-load",
                           1, 0.79, "20260701-000000", [self._dev()], SLO, timeline_segments=segs)
        self.assertEqual(rep["family"], "qwen3-coder")
        self.assertEqual(rep["quant"], "fp8")
        self.assertEqual(rep["hardware"], "l40s")
        self.assertEqual([d["devs"] for d in rep["by_devs"]], [8])
        self.assertNotIn("knee", rep)
        self.assertNotIn("quality_curve", rep)   # the flat report never reintroduces the curve
        self.assertAlmostEqual(rep["pod_cost_usd"], 0.79)     # 3600s @ 0.79/h
        self.assertEqual(rep["tokens"]["total"], 300000)

    def test_utilization_against_measured_theta_max(self):
        # With a measured Θmax (saturation probe) utilization is the paper's true
        # U = Θachieved/Θmax — meaningful even for a single level.
        rec = dev_record(8, dict(LAT_FULL), _agents((True, 300000, 200.0)),
                         SLO, cost_per_hour=7.2)                      # 2.16M tok/h
        theta = {"tokens_per_hour": 4_320_000.0, "output_tokens_per_hour": 720_000.0,
                 "duration_s": 120.0, "concurrency": 64,
                 "shape": {"prompt_tokens": 2560, "output_tokens": 256},
                 "server_tokens": {"prompt": 120000, "generation": 24000}}
        rep = build_report("glm-5.2", "fp8", "m", "h200", "dev-load",
                           8, 0.9, "20260704-000000", [rec], SLO, theta_max=theta)
        self.assertEqual(rep["theta_max"], theta)
        self.assertEqual(rep["utilization_basis"], "theta_max")
        self.assertAlmostEqual(rep["by_devs"][0]["utilization"], 0.5)
        self.assertAlmostEqual(rep["by_devs"][0]["underutilization_penalty"], 2.0)

    def test_utilization_basis_marks_best_level_fallback(self):
        recs = [dev_record(4, dict(LAT_FULL, server_tokens={"prompt": 450000, "generation": 90000}),
                           _agents((True, 300000, 200.0)), SLO, cost_per_hour=7.2),
                dev_record(8, dict(LAT_FULL), _agents((True, 300000, 200.0)),
                           SLO, cost_per_hour=7.2)]
        rep = build_report("glm-5.2", "fp8", "m", "h200", "dev-load",
                           8, 0.9, "20260704-000000", recs, SLO)
        self.assertEqual(rep["utilization_basis"], "best_level")

    def test_utilization_relative_to_best_observed_level(self):
        # U(N) = level throughput / best throughput observed in the run (a lower bound on
        # true saturation), and the 1/U underutilization penalty (arXiv:2606.11690): what
        # you pay at low N for headroom you aren't using.
        fast = dict(LAT_FULL)                                     # 2.16M tok/h
        slow = dict(LAT_FULL, server_tokens={"prompt": 450000, "generation": 90000})  # 1.08M
        recs = [dev_record(4, slow, _agents((True, 300000, 200.0)), SLO, cost_per_hour=7.2),
                dev_record(8, fast, _agents((True, 300000, 200.0)), SLO, cost_per_hour=7.2)]
        rep = build_report("glm-5.2", "fp8", "m", "h200", "dev-load",
                           8, 0.9, "20260704-000000", recs, SLO)
        by_devs = {d["devs"]: d for d in rep["by_devs"]}
        self.assertAlmostEqual(by_devs[8]["utilization"], 1.0)
        self.assertAlmostEqual(by_devs[4]["utilization"], 0.5)
        self.assertAlmostEqual(by_devs[4]["underutilization_penalty"], 2.0)

    def test_utilization_omitted_for_a_single_measured_level(self):
        # One level is trivially its own best — a meaningless 100% would mislead.
        rec = dev_record(8, dict(LAT_FULL), _agents((True, 300000, 200.0)),
                         SLO, cost_per_hour=7.2)
        rep = build_report("glm-5.2", "fp8", "m", "h200", "dev-load",
                           8, 0.9, "20260704-000000", [rec], SLO)
        self.assertNotIn("utilization", rep["by_devs"][0])

    def test_utilization_omitted_without_throughput(self):
        rep = build_report("qwen3-coder", "fp8", "m", "l40s", "dev-load",
                           1, 0.79, "20260701-000000", [self._dev()], SLO)
        self.assertNotIn("utilization", rep["by_devs"][0])

    def test_carries_model_meta_when_given(self):
        # Architecture metadata (dense/MoE, total/active params) enables the
        # active-parameters analyses of arXiv:2606.11690 (§5.2, Result 3).
        meta = {"architecture": "moe", "total_params_b": 30.5, "active_params_b": 3.3}
        rep = build_report("qwen3-coder", "fp8", "m", "l40s", "dev-load",
                           1, 0.79, "20260704-000000", [self._dev()], SLO, model_meta=meta)
        self.assertEqual(rep["model_meta"], meta)

    def test_no_model_meta_omits_key(self):
        rep = build_report("qwen3-coder", "fp8", "m", "l40s", "dev-load",
                           1, 0.79, "20260704-000000", [self._dev()], SLO)
        self.assertNotIn("model_meta", rep)

    def test_truncated_flag_marks_watchdog_aborted_runs(self):
        # A watchdog abort (stall/max_run/max_phase) must not lose the completed levels:
        # the runner writes a partial report marked truncated (two paid runs died with
        # ALL their measurements before this existed — live 2026-07-04).
        rep = build_report("qwen3-14b", "fp8", "m", "l40s", "dev-load",
                           1, 0.79, "20260704-000000", [self._dev()], SLO,
                           truncated="max_run")
        self.assertEqual(rep["truncated"], "max_run")

    def test_not_truncated_omits_the_key(self):
        rep = build_report("qwen3-coder", "fp8", "m", "l40s", "dev-load",
                           1, 0.79, "20260704-000000", [self._dev()], SLO)
        self.assertNotIn("truncated", rep)

    def test_no_timeline_omits_keys(self):
        rep = build_report("qwen3-coder", "fp8", "m", "l40s", "dev-load",
                           1, 0.79, "20260701-000000", [self._dev()], SLO)  # no timeline_segments
        self.assertNotIn("timeline", rep)
        self.assertNotIn("pod_cost_usd", rep)
