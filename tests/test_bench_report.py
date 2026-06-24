import unittest
from scripts.lib.bench_report import agent_record, dev_record, build_report

SLO = {"max_ttft": 2.0, "min_tps": 20.0}


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

    def test_no_timeline_omits_keys(self):
        rep = build_report("qwen3-coder", "fp8", "m", "l40s", "dev-load",
                           1, 0.79, "20260701-000000", [self._dev()], SLO)  # no timeline_segments
        self.assertNotIn("timeline", rep)
        self.assertNotIn("pod_cost_usd", rep)
