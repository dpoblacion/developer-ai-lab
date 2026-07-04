import json
import tempfile
import unittest
from pathlib import Path

from scripts.dashboard import (load_runs, benchmarks, devs_for, combo_rows,
                               compare_narrative, combo_series, benchmark_min_tps, slo_headroom,
                               self_host_usd_per_mtok, best_combo)


def _rep(family="qwen3-coder", quant="fp8", hardware="l40s", gpus=1, benchmark="dev-load",
         run_id="20260701-000000", by_devs=None):
    return {"family": family, "quant": quant, "hardware": hardware, "gpu_count": gpus,
            "benchmark": benchmark, "run_id": run_id, "pod_cost_usd": 0.4,
            "by_devs": by_devs or [{"devs": 8, "holds_slo": True, "median_tps": 30.0,
                                    "median_ttft": 0.09, "cost_per_dev_month": 72.0,
                                    "valid": True, "agents_ok": 6, "tokens_per_dev": 290000}]}


def _entry(devs=8, holds=True, cost=72.0, tps=30.0):
    return {"devs": devs, "holds_slo": holds, "cost_per_dev_month": cost, "median_tps": tps,
            "median_ttft": 0.09, "valid": True, "agents_ok": devs, "tokens_per_dev": 1}


class LoadRunsTest(unittest.TestCase):
    def test_loads_reports_from_results_tree(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "20260629-000000" / "dev-load"
            run.mkdir(parents=True)
            (run / "report.json").write_text(json.dumps({"model": "m", "knee": 2}))
            runs = load_runs(d)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["model"], "m")

    def test_empty_tree_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_runs(d), [])

    def test_loads_reports_from_deep_config_keyed_tree(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "qwen3-coder-fp8" / "h200-1gpu" / "dev-load" / "20260701-115435"
            run.mkdir(parents=True)
            (run / "report.json").write_text(json.dumps({"family": "qwen3-coder"}))
            runs = load_runs(d)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["family"], "qwen3-coder")


class BenchmarksTest(unittest.TestCase):
    def test_distinct_sorted_excludes_smoke(self):
        reps = [_rep(benchmark="dev-load"), _rep(benchmark="smoke"),
                _rep(benchmark="other"), _rep(benchmark=None)]
        self.assertEqual(benchmarks(reps), ["dev-load", "other"])   # smoke + falsy dropped


class DevsForTest(unittest.TestCase):
    def test_distinct_devs_for_benchmark(self):
        r = _rep(benchmark="dev-load", by_devs=[_entry(devs=4), _entry(devs=8)])
        other = _rep(benchmark="other", by_devs=[_entry(devs=16)])
        self.assertEqual(devs_for([r, other], "dev-load"), [4, 8])   # only dev-load's Ns


class ComboRowsTest(unittest.TestCase):
    def test_one_row_per_combo_sorted_by_cost(self):
        cheap = _rep(family="qwen3-coder", quant="fp8", hardware="l40s",
                     by_devs=[_entry(cost=72.0, tps=30.0)])
        pricey = _rep(family="qwen3-coder", quant="fp8", hardware="h200",
                      by_devs=[_entry(cost=328.0, tps=69.0)])
        rows = combo_rows([pricey, cheap], "dev-load", 8)
        self.assertEqual([(r["family"], r["hardware"]) for r in rows],
                         [("qwen3-coder", "l40s"), ("qwen3-coder", "h200")])   # cheapest first
        self.assertEqual(rows[0]["cost_per_dev_month"], 72.0)

    def test_awq_and_fp8_are_distinct_combos(self):
        fp8 = _rep(quant="fp8", hardware="l40s", by_devs=[_entry(cost=72.0)])
        awq = _rep(quant="awq", hardware="l40s", by_devs=[_entry(cost=109.0)])
        rows = combo_rows([fp8, awq], "dev-load", 8)
        self.assertEqual(len(rows), 2)   # quant is part of the key

    def test_repeat_runs_aggregate_per_combo(self):
        # Same (combo, N) measured twice = a repeat (arXiv:2606.11690 §5.8): one row,
        # metrics averaged, repeat count carried.
        old = _rep(hardware="l40s", run_id="20260629-000000", by_devs=[_entry(cost=80.0)])
        new = _rep(hardware="l40s", run_id="20260701-000000", by_devs=[_entry(cost=72.0)])
        rows = combo_rows([old, new], "dev-load", 8)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["cost_per_dev_month"], 76.0)
        self.assertEqual(rows[0]["repeats"], 2)

    def test_skips_other_benchmark_and_missing_n(self):
        wrong_bench = _rep(benchmark="other", by_devs=[_entry(devs=8)])
        wrong_n = _rep(benchmark="dev-load", by_devs=[_entry(devs=4)])
        self.assertEqual(combo_rows([wrong_bench, wrong_n], "dev-load", 8), [])


class CompareNarrativeTest(unittest.TestCase):
    def _rows(self):
        return [{"family": "qwen3-coder", "quant": "fp8", "hardware": "l40s", "gpus": 1,
                 "holds_slo": True, "cost_per_dev_month": 72.0},
                {"family": "qwen3-coder", "quant": "fp8", "hardware": "h200", "gpus": 1,
                 "holds_slo": False, "cost_per_dev_month": 328.0}]

    def test_names_cheapest_holding_combo_and_flags_failures(self):
        text = " ".join(compare_narrative(self._rows(), 8))
        self.assertIn("qwen3-coder-fp8 × l40s-1gpu", text)   # cheapest holding combo
        self.assertIn("does not hold", text.lower())
        self.assertIn("h200", text)

    def test_empty(self):
        self.assertEqual(compare_narrative([], 8), [])

    def test_all_failing_says_none_once(self):
        rows = [dict(self._rows()[0], holds_slo=False), self._rows()[1]]
        out = compare_narrative(rows, 8)
        self.assertEqual(len(out), 1)
        self.assertIn("No model×hardware combo holds", out[0])


class ComboSeriesTest(unittest.TestCase):
    def test_series_merges_runs_points_sorted(self):
        old = _rep(hardware="l40s", run_id="20260629-000000",
                   by_devs=[_entry(devs=8, cost=80.0)])
        new = _rep(hardware="l40s", run_id="20260701-000000",
                   by_devs=[_entry(devs=8, cost=72.0), _entry(devs=4, cost=144.0)])
        series = combo_series([old, new], "dev-load")
        self.assertEqual(len(series), 1)                       # one combo
        self.assertEqual([p["devs"] for p in series[0]["points"]], [4, 8])   # sorted
        # N=8 was measured twice -> a repeat: averaged, not silently replaced
        self.assertAlmostEqual(series[0]["points"][1]["cost_per_dev_month"], 76.0)
        self.assertEqual(series[0]["points"][1]["repeats"], 2)

    def test_skips_other_benchmark(self):
        r = _rep(benchmark="other", by_devs=[_entry(devs=8)])
        self.assertEqual(combo_series([r], "dev-load"), [])


class BenchmarkMinTpsTest(unittest.TestCase):
    def test_reads_floor(self):
        r = dict(_rep(), slo={"min_tps": 20.0, "max_ttft": 2.0})
        self.assertEqual(benchmark_min_tps([r], "dev-load"), 20.0)

    def test_none_when_absent(self):
        self.assertIsNone(benchmark_min_tps([], "dev-load"))


class SloHeadroomTest(unittest.TestCase):
    def test_ratios(self):
        h = slo_headroom(58.0, 20.0, 0.1, 2.0)
        self.assertAlmostEqual(h["tps_ratio"], 2.9)
        self.assertAlmostEqual(h["ttft_ratio"], 20.0)

    def test_zero_safe(self):
        h = slo_headroom(None, 0, 0.0, None)
        self.assertIsNone(h["tps_ratio"])
        self.assertIsNone(h["ttft_ratio"])


class SelfHostPerMtokTest(unittest.TestCase):
    def test_value(self):
        # 0.79/h ÷ (30 tps × 8 × 3600 / 1e6 tok/h) = 0.79e6 / 864000 ≈ 0.914
        self.assertAlmostEqual(self_host_usd_per_mtok(0.79, 30.0, 8), 0.9143, places=3)

    def test_zero_safe(self):
        self.assertIsNone(self_host_usd_per_mtok(0.79, 0.0, 8))
        self.assertIsNone(self_host_usd_per_mtok(0.79, 30.0, 0))


class BestComboTest(unittest.TestCase):
    def test_cheapest_holding(self):
        cheap = _rep(hardware="l40s", by_devs=[_entry(devs=8, holds=True, cost=72.0)])
        pricey = _rep(hardware="h200", by_devs=[_entry(devs=8, holds=True, cost=328.0)])
        b = best_combo([pricey, cheap], "dev-load")
        self.assertEqual(b["cost_per_dev_month"], 72.0)
        self.assertIn("l40s", b["label"])

    def test_none_when_nothing_holds(self):
        r = _rep(by_devs=[_entry(devs=8, holds=False, cost=72.0)])
        self.assertIsNone(best_combo([r], "dev-load"))


class FamiliesTest(unittest.TestCase):
    def test_distinct_families_sorted_for_a_benchmark(self):
        from scripts.dashboard import families
        reports = [_rep(family="qwen3-coder"), _rep(family="llama3.3-70b"),
                   _rep(family="qwen3-coder"), _rep(family="qwen3-14b")]
        self.assertEqual(families(reports, "dev-load"),
                         ["llama3.3-70b", "qwen3-14b", "qwen3-coder"])

    def test_excludes_other_benchmarks(self):
        from scripts.dashboard import families
        reports = [_rep(family="qwen3-coder", benchmark="dev-load"),
                   _rep(family="glm-5.2", benchmark="other")]
        self.assertEqual(families(reports, "dev-load"), ["qwen3-coder"])


class SeriesMergesAcrossRunsTest(unittest.TestCase):
    """N measurements are independent: a new run that tests only new team sizes must ADD
    its points to a combo's curve, not erase the previously measured Ns. Same N measured
    twice → the later run wins."""

    def test_new_run_with_new_ns_extends_the_curve(self):
        from scripts.dashboard import combo_series
        old = _rep(run_id="20260701-000000",
                   by_devs=[_entry(devs=4, cost=144.18), _entry(devs=8, cost=72.09)])
        new = _rep(run_id="20260703-000000", by_devs=[_entry(devs=16, cost=36.04)])
        pts = combo_series([old, new], "dev-load")[0]["points"]
        self.assertEqual([p["devs"] for p in pts], [4, 8, 16])

    def test_same_n_remeasured_aggregates_as_repeats(self):
        # Repeat measurements of the same (combo, N) are the stability protocol of
        # arXiv:2606.11690 §5.8: they average, with a CV, instead of the newest silently
        # replacing the older one.
        from scripts.dashboard import combo_series
        old = _rep(run_id="20260701-000000", by_devs=[_entry(devs=8, cost=72.0, tps=30.0)])
        new = _rep(run_id="20260703-000000", by_devs=[_entry(devs=8, cost=70.0, tps=34.0)])
        pts = combo_series([old, new], "dev-load")[0]["points"]
        self.assertEqual(len(pts), 1)
        self.assertAlmostEqual(pts[0]["cost_per_dev_month"], 71.0)
        self.assertAlmostEqual(pts[0]["median_tps"], 32.0)
        self.assertEqual(pts[0]["repeats"], 2)


class SeriesCarriesTtftAndPriceTest(unittest.TestCase):
    def test_points_include_median_ttft(self):
        from scripts.dashboard import combo_series
        r = _rep(by_devs=[_entry(devs=8)])
        pts = combo_series([r], "dev-load")[0]["points"]
        self.assertIn("median_ttft", pts[0])

    def test_series_carry_gpu_price(self):
        from scripts.dashboard import combo_series
        r = dict(_rep(by_devs=[_entry(devs=8)]), price_usd_per_gpu_hour=0.79)
        s = combo_series([r], "dev-load")[0]
        self.assertEqual(s["price_usd_per_gpu_hour"], 0.79)

    def test_rows_carry_gpu_price(self):
        from scripts.dashboard import combo_rows
        r = dict(_rep(by_devs=[_entry(devs=8)]), price_usd_per_gpu_hour=0.79)
        row = combo_rows([r], "dev-load", 8)[0]
        self.assertEqual(row["price_usd_per_gpu_hour"], 0.79)


class AggregateEntriesTest(unittest.TestCase):
    """Measurement stability (arXiv:2606.11690 §5.8): deliberate repeat runs of the same
    (combo, N) aggregate as mean + CV — the CV is the guardrail that exposes runs that
    disagree — and the SLO verdict is conservative (every repeat must hold)."""

    def test_mean_cv_and_conservative_slo(self):
        from scripts.dashboard import aggregate_entries
        a = dict(_entry(devs=8, tps=30.0), cost_per_mtok=4.0, tokens_per_hour=2e6)
        b = dict(_entry(devs=8, tps=34.0), cost_per_mtok=6.0, tokens_per_hour=3e6)
        out = aggregate_entries([a, b])
        self.assertEqual(out["repeats"], 2)
        self.assertAlmostEqual(out["median_tps"], 32.0)
        self.assertAlmostEqual(out["cost_per_mtok"], 5.0)
        self.assertAlmostEqual(out["tokens_per_hour"], 2.5e6)
        # sample stdev (4,6) = sqrt(2); CV = sqrt(2)/5
        self.assertAlmostEqual(out["cv"]["cost_per_mtok"], 2 ** 0.5 / 5.0, places=6)
        self.assertTrue(out["holds_slo"])

    def test_slo_fails_if_any_repeat_fails(self):
        from scripts.dashboard import aggregate_entries
        out = aggregate_entries([_entry(devs=8, holds=True), _entry(devs=8, holds=False)])
        self.assertFalse(out["holds_slo"])

    def test_single_run_has_no_cv(self):
        from scripts.dashboard import aggregate_entries
        out = aggregate_entries([_entry(devs=8)])
        self.assertEqual(out["repeats"], 1)
        self.assertNotIn("cv", out)


class PaperMetricsTest(unittest.TestCase):
    """Concurrency-aware metrics (arXiv:2606.11690) flow report → dashboard: effective
    $/MTok, utilization, and the p90 SLO tail. Old reports lack them → None, never KeyError."""

    def test_series_points_carry_effective_mtok_and_p90(self):
        from scripts.dashboard import combo_series
        e = dict(_entry(devs=8), cost_per_mtok=3.2, cost_per_mtok_output=19.0,
                 holds_slo_p90=False, tokens_source="client")
        pts = combo_series([_rep(by_devs=[e])], "dev-load")[0]["points"]
        self.assertEqual(pts[0]["cost_per_mtok"], 3.2)
        # cost_per_mtok_output feeds the API-crossover chart: dropping it here renders
        # that chart empty (live 2026-07-04).
        self.assertEqual(pts[0]["cost_per_mtok_output"], 19.0)
        self.assertIs(pts[0]["holds_slo_p90"], False)
        # provenance travels with the point so the page can state it honestly
        self.assertEqual(pts[0]["tokens_source"], "client")

    def test_series_points_default_none_on_old_reports(self):
        from scripts.dashboard import combo_series
        pts = combo_series([_rep(by_devs=[_entry(devs=8)])], "dev-load")[0]["points"]
        self.assertIsNone(pts[0]["cost_per_mtok"])
        self.assertIsNone(pts[0]["utilization"])
        self.assertIsNone(pts[0]["holds_slo_p90"])

    def test_series_utilization_derived_across_merged_runs(self):
        # a40-style history: each N measured in its OWN run (per-run utilization would be
        # a meaningless 100% each). The series derives U over the combo's merged points.
        from scripts.dashboard import combo_series
        r1 = _rep(run_id="20260702-000000",
                  by_devs=[dict(_entry(devs=4), tokens_per_hour=1_000_000.0)])
        r2 = _rep(run_id="20260703-000000",
                  by_devs=[dict(_entry(devs=16), tokens_per_hour=4_000_000.0)])
        pts = combo_series([r1, r2], "dev-load")[0]["points"]
        self.assertAlmostEqual(pts[0]["utilization"], 0.25)
        self.assertAlmostEqual(pts[1]["utilization"], 1.0)

    def test_series_utilization_not_derived_from_a_single_point(self):
        from scripts.dashboard import combo_series
        r = _rep(by_devs=[dict(_entry(devs=8), tokens_per_hour=1_000_000.0)])
        pts = combo_series([r], "dev-load")[0]["points"]
        self.assertIsNone(pts[0]["utilization"])

    def test_rows_carry_paper_metrics(self):
        from scripts.dashboard import combo_rows
        e = dict(_entry(devs=8), cost_per_mtok=3.2, cost_per_mtok_output=19.0,
                 utilization=0.5, underutilization_penalty=2.0,
                 holds_slo_p90=True, p90_tps=22.0, p90_ttft=1.8)
        row = combo_rows([_rep(by_devs=[e])], "dev-load", 8)[0]
        self.assertEqual(row["cost_per_mtok"], 3.2)
        self.assertEqual(row["cost_per_mtok_output"], 19.0)
        self.assertEqual(row["utilization"], 0.5)
        self.assertEqual(row["underutilization_penalty"], 2.0)
        self.assertIs(row["holds_slo_p90"], True)
        self.assertEqual(row["p90_tps"], 22.0)
        self.assertEqual(row["p90_ttft"], 1.8)

    def test_rows_default_none_on_old_reports(self):
        from scripts.dashboard import combo_rows
        row = combo_rows([_rep(by_devs=[_entry(devs=8)])], "dev-load", 8)[0]
        self.assertIsNone(row["cost_per_mtok"])
        self.assertIsNone(row["holds_slo_p90"])


class BenchmarkMaxTtftTest(unittest.TestCase):
    def test_reads_the_slo_ceiling(self):
        from scripts.dashboard import benchmark_max_ttft
        r = dict(_rep(by_devs=[_entry(devs=8)]), slo={"min_tps": 20.0, "max_ttft": 2.0})
        self.assertEqual(benchmark_max_ttft([r], "dev-load"), 2.0)

    def test_none_when_absent(self):
        from scripts.dashboard import benchmark_max_ttft
        self.assertIsNone(benchmark_max_ttft([_rep(by_devs=[_entry(devs=8)])], "dev-load"))


def _series_one(hardware="l40s", gpus=1, price=0.79, cost=72.0875):
    return {"label": f"q-fp8 × {hardware}-{gpus}gpu", "hardware": hardware, "gpus": gpus,
            "quant": "fp8", "family": "q", "price_usd_per_gpu_hour": price,
            "points": [{"devs": 8, "cost_per_dev_month": cost, "median_tps": 30.0,
                        "median_ttft": 0.09, "holds_slo": True}]}


class RepriceTest(unittest.TestCase):
    """What-if pricing: $/dev is pure price math (price × gpus × 730 ÷ N), so the dashboard
    can recompute it live from a visitor's own GPU price. SLO outcomes are measured and
    must never change with price."""

    def test_reprice_series_recomputes_cost_and_keeps_slo(self):
        from scripts.dashboard import reprice_series
        series = [_series_one()]
        out = reprice_series(series, {"l40s": 0.40})
        self.assertAlmostEqual(out[0]["points"][0]["cost_per_dev_month"], 0.40 * 730 / 8)
        self.assertTrue(out[0]["points"][0]["holds_slo"])
        # pure: the input series is untouched
        self.assertAlmostEqual(series[0]["points"][0]["cost_per_dev_month"], 72.0875)

    def test_no_override_leaves_series_unchanged(self):
        from scripts.dashboard import reprice_series
        out = reprice_series([_series_one()], {})
        self.assertAlmostEqual(out[0]["points"][0]["cost_per_dev_month"], 72.0875)

    def test_reprice_rows_recomputes_and_resorts(self):
        from scripts.dashboard import reprice_rows
        cheap = {"hardware": "l40s", "gpus": 1, "cost_per_dev_month": 72.09,
                 "cost_per_hour": 0.79, "price_usd_per_gpu_hour": 0.79, "holds_slo": True}
        mid = {"hardware": "a100-80gb", "gpus": 1, "cost_per_dev_month": 108.59,
               "cost_per_hour": 1.19, "price_usd_per_gpu_hour": 1.19, "holds_slo": True}
        out = reprice_rows([cheap, mid], {"l40s": 2.0}, n=8)
        self.assertEqual(out[0]["hardware"], "a100-80gb")        # order flipped
        self.assertAlmostEqual(out[1]["cost_per_dev_month"], 2.0 * 730 / 8)
        self.assertAlmostEqual(out[1]["cost_per_hour"], 2.0)


class SeriesWithMetricTest(unittest.TestCase):
    def test_filters_unmeasured_points_and_drops_empty_series(self):
        from scripts.dashboard import series_with_metric
        measured = _series_one()
        measured["points"][0]["cost_per_mtok"] = 4.0
        old = _series_one(hardware="a40")            # pre-paper run: no mtok anywhere
        out = series_with_metric([measured, old], "cost_per_mtok")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["hardware"], "l40s")
        # pure: inputs untouched
        self.assertEqual(len(old["points"]), 1)


def _paper_series(label="glm-fp8 × h200-8gpu", pts=None):
    return {"label": label, "hardware": "h200", "gpus": 8, "quant": "fp8", "family": "glm",
            "price_usd_per_gpu_hour": 3.59,
            "points": pts if pts is not None else [
                {"devs": 1, "holds_slo": True, "cost_per_dev_month": 100.0,
                 "median_tps": 40.0, "median_ttft": 0.1,
                 "cost_per_mtok": 24.0, "cost_per_mtok_output": 120.0, "utilization": 0.1},
                {"devs": 8, "holds_slo": True, "cost_per_dev_month": 12.5,
                 "median_tps": 30.0, "median_ttft": 0.3,
                 "cost_per_mtok": 3.0, "cost_per_mtok_output": 18.0, "utilization": 0.8},
                {"devs": 16, "holds_slo": False, "cost_per_dev_month": 6.25,
                 "median_tps": 10.0, "median_ttft": 4.0,
                 "cost_per_mtok": 2.0, "cost_per_mtok_output": 12.0, "utilization": 1.0}]}


class MtokSpreadTest(unittest.TestCase):
    """The paper's headline: the same hardware's effective $/MTok spans a huge factor
    across offered load (17.5-36.3x on H100). Per combo: max/min measured cost_per_mtok."""

    def test_spread_across_measured_points(self):
        from scripts.dashboard import mtok_spread
        out = mtok_spread([_paper_series()])
        self.assertEqual(len(out), 1)
        s = out[0]
        self.assertEqual(s["label"], "glm-fp8 × h200-8gpu")
        self.assertAlmostEqual(s["max_mtok"], 24.0)      # at N=1
        self.assertAlmostEqual(s["min_mtok"], 2.0)       # at N=16
        self.assertAlmostEqual(s["spread"], 12.0)

    def test_skips_series_without_measurements_or_single_point(self):
        from scripts.dashboard import mtok_spread
        old = _series_one()                               # no cost_per_mtok
        single = _paper_series(pts=[_paper_series()["points"][0]])
        self.assertEqual(mtok_spread([old, single]), [])


class ApiCrossoverTest(unittest.TestCase):
    """Crossover vs a per-token API price (paper §4): the smallest measured N where the
    self-host output-$/MTok drops to or below the reference API price."""

    def test_first_n_at_or_below_api_price(self):
        from scripts.dashboard import api_crossover
        out = api_crossover([_paper_series()], api_price=18.0)
        self.assertEqual(out[0]["crossover_devs"], 8)     # 120 > 18; 18 <= 18 at N=8

    def test_none_when_never_cheaper(self):
        from scripts.dashboard import api_crossover
        out = api_crossover([_paper_series()], api_price=1.0)
        self.assertIsNone(out[0]["crossover_devs"])


def _quant_series(quant, hardware="l40s", tph_by_n=None):
    pts = [{"devs": n, "holds_slo": True, "cost_per_dev_month": 10.0, "median_tps": 30.0,
            "median_ttft": 0.1, "tokens_per_hour": tph, "cost_per_mtok": 1e6 / tph}
           for n, tph in (tph_by_n or {}).items()]
    return {"label": f"q-{quant} × {hardware}-1gpu", "family": "q", "quant": quant,
            "hardware": hardware, "gpus": 1, "price_usd_per_gpu_hour": 1.0, "points": pts}


class QuantImpactTest(unittest.TestCase):
    """Quantization impact (arXiv:2606.11690 §FP8-asymmetry) is only measurable between
    series that differ in NOTHING but the quant: same family, same hardware, same GPU
    count, compared at their common team sizes. Anything else confounds quant with the
    GPU, so it must never be paired."""

    def test_pairs_same_family_hardware_gpus_at_common_ns(self):
        from scripts.dashboard import quant_impact
        awq = _quant_series("awq", tph_by_n={4: 1_000_000.0, 8: 2_000_000.0})
        fp8 = _quant_series("fp8", tph_by_n={8: 3_000_000.0, 16: 4_000_000.0})
        out = quant_impact([awq, fp8])
        self.assertEqual(len(out), 1)
        pair = out[0]
        self.assertEqual((pair["quant_a"], pair["quant_b"]), ("awq", "fp8"))
        self.assertEqual([p["devs"] for p in pair["points"]], [8])   # only the common N
        p = pair["points"][0]
        self.assertAlmostEqual(p["tph_a"], 2_000_000.0)
        self.assertAlmostEqual(p["tph_b"], 3_000_000.0)
        self.assertAlmostEqual(p["uplift"], 0.5)                     # fp8 = +50% vs awq

    def test_never_pairs_across_hardware(self):
        from scripts.dashboard import quant_impact
        awq = _quant_series("awq", hardware="a40", tph_by_n={8: 1_000_000.0})
        fp8 = _quant_series("fp8", hardware="l40s", tph_by_n={8: 3_000_000.0})
        self.assertEqual(quant_impact([awq, fp8]), [])

    def test_no_common_measured_ns_yields_no_pair(self):
        from scripts.dashboard import quant_impact
        awq = _quant_series("awq", tph_by_n={4: 1_000_000.0})
        fp8 = _quant_series("fp8", tph_by_n={16: 4_000_000.0})
        self.assertEqual(quant_impact([awq, fp8]), [])


class ThetaMaxSeriesTest(unittest.TestCase):
    def test_series_utilization_uses_measured_theta_max(self):
        # With a probed Θmax the series' utilization is the paper's true U — even for a
        # single measured N — and the basis is reported for the caption.
        from scripts.dashboard import combo_series
        r = _rep(by_devs=[dict(_entry(devs=8), tokens_per_hour=1_000_000.0)])
        r["theta_max"] = {"tokens_per_hour": 4_000_000.0,
                          "output_tokens_per_hour": 800_000.0}
        s = combo_series([r], "dev-load")[0]
        self.assertEqual(s["theta_max"]["tokens_per_hour"], 4_000_000.0)
        self.assertEqual(s["utilization_basis"], "theta_max")
        self.assertAlmostEqual(s["points"][0]["utilization"], 0.25)

    def test_series_utilization_prefers_computed_throughput(self):
        # Cache-hit tokens are discounted from U's numerator (compute vs compute).
        from scripts.dashboard import combo_series
        r = _rep(by_devs=[dict(_entry(devs=8), tokens_per_hour=3_000_000.0,
                               computed_tokens_per_hour=1_000_000.0)])
        r["theta_max"] = {"tokens_per_hour": 4_000_000.0}
        s = combo_series([r], "dev-load")[0]
        self.assertAlmostEqual(s["points"][0]["utilization"], 0.25)

    def test_series_falls_back_to_best_level_basis(self):
        from scripts.dashboard import combo_series
        r1 = _rep(run_id="a", by_devs=[dict(_entry(devs=4), tokens_per_hour=1_000_000.0)])
        r2 = _rep(run_id="b", by_devs=[dict(_entry(devs=16), tokens_per_hour=4_000_000.0)])
        s = combo_series([r1, r2], "dev-load")[0]
        self.assertEqual(s["utilization_basis"], "best_level")
        self.assertAlmostEqual(s["points"][0]["utilization"], 0.25)


class SlaTableTest(unittest.TestCase):
    """arXiv:2606.11690 Table 4: the SLO's price. Per combo, the largest team size that
    holds the SLO, the output-$/MTok there, the saturation floor (from measured Θmax),
    and the premium the SLO imposes over that floor."""

    def test_sla_row_with_measured_theta_max(self):
        from scripts.dashboard import sla_table
        s = _paper_series()                                     # N=16 breaks SLO
        s["theta_max"] = {"tokens_per_hour": 5_000_000.0,
                          "output_tokens_per_hour": 1_000_000.0}
        s["gpus"] = 8
        s["price_usd_per_gpu_hour"] = 3.59
        rows = sla_table([s])
        row = rows[0]
        self.assertEqual(row["sla_devs"], 8)                    # largest N holding SLO
        self.assertAlmostEqual(row["mtok_at_sla"], 18.0)        # output $/MTok at N=8
        # Csat = cost_per_hour / Θmax_output × 1e6 = 3.59*8 / 1.0M × 1e6 = 28.72
        self.assertAlmostEqual(row["c_sat"], 28.72)
        self.assertAlmostEqual(row["premium"], 18.0 / 28.72)

    def test_sla_row_without_theta_max_has_no_floor(self):
        from scripts.dashboard import sla_table
        rows = sla_table([_paper_series()])
        self.assertEqual(rows[0]["sla_devs"], 8)
        self.assertIsNone(rows[0]["c_sat"])
        self.assertIsNone(rows[0]["premium"])

    def test_no_row_when_nothing_holds(self):
        from scripts.dashboard import sla_table
        pts = [dict(p, holds_slo=False) for p in _paper_series()["points"]]
        rows = sla_table([_paper_series(pts=pts)])
        self.assertIsNone(rows[0]["sla_devs"])


class RepricePaperMetricsTest(unittest.TestCase):
    """cost_per_mtok is price-derived (price/hour ÷ measured throughput) → the what-if
    override rescales it by the price ratio. Utilization is measured → never changes."""

    def test_reprice_series_rescales_mtok_keeps_utilization(self):
        from scripts.dashboard import reprice_series
        s = _series_one()
        s["points"][0].update(cost_per_mtok=4.0, utilization=0.5)
        out = reprice_series([s], {"l40s": 1.58})           # 2× the 0.79 list price
        self.assertAlmostEqual(out[0]["points"][0]["cost_per_mtok"], 8.0)
        self.assertEqual(out[0]["points"][0]["utilization"], 0.5)
        self.assertAlmostEqual(s["points"][0]["cost_per_mtok"], 4.0)   # pure

    def test_reprice_rows_rescales_both_mtok_figures(self):
        from scripts.dashboard import reprice_rows
        row = {"hardware": "l40s", "gpus": 1, "cost_per_dev_month": 72.09,
               "cost_per_hour": 0.79, "price_usd_per_gpu_hour": 0.79, "holds_slo": True,
               "cost_per_mtok": 4.0, "cost_per_mtok_output": 20.0, "utilization": 0.5}
        out = reprice_rows([row], {"l40s": 1.58}, n=8)
        self.assertAlmostEqual(out[0]["cost_per_mtok"], 8.0)
        self.assertAlmostEqual(out[0]["cost_per_mtok_output"], 40.0)
        self.assertEqual(out[0]["utilization"], 0.5)


class BestComboFromSeriesTest(unittest.TestCase):
    def test_picks_cheapest_holding_point(self):
        from scripts.dashboard import best_combo_from_series
        s = _series_one()
        best = best_combo_from_series([s])
        self.assertEqual(best["devs"], 8)
        self.assertAlmostEqual(best["cost_per_dev_month"], 72.0875)


class AssignSeriesColorsTest(unittest.TestCase):
    """Color follows the entity, never its rank: a combo keeps its color across charts and
    across filter changes, and hues come from the fixed validated palette in fixed order."""

    def test_mapping_is_stable_regardless_of_input_order(self):
        from scripts.dashboard import assign_series_colors
        a = assign_series_colors(["b-combo", "a-combo"])
        b = assign_series_colors(["a-combo", "b-combo"])
        self.assertEqual(a, b)

    def test_hues_assigned_in_fixed_palette_order_by_sorted_label(self):
        from scripts.dashboard import assign_series_colors, CATEGORICAL
        m = assign_series_colors(["zeta", "alpha"])
        self.assertEqual(m["alpha"], CATEGORICAL[0])
        self.assertEqual(m["zeta"], CATEGORICAL[1])

    def test_more_labels_than_slots_wraps_without_crashing(self):
        from scripts.dashboard import assign_series_colors, CATEGORICAL
        labels = [f"combo-{i}" for i in range(len(CATEGORICAL) + 1)]
        m = assign_series_colors(labels)
        self.assertEqual(len(m), len(labels))


class ShortSeriesLabelsTest(unittest.TestCase):
    def test_compact_labels_use_hardware_and_quant(self):
        from scripts.dashboard import short_series_labels
        series = [{"label": "qwen3-coder-fp8 × l40s-1gpu", "hardware": "l40s", "gpus": 1,
                   "quant": "fp8", "family": "qwen3-coder"},
                  {"label": "qwen3-coder-awq × a100-80gb-1gpu", "hardware": "a100-80gb",
                   "gpus": 1, "quant": "awq", "family": "qwen3-coder"}]
        shorts = short_series_labels(series)
        self.assertEqual(shorts["qwen3-coder-fp8 × l40s-1gpu"], "l40s-1gpu fp8")
        self.assertEqual(shorts["qwen3-coder-awq × a100-80gb-1gpu"], "a100-80gb-1gpu awq")

    def test_falls_back_to_full_labels_when_ambiguous(self):
        from scripts.dashboard import short_series_labels
        # two families, same hardware+quant → the short form would collide
        series = [{"label": "qwen3-coder-fp8 × h200-8gpu", "hardware": "h200", "gpus": 8,
                   "quant": "fp8", "family": "qwen3-coder"},
                  {"label": "glm-5.2-fp8 × h200-8gpu", "hardware": "h200", "gpus": 8,
                   "quant": "fp8", "family": "glm-5.2"}]
        shorts = short_series_labels(series)
        self.assertEqual(shorts["qwen3-coder-fp8 × h200-8gpu"], "qwen3-coder-fp8 × h200-8gpu")
        self.assertEqual(shorts["glm-5.2-fp8 × h200-8gpu"], "glm-5.2-fp8 × h200-8gpu")


class ComboRowsExtraFieldsTest(unittest.TestCase):
    def test_carries_slo_and_cost_per_hour(self):
        r = dict(_rep(by_devs=[_entry(devs=8)]), slo={"min_tps": 20.0, "max_ttft": 2.0},
                 cost_per_hour=0.79)
        row = combo_rows([r], "dev-load", 8)[0]
        self.assertEqual(row["min_tps"], 20.0)
        self.assertEqual(row["max_ttft"], 2.0)
        self.assertEqual(row["cost_per_hour"], 0.79)


if __name__ == "__main__":
    unittest.main()
