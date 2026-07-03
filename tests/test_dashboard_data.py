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

    def test_latest_run_wins_per_combo(self):
        old = _rep(hardware="l40s", run_id="20260629-000000", by_devs=[_entry(cost=80.0)])
        new = _rep(hardware="l40s", run_id="20260701-000000", by_devs=[_entry(cost=72.0)])
        rows = combo_rows([old, new], "dev-load", 8)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cost_per_dev_month"], 72.0)

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
    def test_series_latest_wins_points_sorted(self):
        old = _rep(hardware="l40s", run_id="20260629-000000",
                   by_devs=[_entry(devs=8, cost=80.0)])
        new = _rep(hardware="l40s", run_id="20260701-000000",
                   by_devs=[_entry(devs=8, cost=72.0), _entry(devs=4, cost=144.0)])
        series = combo_series([old, new], "dev-load")
        self.assertEqual(len(series), 1)                       # one combo, latest wins
        self.assertEqual([p["devs"] for p in series[0]["points"]], [4, 8])   # sorted
        self.assertEqual(series[0]["points"][1]["cost_per_dev_month"], 72.0)  # from newest run

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

    def test_same_n_remeasured_takes_the_latest(self):
        from scripts.dashboard import combo_series
        old = _rep(run_id="20260701-000000", by_devs=[_entry(devs=8, cost=72.09)])
        new = _rep(run_id="20260703-000000", by_devs=[_entry(devs=8, cost=70.0)])
        pts = combo_series([old, new], "dev-load")[0]["points"]
        self.assertEqual(len(pts), 1)
        self.assertEqual(pts[0]["cost_per_dev_month"], 70.0)


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
