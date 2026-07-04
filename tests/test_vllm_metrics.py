import unittest

from scripts.lib.vllm_metrics import (
    counter_total, histogram_buckets, quantile_from_buckets, level_latency, level_metrics,
    level_tokens)

# Minimal Prometheus snapshots: a TTFT histogram and a time-per-output-token histogram.
# Buckets are cumulative. BEFORE has all-zero counts; AFTER has 4 requests whose TTFT and
# per-token times land in known buckets so the medians are predictable.
BEFORE = """
vllm:time_to_first_token_seconds_bucket{le="0.5"} 0
vllm:time_to_first_token_seconds_bucket{le="1.0"} 0
vllm:time_to_first_token_seconds_bucket{le="2.0"} 0
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 0
vllm:time_per_output_token_seconds_bucket{le="0.01"} 0
vllm:time_per_output_token_seconds_bucket{le="0.05"} 0
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 0
vllm:e2e_request_latency_seconds_bucket{le="10.0"} 0
vllm:e2e_request_latency_seconds_bucket{le="60.0"} 0
vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 0
vllm:prompt_tokens_total{engine="0"} 100
vllm:generation_tokens_total{engine="0"} 50
vllm:gpu_prefix_cache_queries{engine="0"} 1000
vllm:gpu_prefix_cache_hits{engine="0"} 400
"""
AFTER = """
vllm:time_to_first_token_seconds_bucket{le="0.5"} 1
vllm:time_to_first_token_seconds_bucket{le="1.0"} 2
vllm:time_to_first_token_seconds_bucket{le="2.0"} 4
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 4
vllm:time_per_output_token_seconds_bucket{le="0.01"} 0
vllm:time_per_output_token_seconds_bucket{le="0.05"} 4
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 4
vllm:e2e_request_latency_seconds_bucket{le="10.0"} 2
vllm:e2e_request_latency_seconds_bucket{le="60.0"} 4
vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 4
vllm:prompt_tokens_total{engine="0"} 40100
vllm:generation_tokens_total{engine="0"} 12050
vllm:gpu_prefix_cache_queries{engine="0"} 21000
vllm:gpu_prefix_cache_hits{engine="0"} 15400
"""


class HistogramTest(unittest.TestCase):
    def test_buckets_parsed_sorted_with_inf(self):
        b = histogram_buckets(AFTER, "vllm:time_to_first_token_seconds")
        self.assertEqual(b[0], (0.5, 1.0))
        self.assertEqual(b[-1][0], float("inf"))
        self.assertEqual(b[-1][1], 4.0)

    def test_quantile_interpolates(self):
        # cumulative: (0.5,1),(1.0,2),(2.0,4),(inf,4); total 4; median rank=2 -> at le=1.0
        b = histogram_buckets(AFTER, "vllm:time_to_first_token_seconds")
        self.assertAlmostEqual(quantile_from_buckets(b, 0.5), 1.0, places=3)

    def test_quantile_empty_is_zero(self):
        b = histogram_buckets(BEFORE, "vllm:time_to_first_token_seconds")
        self.assertEqual(quantile_from_buckets(b, 0.5), 0.0)


class CounterTest(unittest.TestCase):
    def test_counter_total_sums_label_sets(self):
        text = ('vllm:prompt_tokens_total{engine="0"} 100\n'
                'vllm:prompt_tokens_total{engine="1"} 20\n')
        self.assertEqual(counter_total(text, "vllm:prompt_tokens_total"), 120.0)

    def test_counter_total_ignores_other_metrics_and_buckets(self):
        # A histogram bucket line must not leak into a counter sum.
        text = ('vllm:prompt_tokens_total 7\n'
                'vllm:generation_tokens_total{engine="0"} 99\n'
                'vllm:time_to_first_token_seconds_bucket{le="0.5"} 3\n')
        self.assertEqual(counter_total(text, "vllm:prompt_tokens_total"), 7.0)

    def test_counter_total_missing_metric_is_zero(self):
        self.assertEqual(counter_total("", "vllm:prompt_tokens_total"), 0.0)


class LevelTokensTest(unittest.TestCase):
    def test_delta_prompt_and_generation_tokens(self):
        # Server-side truth for the level: counters diffed across the two snapshots.
        self.assertEqual(level_tokens(BEFORE, AFTER),
                         {"prompt": 40000, "generation": 12000})


class PrefixCacheTest(unittest.TestCase):
    def test_hit_rate_from_counter_diffs(self):
        # Agentic traffic shares real prefixes (system prompts, tool loops), so unlike
        # arXiv:2606.11690's random-token protocol the cache is live here — the hit rate
        # bounds how our numbers compare to cache-free measurements (their §5.7 probe:
        # real hits cut saturation Ceff by 20-22%).
        from scripts.lib.vllm_metrics import level_prefix_cache
        out = level_prefix_cache(BEFORE, AFTER)
        self.assertEqual(out, {"queries": 20000, "hits": 15000, "hit_rate": 0.75})

    def test_none_hit_rate_when_no_queries(self):
        from scripts.lib.vllm_metrics import level_prefix_cache
        out = level_prefix_cache(BEFORE, BEFORE)
        self.assertIsNone(out["hit_rate"])

    def test_vllm_011_counter_names(self):
        # vLLM 0.11 spells these WITHOUT the gpu_ prefix (vllm:prefix_cache_queries_total)
        # — seen live 2026-07-04 in a run's metrics snapshot; the gpu_-less names must
        # parse too, and the *_created timestamp companions must not leak into the sum.
        from scripts.lib.vllm_metrics import level_prefix_cache
        before = ('vllm:prefix_cache_queries_total{engine="0"} 1000\n'
                  'vllm:prefix_cache_hits_total{engine="0"} 400\n'
                  'vllm:prefix_cache_queries_created{engine="0"} 1.78e9\n')
        after = ('vllm:prefix_cache_queries_total{engine="0"} 21000\n'
                 'vllm:prefix_cache_hits_total{engine="0"} 15400\n'
                 'vllm:prefix_cache_queries_created{engine="0"} 1.78e9\n')
        out = level_prefix_cache(before, after)
        self.assertEqual(out, {"queries": 20000, "hits": 15000, "hit_rate": 0.75})


class LevelMetricsTest(unittest.TestCase):
    def test_merges_latency_tokens_and_duration(self):
        # The one dict a level hands to dev_record: latency quantiles + server tokens +
        # how long the level ran (the denominator of tokens/hour).
        out = level_metrics(BEFORE, AFTER, duration_s=120.0)
        self.assertAlmostEqual(out["median_ttft"], 1.0, places=3)
        self.assertAlmostEqual(out["p90_ttft"], 1.8, places=3)
        self.assertEqual(out["server_tokens"], {"prompt": 40000, "generation": 12000})
        self.assertEqual(out["duration_s"], 120.0)
        self.assertEqual(out["prefix_cache"]["hit_rate"], 0.75)


class LevelLatencyTest(unittest.TestCase):
    def test_delta_median_ttft_and_tps(self):
        out = level_latency(BEFORE, AFTER)
        # median TTFT = 1.0s (from the delta histogram)
        self.assertAlmostEqual(out["median_ttft"], 1.0, places=3)
        # median per-output-token lands in (0.01, 0.05]; interp at rank 2 of 4 -> 0.03s/token
        # tps = 1 / 0.03 ~= 33.3
        self.assertAlmostEqual(out["median_tps"], 1.0 / 0.03, places=1)

    def test_p99_tail_and_e2e_latency(self):
        # arXiv:2606.11690 measures P50/P90/P99 of TTFT, TPOT and E2E (§4.3); its example
        # SLA is expressed at p99 (TTFT ≤300ms, TPOT ≤50ms).
        out = level_latency(BEFORE, AFTER)
        # TTFT p99: rank 3.96 of 4 in (1.0, 2.0] -> 1.0 + (3.96-2)/2 = 1.98s
        self.assertAlmostEqual(out["p99_ttft"], 1.98, places=3)
        # TPOT p99: rank 3.96 in (0.01, 0.05] -> 0.0496 s/token -> 20.2 tok/s
        self.assertAlmostEqual(out["p99_tps"], 1.0 / 0.0496, places=1)
        # E2E: (10,2),(60,4); p50 rank 2 -> 10.0s; p99 rank 3.96 -> 10+50*(1.96/2)=59.0s
        self.assertAlmostEqual(out["e2e_p50"], 10.0, places=3)
        self.assertAlmostEqual(out["e2e_p99"], 59.0, places=3)

    def test_p90_tail_ttft_and_tps(self):
        # Medians hide the tail: a level can hold SLO at p50 while 40% of requests violate
        # it (concurrency-aware costing needs the tail — arXiv:2606.11690 conditions on p99).
        out = level_latency(BEFORE, AFTER)
        # TTFT p90: rank 3.6 of 4 in (1.0, 2.0] -> 1.0 + 1.0*(3.6-2)/(4-2) = 1.8s
        self.assertAlmostEqual(out["p90_ttft"], 1.8, places=3)
        # TPOT p90: rank 3.6 in (0.01, 0.05] -> 0.046 s/token -> 21.7 tok/s
        self.assertAlmostEqual(out["p90_tps"], 1.0 / 0.046, places=1)
