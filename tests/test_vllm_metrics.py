import unittest

from scripts.lib.vllm_metrics import histogram_buckets, quantile_from_buckets, level_latency

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
"""
AFTER = """
vllm:time_to_first_token_seconds_bucket{le="0.5"} 1
vllm:time_to_first_token_seconds_bucket{le="1.0"} 2
vllm:time_to_first_token_seconds_bucket{le="2.0"} 4
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 4
vllm:time_per_output_token_seconds_bucket{le="0.01"} 0
vllm:time_per_output_token_seconds_bucket{le="0.05"} 4
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 4
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


class LevelLatencyTest(unittest.TestCase):
    def test_delta_median_ttft_and_tps(self):
        out = level_latency(BEFORE, AFTER)
        # median TTFT = 1.0s (from the delta histogram)
        self.assertAlmostEqual(out["median_ttft"], 1.0, places=3)
        # median per-output-token lands in (0.01, 0.05]; interp at rank 2 of 4 -> 0.03s/token
        # tps = 1 / 0.03 ~= 33.3
        self.assertAlmostEqual(out["median_tps"], 1.0 / 0.03, places=1)
