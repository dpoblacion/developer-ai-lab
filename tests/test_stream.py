import unittest

from scripts.lib.stream import compute_stream_metrics


class StreamMetricsTest(unittest.TestCase):
    def test_basic(self):
        # 100 tokens; first belongs to TTFT, so 99 decode tokens over the 5s window.
        m = compute_stream_metrics(
            start=100.0, first_token=101.0, last_token=106.0, completion_tokens=100)
        self.assertEqual(m["ttft"], 1.0)
        self.assertEqual(m["decode_tps"], 99 / 5)
        self.assertEqual(m["latency"], 6.0)
        self.assertEqual(m["completion_tokens"], 100)

    def test_zero_decode_window_gives_zero_tps(self):
        m = compute_stream_metrics(
            start=10.0, first_token=11.0, last_token=11.0, completion_tokens=5)
        self.assertEqual(m["decode_tps"], 0.0)

    def test_single_token_gives_zero_tps(self):
        m = compute_stream_metrics(
            start=10.0, first_token=11.0, last_token=11.0, completion_tokens=1)
        self.assertEqual(m["decode_tps"], 0.0)


if __name__ == "__main__":
    unittest.main()
