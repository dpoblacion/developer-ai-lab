import unittest

from scripts.lib.slo import summarize_level, evaluate_slo, find_knee

SLO = {"max_ttft": 2.0, "min_tps": 20.0}


class SloTest(unittest.TestCase):
    def test_summarize_level(self):
        samples = [
            {"ttft": 1.0, "decode_tps": 30.0},
            {"ttft": 1.5, "decode_tps": 25.0},
        ]
        s = summarize_level(2, samples)
        self.assertEqual(s["concurrency"], 2)
        self.assertEqual(s["streams"], 2)
        self.assertEqual(s["ttft_median"], 1.25)
        self.assertEqual(s["ttft_max"], 1.5)
        self.assertEqual(s["tps_median"], 27.5)
        self.assertEqual(s["tps_min"], 25.0)

    def test_evaluate_slo_pass(self):
        s = {"ttft_median": 1.25, "tps_median": 27.5}
        self.assertTrue(evaluate_slo(s, SLO))

    def test_evaluate_slo_fail_on_ttft(self):
        s = {"ttft_median": 3.0, "tps_median": 27.5}
        self.assertFalse(evaluate_slo(s, SLO))

    def test_evaluate_slo_fail_on_tps(self):
        s = {"ttft_median": 1.0, "tps_median": 12.0}
        self.assertFalse(evaluate_slo(s, SLO))

    def test_find_knee_returns_largest_passing_prefix(self):
        summaries = [
            {"concurrency": 1, "ttft_median": 1.0, "tps_median": 40.0},
            {"concurrency": 2, "ttft_median": 1.2, "tps_median": 35.0},
            {"concurrency": 4, "ttft_median": 1.8, "tps_median": 22.0},
            {"concurrency": 8, "ttft_median": 3.5, "tps_median": 10.0},
        ]
        self.assertEqual(find_knee(summaries, SLO), 4)

    def test_find_knee_zero_when_first_fails(self):
        summaries = [{"concurrency": 1, "ttft_median": 5.0, "tps_median": 5.0}]
        self.assertEqual(find_knee(summaries, SLO), 0)


if __name__ == "__main__":
    unittest.main()
