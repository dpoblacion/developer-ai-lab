import unittest
from pathlib import Path

from scripts.run_agent import build_metrics

FIXTURE = Path(__file__).parent / "fixtures" / "stream.jsonl"


class BuildMetricsTest(unittest.TestCase):
    def test_combines_parse_and_walltime(self):
        lines = FIXTURE.read_text().splitlines()
        m = build_metrics(lines, wall_time=12.5, label="smoke", model="dev-model")
        self.assertEqual(m["wall_time"], 12.5)
        self.assertEqual(m["label"], "smoke")
        self.assertEqual(m["model"], "dev-model")
        self.assertEqual(m["total_tokens"], 380)
        self.assertEqual(m["tool_calls"], {"Read": 1, "Edit": 1})


if __name__ == "__main__":
    unittest.main()
