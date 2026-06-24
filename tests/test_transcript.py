import unittest
from pathlib import Path

from scripts.lib.transcript import parse_metrics

FIXTURE = Path(__file__).parent / "fixtures" / "stream.jsonl"


class ParseMetricsTest(unittest.TestCase):
    def test_parse_metrics_from_fixture(self):
        lines = FIXTURE.read_text().splitlines()
        m = parse_metrics(lines)
        self.assertEqual(m["prompt_tokens"], 300)
        self.assertEqual(m["completion_tokens"], 80)
        self.assertEqual(m["total_tokens"], 380)
        self.assertEqual(m["tool_calls"], {"Read": 1, "Edit": 1})
        self.assertEqual(m["turns"], 2)
        self.assertEqual(m["result_text"], "done")

    def test_ignores_blank_lines(self):
        lines = ["", "  ", '{"type":"result","result":"x","usage":{"input_tokens":1,"output_tokens":2}}']
        m = parse_metrics(lines)
        self.assertEqual(m["total_tokens"], 3)
        self.assertEqual(m["turns"], 0)
        self.assertEqual(m["tool_calls"], {})


if __name__ == "__main__":
    unittest.main()
