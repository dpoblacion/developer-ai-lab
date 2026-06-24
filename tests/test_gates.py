import unittest

from scripts.lib.gates import run_gates, tail


class TailTest(unittest.TestCase):
    def test_returns_whole_string_when_short(self):
        self.assertEqual(tail("hello", 2048), "hello")

    def test_keeps_last_limit_chars(self):
        self.assertEqual(tail("abcdef", 3), "def")


class GateOutputTest(unittest.TestCase):
    def test_failing_gate_keeps_output_tail(self):
        summary = run_gates([{"id": "boom", "cmd": "echo broken-here >&2; exit 1"}], cwd=".")
        result = summary["results"][0]
        self.assertFalse(result["passed"])
        self.assertIn("broken-here", result["output"])

    def test_passing_gate_has_no_output(self):
        summary = run_gates([{"id": "ok", "cmd": "true"}], cwd=".")
        self.assertNotIn("output", summary["results"][0])


class RunGatesTest(unittest.TestCase):
    def test_pass_fail_and_custom_exit(self):
        gates = [
            {"id": "ok", "cmd": "true", "expect_exit": 0},
            {"id": "bad", "cmd": "false", "expect_exit": 0},
            {"id": "custom", "cmd": "exit 3", "expect_exit": 3},
        ]
        summary = run_gates(gates)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["passed"], 2)
        by_id = {r["id"]: r for r in summary["results"]}
        self.assertTrue(by_id["ok"]["passed"])
        self.assertFalse(by_id["bad"]["passed"])
        self.assertEqual(by_id["bad"]["exit_code"], 1)
        self.assertTrue(by_id["custom"]["passed"])

    def test_expect_exit_defaults_to_zero(self):
        summary = run_gates([{"id": "g", "cmd": "true"}])
        self.assertTrue(summary["results"][0]["passed"])


if __name__ == "__main__":
    unittest.main()
