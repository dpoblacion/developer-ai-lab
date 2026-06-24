import unittest

from scripts.lib.gates import run_gates


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
