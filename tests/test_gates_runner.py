import unittest

from scripts.run_gates import build_hard_score


class BuildHardScoreTest(unittest.TestCase):
    def test_assembles_schema(self):
        gate_summary = {"total": 2, "passed": 1, "results": []}
        hs = build_hard_score("20260624-000000", "dev-model", "todo-app", gate_summary)
        self.assertEqual(hs["run_id"], "20260624-000000")
        self.assertEqual(hs["model"], "dev-model")
        self.assertEqual(hs["scenario"], "todo-app")
        self.assertEqual(hs["gates"], gate_summary)


if __name__ == "__main__":
    unittest.main()
