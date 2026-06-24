import unittest

from scripts.run_sdd_scenario import build_hard_score


class BuildHardScoreTest(unittest.TestCase):
    def test_assembles_schema(self):
        gate_summary = {"total": 2, "passed": 1, "results": []}
        phases = [{"phase": "01-spec", "wall_time": 3.0, "total_tokens": 100}]
        hs = build_hard_score("20260624-000000", "dev-model", "todo-app",
                              gate_summary, phases)
        self.assertEqual(hs["run_id"], "20260624-000000")
        self.assertEqual(hs["model"], "dev-model")
        self.assertEqual(hs["scenario"], "todo-app")
        self.assertEqual(hs["gates"], gate_summary)
        self.assertEqual(hs["phases"], phases)


if __name__ == "__main__":
    unittest.main()
