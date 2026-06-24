import unittest

from scripts.run_gates import build_hard_score


class BuildHardScoreTest(unittest.TestCase):
    def test_assembles_schema(self):
        gate_summary = {"total": 2, "passed": 1, "results": []}
        hs = build_hard_score("20260624-000000", "dev-model", "dev-load", gate_summary)
        self.assertEqual(hs["run_id"], "20260624-000000")
        self.assertEqual(hs["model"], "dev-model")
        self.assertEqual(hs["scenario"], "dev-load")
        self.assertEqual(hs["gates"], gate_summary)


if __name__ == "__main__":
    unittest.main()
