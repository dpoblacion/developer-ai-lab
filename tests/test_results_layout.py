import unittest
import pathlib
import tempfile
from scripts.lib.results_layout import run_out_dir, artifacts_to_prune
from tests.support import mk_run


class RunOutDirTest(unittest.TestCase):
    def test_builds_benchmark_keyed_path(self):
        # Benchmark first: results/ answers "what did dev-load measure?" before
        # "which configs exist" — the benchmark is the question, the config the answer.
        p = run_out_dir("qwen3-coder", "fp8", "h200", 1, "dev-load", "20260701-115435")
        self.assertEqual(
            p, pathlib.Path("results/dev-load/qwen3-coder-fp8/h200-1gpu/20260701-115435"))

    def test_root_override_and_gpu_count(self):
        p = run_out_dir("glm-5.2", "fp8", "h200", 8, "smoke", "20260701-000000", root="/tmp/r")
        self.assertEqual(p, pathlib.Path("/tmp/r/smoke/glm-5.2-fp8/h200-8gpu/20260701-000000"))


class ArtifactsToPruneTest(unittest.TestCase):
    def test_keeps_latest_prunes_older(self):
        with tempfile.TemporaryDirectory() as d:
            mk_run(d, "20260701-000000")
            mk_run(d, "20260702-000000")  # latest
            prune = artifacts_to_prune(d)
            self.assertEqual([str(p) for p in prune],
                             [str(pathlib.Path(d) / "dev-load/qwen3-coder-fp8/h200-1gpu"
                              / "20260701-000000" / "artifacts")])

    def test_single_run_prunes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            mk_run(d, "20260701-000000")
            self.assertEqual(artifacts_to_prune(d), [])

    def test_never_lists_report_and_empty_tree_ok(self):
        with tempfile.TemporaryDirectory() as d:
            # older run has NO artifacts dir -> nothing to prune even though it is not latest
            mk_run(d, "20260701-000000", with_artifacts=False)
            mk_run(d, "20260702-000000")
            self.assertEqual(artifacts_to_prune(d), [])
            self.assertEqual(artifacts_to_prune(d + "/does-not-exist"), [])


if __name__ == "__main__":
    unittest.main()
