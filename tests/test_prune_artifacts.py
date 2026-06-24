"""prune_artifacts deletes files — pin exactly what it may remove: every stale run's
artifacts/, never a report.json, never the latest run."""
import pathlib
import tempfile
import unittest

from scripts.prune_artifacts import main
from tests.support import mk_run as _mk_run


class PruneMainTest(unittest.TestCase):
    def test_removes_only_stale_artifacts_keeps_reports(self):
        with tempfile.TemporaryDirectory() as d:
            old = _mk_run(d, "20260101-000000")
            new = _mk_run(d, "20260102-000000")
            main(results_dir=d)
            self.assertFalse((old / "artifacts").exists())   # stale run pruned
            self.assertTrue((new / "artifacts").exists())    # latest kept
            self.assertTrue((old / "report.json").exists())  # reports always kept
            self.assertTrue((new / "report.json").exists())

    def test_noop_when_nothing_to_prune(self):
        with tempfile.TemporaryDirectory() as d:
            only = _mk_run(d, "20260101-000000")
            main(results_dir=d)                              # single run: nothing stale
            self.assertTrue((only / "artifacts").exists())

    def test_noop_on_missing_results_dir(self):
        with tempfile.TemporaryDirectory() as d:
            main(results_dir=str(pathlib.Path(d) / "nope"))  # must not raise
