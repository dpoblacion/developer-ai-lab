"""Render smoke tests: every dashboard page must execute without exceptions against the
committed results/**/report.json. Runs via Streamlit's AppTest (no browser); skipped
automatically where streamlit isn't installed (the plain-unit CI job) — the dedicated
dashboard CI job installs requirements-report.txt and runs it."""
import importlib.util
import unittest

HAVE_STREAMLIT = importlib.util.find_spec("streamlit") is not None


def _overview_app():
    from scripts.dashboard import main
    main()


def _benchmark_app():
    from scripts import dashboard
    reports = dashboard.load_runs("results")
    dashboard._PAGES.clear()
    dashboard._PAGES.update(reports=reports, benchmarks={}, detail=None)
    names = dashboard.benchmarks(reports)
    if names:
        dashboard.benchmark_page(names[0])


def _detail_app():
    from scripts import dashboard
    reports = dashboard.load_runs("results")
    if reports:
        dashboard.render_detail(reports[0])


@unittest.skipUnless(HAVE_STREAMLIT, "streamlit not installed (dashboard CI job covers this)")
class DashboardRenderTest(unittest.TestCase):
    def _run(self, app):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_function(app, default_timeout=60)
        at.run()
        self.assertEqual([str(e.value) for e in at.exception], [])
        return at

    def test_overview_renders(self):
        at = self._run(_overview_app)
        self.assertTrue(at.title)          # page produced content

    def test_benchmark_page_renders(self):
        at = self._run(_benchmark_app)
        self.assertTrue(at.dataframe)      # the combos table is present

    def test_detail_page_renders(self):
        self._run(_detail_app)
