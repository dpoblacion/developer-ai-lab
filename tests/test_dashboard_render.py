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


# The paper-metrics fixture (concurrency-aware fields, arXiv:2606.11690) lives in
# tests/fixtures/paper_report.json: AppTest.from_function serializes the app function
# WITHOUT this module's globals, so the apps below must be self-contained.

def _benchmark_app_paper():
    import json
    import pathlib
    from scripts import dashboard
    rep = json.loads(pathlib.Path("tests/fixtures/paper_report.json").read_text())
    dashboard._PAGES.clear()
    dashboard._PAGES.update(reports=[rep], benchmarks={}, detail=None)
    dashboard.benchmark_page("dev-load")


def _benchmark_app_prepaper():
    # A pre-paper report (no economics fields): the section must hide entirely.
    from scripts import dashboard
    rep = {"family": "q", "quant": "fp8", "hardware": "l40s", "gpu_count": 1,
           "benchmark": "dev-load", "run_id": "20260601-000000", "pod_cost_usd": 0.4,
           "price_usd_per_gpu_hour": 0.79, "cost_per_hour": 0.79,
           "slo": {"max_ttft": 2.0, "min_tps": 20.0},
           "by_devs": [{"devs": 8, "holds_slo": True, "median_tps": 30.0,
                        "median_ttft": 0.1, "cost_per_dev_month": 72.0, "valid": True,
                        "agents_ok": 8, "tokens_per_dev": 1000, "agents": []}]}
    dashboard._PAGES.clear()
    dashboard._PAGES.update(reports=[rep], benchmarks={}, detail=None)
    dashboard.benchmark_page("dev-load")


def _detail_app_paper():
    import json
    import pathlib
    from scripts import dashboard
    rep = json.loads(pathlib.Path("tests/fixtures/paper_report.json").read_text())
    dashboard.render_detail(rep)


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
        from scripts import dashboard
        at = self._run(_benchmark_app)
        if dashboard.benchmarks(dashboard.load_runs("results")):
            self.assertTrue(at.dataframe)  # the combos table is present when there's data

    def test_detail_page_renders(self):
        self._run(_detail_app)

    def test_benchmark_page_shows_effective_mtok_and_utilization(self):
        # With paper metrics present the page grows the per-token economics section:
        # its own header, the spread/SLO-floor/crossover narrative, and the three curves
        # (blended $/MTok, output $/MTok vs an API reference price, utilization) —
        # hidden entirely on pre-paper reports.
        at = self._run(_benchmark_app_paper)
        self.assertIn("Per-token economics",
                      " ".join(str(s.value) for s in at.subheader))
        md = " ".join(str(m.value) for m in at.markdown)
        self.assertIn("Effective $/MTok", md)
        self.assertIn("Utilization", md)
        self.assertIn("Cost spread", md)             # spread table (paper Fig 1 takeaway)
        self.assertIn("penalty", md.lower())         # penalty heatmap (paper Fig 3)
        self.assertIn("API", md)                     # crossover chart/narrative (Fig 5)
        captions = " ".join(str(c.value) for c in at.caption) + md
        self.assertIn("Quantization impact", captions)   # Fig 2 (bars or requirement)
        self.assertIn("The SLO's price", md)             # Fig 4 + Table 4 (SLA floor)
        # Streamlit renders $…$ as LaTeX: a raw dollar in markdown mangles the text
        # (seen live 2026-07-04). Every dollar in narrative markdown must be escaped.
        for m in at.markdown:
            text = str(m.value)
            self.assertNotRegex(text, r"(?<!\\)\$\d",
                                f"unescaped dollar in markdown: {text[:80]!r}")

    def test_benchmark_page_hides_economics_without_paper_metrics(self):
        at = self._run(_benchmark_app_prepaper)
        md = " ".join(str(m.value) for m in at.markdown)
        self.assertNotIn("Effective $/MTok", md)

    def test_detail_page_renders_paper_metrics(self):
        self._run(_detail_app_paper)
