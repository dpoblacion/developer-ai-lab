import unittest

from scripts.lib.report_terminal import _render_plain


def _report():
    return {
        "family": "qwen3-coder", "quant": "fp8", "hardware": "l40s", "gpu_count": 1,
        "by_devs": [
            {"devs": 4, "holds_slo": True, "median_tps": 56.8, "median_ttft": 0.08,
             "cost_per_dev_month": 144.0, "valid": True, "agents_ok": 4, "tokens_per_dev": 290000,
             "agents": []},
            {"devs": 8, "holds_slo": True, "median_tps": 30.1, "median_ttft": 0.09,
             "cost_per_dev_month": 72.0, "valid": True, "agents_ok": 6, "tokens_per_dev": 293000,
             "agents": [{"agent": 2, "passed": False,
                         "gates": [{"id": "acceptance", "passed": False, "output": "boom"}]}]},
        ],
        "timeline": [{"label": "generation", "seconds": 1500, "cost_usd": 0.33, "pct": 0.8}],
        "pod_cost_usd": 0.41,
    }


class RenderPlainTest(unittest.TestCase):
    def setUp(self):
        self.lines = []
        _render_plain(_report(), out=self.lines.append)
        self.text = "\n".join(self.lines)

    def test_header(self):
        self.assertIn("qwen3-coder", self.text)
        self.assertIn("fp8", self.text)
        self.assertIn("l40s", self.text)

    def test_per_devs_lines(self):
        self.assertIn("n=8", self.text)
        self.assertIn("$72", self.text)      # cost per dev-month at 8
        self.assertIn("30.1", self.text)     # tps at 8

    def test_timeline_and_pod_cost(self):
        self.assertIn("generation", self.text)
        self.assertIn("$0.41", self.text)

    def test_render_does_not_raise(self):
        # Patch BOTH rich names: on machines without rich, Table is None, and patching
        # only Console steered render() down the rich path into `None(...)` — exactly
        # how the first CI run (no rich installed) failed.
        from unittest.mock import patch, MagicMock
        from scripts.lib.report_terminal import render
        with patch("scripts.lib.report_terminal.Console", MagicMock()), \
             patch("scripts.lib.report_terminal.Table", MagicMock()):
            render(_report())

    def test_render_falls_back_to_plain_without_rich(self):
        from unittest.mock import patch
        from scripts.lib.report_terminal import render
        lines = []
        with patch("scripts.lib.report_terminal.Console", None):
            render(_report(), out=lines.append)
        self.assertIn("n=8", "\n".join(lines))
