"""Human-readable terminal summary of a benchmark report. Uses ``rich`` when available,
falling back to plain lines so the module (and the unit suite) never require ``rich``."""

# Lazy-import guards: rich is optional. These names are None until render() populates them,
# and the test can patch them. Absence is tolerated via the guard in render().
try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console = None
    Table = None

from scripts.lib.timeline import fmt_duration


def _dollar(v):
    return f"${v:.2f}" if isinstance(v, (int, float)) else "—"


def _render_plain(report, out=print):
    """Plain-text per-run summary via ``out`` (one line per call). Pure."""
    out(f"=== {report.get('family','?')}/{report.get('quant','?')} "
        f"× {report.get('hardware','?')} × {report.get('gpu_count','?')}gpu ===")
    out(f"  {'n':>4}  {'holds':>5}  {'$/dev-mo':>9}  {'tps':>6}  {'ttft':>6}  valid  gates")
    for d in report.get("by_devs", []):
        cost = "${:.0f}".format(d["cost_per_dev_month"])
        out(f"  n={d['devs']:<3}  {('yes' if d['holds_slo'] else 'NO'):>5}  "
            f"{cost:>9}  {d['median_tps']:>6.1f}  "
            f"{d['median_ttft']:>5.2f}s  {('ok' if d['valid'] else 'BAD'):>5}  "
            f"{d['agents_ok']}/{d['devs']}")
    timeline = report.get("timeline")
    if timeline:
        out("")
        out("Timeline (billed pod time):")
        for r in timeline:
            seg_cost = "${:.2f}".format(r.get("cost_usd", 0))
            out(f"  {r.get('label','?'):<14} {fmt_duration(r.get('seconds',0)):>8} "
                f"{seg_cost:>8}")
        out(f"  pod cost = {_dollar(report.get('pod_cost_usd'))}")


def render(report, out=print):
    if Console is None or Table is None:   # rich absent (or partially patched) -> plain
        return _render_plain(report, out=out)
    console = Console()
    console.rule(f"[bold]{report.get('family','?')}/{report.get('quant','?')} "
                 f"× {report.get('hardware','?')} × {report.get('gpu_count','?')}gpu")
    table = Table("devs", "holds", "$/dev-mo", "tps", "ttft", "valid", "gates")
    for d in report.get("by_devs", []):
        table.add_row(str(d["devs"]), "yes" if d["holds_slo"] else "[red]NO[/]",
                      f"${d['cost_per_dev_month']:.0f}", f"{d['median_tps']:.1f}",
                      f"{d['median_ttft']:.2f}s", "ok" if d["valid"] else "[red]BAD[/]",
                      f"{d['agents_ok']}/{d['devs']}")
    console.print(table)
    timeline = report.get("timeline")
    if timeline:
        tt = Table("step", "dur", "$", title="Timeline (billed pod time)")
        for r in timeline:
            tt.add_row(r.get("label", "?"), fmt_duration(r.get("seconds", 0)),
                       f"${r.get('cost_usd', 0):.2f}")
        console.print(tt)
        console.print(f"pod cost = [bold]{_dollar(report.get('pod_cost_usd'))}[/]")
