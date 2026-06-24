"""Streamlit + Plotly dashboard: benchmark → team size (N) → model×hardware. For a benchmark, compare
hardware at a chosen N (holds SLO? · $/dev-month) and drill into a run's per-N results, timeline
& cost, and failing agents. Local only — deps live in requirements-report.txt.

Run: make dashboard   (or: streamlit run scripts/dashboard.py)
"""

import json
import pathlib
import sys

# Streamlit Cloud (and a bare `streamlit run scripts/dashboard.py`) put scripts/ on
# sys.path, not the repo root, so the `scripts.*` imports below would fail there.
# (`make dashboard` uses `python -m streamlit`, which masks this by adding the cwd.)
_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.lib.bench_report import HOURS_PER_MONTH  # noqa: E402  (needs the path fix above)


def load_runs(results_dir="results"):
    """All results/**/report.json as dicts, sorted by path (pure; bad/missing files skipped)."""
    out = []
    for p in sorted(pathlib.Path(results_dir).glob("**/report.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            pass
    return out


PREFLIGHT = {"smoke"}   # pre-flight/validate benchmarks — hidden from the dashboard


def benchmarks(reports):
    """Distinct benchmark names, sorted, excluding pre-flight ones (skips falsy)."""
    return sorted({r.get("benchmark") for r in reports
                   if r.get("benchmark") and r.get("benchmark") not in PREFLIGHT})


def devs_for(reports, benchmark):
    """Sorted distinct N tested for a benchmark (across all its runs' by_devs)."""
    ns = set()
    for r in reports:
        if r.get("benchmark") == benchmark:
            for d in r.get("by_devs", []):
                ns.add(d["devs"])
    return sorted(ns)


def run_key(rep):
    """Stable id for a run: (family, quant, hardware, gpu_count)."""
    return (rep.get("family"), rep.get("quant"), rep.get("hardware"), rep.get("gpu_count"))


def combo_rows(reports, benchmark, n):
    """One row per model×hardware combo (latest run wins) that ran this benchmark and tested this N —
    sorted by cost_per_dev_month asc (None last). reports are path-sorted, so last wins."""
    by_key = {}
    for r in reports:
        if r.get("benchmark") != benchmark:
            continue
        entry = next((d for d in r.get("by_devs", []) if d["devs"] == n), None)
        if entry is None:
            continue
        by_key[run_key(r)] = {
            "family": r.get("family"), "quant": r.get("quant"),
            "hardware": r.get("hardware"), "gpus": r.get("gpu_count"),
            "holds_slo": entry["holds_slo"], "cost_per_dev_month": entry["cost_per_dev_month"],
            "median_tps": entry["median_tps"], "median_ttft": entry["median_ttft"],
            "valid": entry["valid"], "agents_ok": entry["agents_ok"],
            "tokens_per_dev": entry["tokens_per_dev"], "pod_cost_usd": r.get("pod_cost_usd"),
            "run_id": r.get("run_id"),
            "min_tps": (r.get("slo") or {}).get("min_tps"),
            "max_ttft": (r.get("slo") or {}).get("max_ttft"),
            "cost_per_hour": r.get("cost_per_hour"),
            "price_usd_per_gpu_hour": r.get("price_usd_per_gpu_hour"),
        }
    rows = list(by_key.values())
    rows.sort(key=lambda x: (x["cost_per_dev_month"] is None, x["cost_per_dev_month"] or 0))
    return rows


def combo_series(reports, benchmark):
    """One entry per model×hardware combo with its points sorted by devs — for the cross-N
    charts. N measurements are independent, so points MERGE across a combo's runs keyed by
    (combo, N): a new run measuring only new team sizes extends the curve instead of
    erasing the old Ns; the same N measured again takes the later run (path-sorted)."""
    by_key = {}
    for r in reports:
        if r.get("benchmark") != benchmark:
            continue
        cur = by_key.setdefault(run_key(r), {"rep": r, "by_n": {}})
        cur["rep"] = r   # latest run's metadata (price, ...) wins
        for d in r.get("by_devs", []):
            cur["by_n"][d["devs"]] = d
    out = []
    for cur in by_key.values():
        r = cur["rep"]
        pts = [{"devs": d["devs"], "cost_per_dev_month": d["cost_per_dev_month"],
                "median_tps": d["median_tps"], "median_ttft": d.get("median_ttft"),
                "holds_slo": d["holds_slo"]}
               for _, d in sorted(cur["by_n"].items())]
        x = {"family": r.get("family"), "quant": r.get("quant"),
             "hardware": r.get("hardware"), "gpus": r.get("gpu_count")}
        out.append({**x, "label": _combo_label(x), "points": pts,
                    "price_usd_per_gpu_hour": r.get("price_usd_per_gpu_hour")})
    out.sort(key=lambda s: s["label"])
    return out


def benchmark_min_tps(reports, benchmark):
    """The SLO floor (slo.min_tps) for a benchmark, from any of its reports; None if absent."""
    for r in reports:
        if r.get("benchmark") == benchmark and r.get("slo"):
            return r["slo"].get("min_tps")
    return None


def benchmark_max_ttft(reports, benchmark):
    """The SLO TTFT ceiling (slo.max_ttft) for a benchmark; None if absent."""
    for r in reports:
        if r.get("benchmark") == benchmark and r.get("slo"):
            return r["slo"].get("max_ttft")
    return None


def slo_headroom(median_tps, min_tps, median_ttft, max_ttft):
    """Room before SLO breaks: tps_ratio=median_tps/min_tps, ttft_ratio=max_ttft/median_ttft
    (each >1 ⇒ headroom). None when a needed value is missing/zero."""
    tps_ratio = (median_tps / min_tps) if (median_tps is not None and min_tps) else None
    ttft_ratio = (max_ttft / median_ttft) if (max_ttft is not None and median_ttft) else None
    return {"tps_ratio": tps_ratio, "ttft_ratio": ttft_ratio}


def self_host_usd_per_mtok(cost_per_hour, median_tps, n):
    """Self-host $/M decode-tokens = cost_per_hour ÷ (median_tps × n × 3600 / 1e6). None if any 0."""
    if not cost_per_hour or not median_tps or not n:
        return None
    return cost_per_hour * 1e6 / (median_tps * n * 3600)


def best_combo_from_series(series):
    """Cheapest (combo, N) point that holds SLO: {label, devs, cost_per_dev_month}; None if
    nothing holds. Takes the series so what-if repricing flows through."""
    best = None
    for s in series:
        for p in s["points"]:
            if p["holds_slo"] and p["cost_per_dev_month"] is not None:
                if best is None or p["cost_per_dev_month"] < best["cost_per_dev_month"]:
                    best = {"label": s["label"], "devs": p["devs"],
                            "cost_per_dev_month": p["cost_per_dev_month"]}
    return best


def best_combo(reports, benchmark):
    """Cheapest (combo, N) that holds SLO for a benchmark, at the configs' list prices."""
    return best_combo_from_series(combo_series(reports, benchmark))


def reprice_series(series, price_overrides):
    """What-if pricing: recompute each point's $/dev-month from an overridden $/GPU-hour,
    keyed by hardware. SLO outcomes are measured and price-independent, so they never
    change. Pure — returns new dicts, inputs untouched."""
    out = []
    for s in series:
        if s["hardware"] not in price_overrides or not s.get("gpus"):
            out.append(s)
            continue
        price = price_overrides[s["hardware"]]
        pts = [{**p, "cost_per_dev_month": price * s["gpus"] * HOURS_PER_MONTH / p["devs"]}
               for p in s["points"]]
        out.append({**s, "price_usd_per_gpu_hour": price, "points": pts})
    return out


def reprice_rows(rows, price_overrides, n):
    """What-if pricing for the at-N rows: recompute $/dev-month and cost_per_hour from the
    override, then re-sort cheapest first. Pure."""
    out = []
    for x in rows:
        if x["hardware"] in price_overrides and x.get("gpus") and n:
            price = price_overrides[x["hardware"]]
            cph = price * x["gpus"]
            out.append({**x, "price_usd_per_gpu_hour": price, "cost_per_hour": cph,
                        "cost_per_dev_month": cph * HOURS_PER_MONTH / n})
        else:
            out.append(dict(x))
    out.sort(key=lambda x: (x["cost_per_dev_month"] is None, x["cost_per_dev_month"] or 0))
    return out


def _combo_label(x):
    """Human label for a model×hardware combo row."""
    return f"{x['family']}-{x['quant']} × {x['hardware']}-{x['gpus']}gpu"


def compare_narrative(rows, n):
    """Insight sentences over the model×hardware combos at N: cheapest that holds SLO + any that don't."""
    if not rows:
        return []
    out = []
    holding = [x for x in rows if x.get("holds_slo") and x.get("cost_per_dev_month") is not None]
    if holding:
        c = min(holding, key=lambda x: x["cost_per_dev_month"])
        out.append(f"Cheapest combo that holds SLO at {n} devs: **{_combo_label(c)}** "
                   f"at ${c['cost_per_dev_month']:.0f}/dev-month.")
    else:
        out.append(f"No model×hardware combo holds the SLO at {n} devs.")
    failing = [_combo_label(x) for x in rows if not x.get("holds_slo")]
    if failing and holding:   # if nothing holds, the sentence above already says so
        out.append(f"Does not hold at {n} devs: {', '.join(failing)}.")
    return out


# --- chart styling -------------------------------------------------------------------
# Validated categorical palette (CVD-checked: worst adjacent ΔE 24.2, all slots in the
# lightness band). The slot ORDER is the colorblind-safety mechanism — assign in order,
# never cycle, and keep a combo's color stable across charts (color follows the entity).
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
               "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
# Status colors are reserved for state (holds/fails SLO) and are distinct from the
# categorical slots; they never appear color-alone (paired with ✓/✗ text + bar pattern).
STATUS = {"good": "#0ca30c", "critical": "#d03b3b"}
INK = {"secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
       "surface": "#fcfcfb"}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def assign_series_colors(labels):
    """Fixed color per combo label: sorted labels take palette slots in order, so a combo
    keeps its hue across charts, reruns, and filter changes — never repainted by rank."""
    return {lab: CATEGORICAL[i % len(CATEGORICAL)]
            for i, lab in enumerate(sorted(set(labels)))}


def _style(fig, *, x_title, y_title):
    """Shared chart chrome: recessive grid/axes, muted ticks, horizontal legend above the
    plot. Chart titles live OUTSIDE plotly (st.markdown above each chart) so the legend
    never collides with them and titles reflow on narrow/mobile viewports."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=INK["surface"],
        font=dict(family=FONT, color=INK["secondary"], size=13),
        xaxis=dict(title=x_title, gridcolor=INK["grid"], linecolor=INK["axis"],
                   tickfont=dict(color=INK["muted"]), zeroline=False),
        yaxis=dict(title=y_title, gridcolor=INK["grid"], linecolor=INK["axis"],
                   tickfont=dict(color=INK["muted"]), zeroline=False, rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=40, r=88, b=48, l=56))
    return fig


def _curve_fig(series, colors, value_key, y_title, hover_fmt):
    """One curve per combo across team sizes. 2px lines, ≥8px markers (x = point fails
    SLO — identity never rides on color alone), direct labels when ≤4 series (text in ink
    tokens; the line end carries the color)."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for s in series:
        xs = [p["devs"] for p in s["points"]]
        ys = [p[value_key] for p in s["points"]]
        symbols = ["circle" if p["holds_slo"] else "x" for p in s["points"]]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=s["label"],
            line=dict(color=colors[s["label"]], width=2),
            marker=dict(size=9, symbol=symbols, color=colors[s["label"]],
                        line=dict(width=1, color=INK["surface"])),
            hovertemplate=f"%{{fullData.name}}<br>N=%{{x}} · {hover_fmt}<extra></extra>"))
    if len(series) <= 4:
        for s in series:
            if s["points"]:
                # short label: full combo names overflow the right edge on phones
                fig.add_annotation(x=s["points"][-1]["devs"], y=s["points"][-1][value_key],
                                   text=s.get("short", s["label"]), showarrow=False,
                                   xanchor="left", xshift=10,
                                   font=dict(size=11, color=INK["secondary"]))
    _style(fig, x_title="developers (N)", y_title=y_title)
    fig.update_layout(height=340)
    fig.update_xaxes(tickmode="array",
                     tickvals=sorted({p["devs"] for s in series for p in s["points"]}))
    return fig


def short_series_labels(series):
    """Compact per-combo labels for direct labels on small screens (hardware + quant, e.g.
    'l40s-1gpu fp8'); falls back to the full label whenever the short form is ambiguous."""
    shorts = {s["label"]: f"{s['hardware']}-{s['gpus']}gpu {s['quant']}" for s in series}
    if len(set(shorts.values())) < len(shorts):
        return {lab: lab for lab in shorts}
    return shorts


# Repopulated on every script run by main() before st.navigation().run(), so the page
# functions below can reach the loaded reports and the Page objects they navigate to.
_PAGES = {}


def render_detail(rep):
    """One run's detail: per-N results, timeline & cost, failing agents."""
    import streamlit as st
    import plotly.graph_objects as go

    c1, c2, c3 = st.columns(3)
    c1.metric("Measured pod cost", f"${rep.get('pod_cost_usd', 0):.2f}",
              help="pod wall-clock × the hardware's list price")
    c2.metric("GPU price", f"${rep.get('price_usd_per_gpu_hour', 0):.2f}/h × {rep.get('gpu_count', 1)}")
    c3.metric("Date", rep.get("date") or "—")

    st.dataframe([{"devs": d["devs"], "SLO": "✓ holds" if d["holds_slo"] else "✗ fails",
                   "$/dev-month": d.get("cost_per_dev_month"), "tok/s": d.get("median_tps"),
                   "TTFT (s)": d.get("median_ttft"), "valid": d.get("valid"),
                   "gates ok": f"{d.get('agents_ok')}/{d['devs']}"}
                  for d in rep.get("by_devs", [])], hide_index=True, width="stretch",
                 column_config={
                     "$/dev-month": st.column_config.NumberColumn(format="$%.0f"),
                     "tok/s": st.column_config.NumberColumn(format="%.1f"),
                     "TTFT (s)": st.column_config.NumberColumn(format="%.2f"),
                 })

    timeline = rep.get("timeline")
    if timeline:
        st.subheader("Where the pod time (and money) went")
        steps = [r.get("label") for r in timeline]
        secs = [round(r.get("seconds", 0)) for r in timeline]
        bar = go.Figure(go.Bar(
            y=steps[::-1], x=secs[::-1], orientation="h",
            marker=dict(color=CATEGORICAL[0], cornerradius=4),
            text=[f"{s}s · ${r.get('cost_usd', 0):.2f}" for s, r in zip(secs, timeline, strict=True)][::-1],
            textposition="outside", textfont=dict(color=INK["secondary"]),
            hovertemplate="%{y}: %{x}s<extra></extra>"))
        _style(bar, x_title="seconds", y_title="")
        bar.update_layout(showlegend=False, margin=dict(t=16), height=90 + 36 * len(steps))
        st.plotly_chart(bar, width="stretch")
        st.caption("Gate scoring runs after the pod is terminated, so it never appears here.")

    st.subheader("Failures")
    any_fail = False
    for d in rep.get("by_devs", []):
        for a in d.get("agents", []):
            for g in a.get("gates", []):
                if not g.get("passed", True):
                    any_fail = True
                    with st.expander(f"n={d['devs']} · agent{a['agent']} · {g['id']}"):
                        st.code(g.get("output", "(no captured output)"))
    if not any_fail:
        st.success("No gate failures in this run.")


def overview():
    import streamlit as st
    st.title("Developer AI Lab — benchmarks")
    reports = _PAGES["reports"]
    bs = benchmarks(reports)
    if not bs:
        st.info("No runs yet — run a benchmark first.")
        return
    rows = []
    for b in bs:
        best = best_combo(reports, b)
        rows.append({"benchmark": b,
                     "team sizes measured": ", ".join(str(n) for n in devs_for(reports, b)),
                     "best deal (holds SLO)": (f"{best['label']} — ${best['cost_per_dev_month']:.0f}/dev-mo "
                                               f"@ {best['devs']} devs") if best else "—"})
    st.dataframe(rows, hide_index=True)
    st.caption("Pick a benchmark in the sidebar → choose a team size → compare model × hardware.")


def benchmark_page(benchmark):
    import streamlit as st
    import plotly.graph_objects as go
    st.title(benchmark)
    reports = _PAGES["reports"]
    ns = devs_for(reports, benchmark)
    if not ns:
        st.info("No runs for this benchmark yet.")
        return
    series = combo_series(reports, benchmark)
    floor = benchmark_min_tps(reports, benchmark)
    ttft_cap = benchmark_max_ttft(reports, benchmark)

    # 0 — what-if pricing: every $ figure on the page recomputes from the visitor's prices
    hw_prices = {}
    for s in series:
        if s.get("price_usd_per_gpu_hour") is not None:
            hw_prices.setdefault(s["hardware"], s["price_usd_per_gpu_hour"])
    overrides = {}
    if hw_prices:
        with st.expander("What-if: use your own GPU prices ($/GPU-hour)"):
            st.caption("Every $ figure on this page recomputes live. SLO results are "
                       "measured and do not change with price.")
            cols = st.columns(min(len(hw_prices), 3))
            for i, (hw, price) in enumerate(sorted(hw_prices.items())):
                val = cols[i % len(cols)].number_input(
                    hw, min_value=0.0, value=float(price), step=0.05, format="%.2f",
                    key=f"price-{benchmark}-{hw}")
                if abs(val - float(price)) > 1e-9:
                    overrides[hw] = val
    if overrides:
        series = reprice_series(series, overrides)

    # 1 — the benchmark-level takeaway (independent of any selector)
    best = best_combo_from_series(series)
    tag = " — at your what-if prices" if overrides else ""
    if best:
        st.markdown(f"**Best deal measured:** {best['label']} at "
                    f"**${best['cost_per_dev_month']:.0f}/dev-month** @ {best['devs']} devs "
                    f"(holds the SLO{f': ≥ {floor:.0f} tok/s' if floor is not None else ''}){tag}.")
    else:
        st.markdown("**No measured combo holds the SLO yet.**")

    # 2 — the analysis charts: every measured N (the selector below does NOT affect these)
    st.subheader("Analysis across team sizes")
    colors = assign_series_colors([s["label"] for s in series])
    shorts = short_series_labels(series)
    for s in series:
        s["short"] = shorts[s["label"]]
    col_cost, col_tps, col_ttft = st.columns(3)   # stack vertically on narrow viewports
    col_cost.markdown("**Cost per developer** — falls as the GPU amortizes")
    cost_fig = _curve_fig(series, colors, "cost_per_dev_month", "$/dev-month",
                          "$%{y:.0f}/dev-month")
    col_cost.plotly_chart(cost_fig, width="stretch")

    col_tps.markdown("**Throughput per stream** — ✕ marks a broken SLO")
    tps_fig = _curve_fig(series, colors, "median_tps", "median tok/s per stream",
                         "%{y:.1f} tok/s")
    if floor is not None:
        tps_fig.add_hline(y=floor, line_dash="dot", line_color=STATUS["critical"],
                          annotation_text=f"SLO floor {floor:.0f} tok/s",
                          annotation_font_color=INK["secondary"])
    col_tps.plotly_chart(tps_fig, width="stretch")

    col_ttft.markdown("**Time to first token** — must stay under the cap")
    ttft_fig = _curve_fig(series, colors, "median_ttft", "median TTFT (s)", "%{y:.2f}s")
    if ttft_cap is not None:
        ttft_fig.add_hline(y=ttft_cap, line_dash="dot", line_color=STATUS["critical"],
                           annotation_text=f"SLO cap {ttft_cap:.1f}s",
                           annotation_font_color=INK["secondary"])
    col_ttft.plotly_chart(ttft_fig, width="stretch")

    # 3 — drill into one team size: the selector sits WITH the ranking + table it controls
    st.divider()
    st.subheader("Drill into one team size")
    n = st.selectbox("Developers (N)", ns, index=len(ns) - 1)
    rows = combo_rows(reports, benchmark, n)
    if not rows:
        st.info(f"No model × hardware measured at {n} devs.")
        return
    if overrides:
        rows = reprice_rows(rows, overrides, n)
    for line in compare_narrative(rows, n):
        st.markdown(f"- {line}")

    st.markdown(f"##### Ranking at {n} developers — cheapest that holds SLO wins")
    # Horizontal bars: long combo names read naturally and never collide on phones.
    # Reversed so the cheapest sits on top.
    rrows = rows[::-1]
    labels_bar = [_combo_label(x) for x in rrows]
    vals = [x.get("cost_per_dev_month") for x in rrows]
    bar = go.Figure(go.Bar(
        y=labels_bar, x=vals, orientation="h",
        marker=dict(color=[STATUS["good"] if x["holds_slo"] else STATUS["critical"] for x in rrows],
                    cornerradius=4,
                    pattern=dict(shape=["" if x["holds_slo"] else "/" for x in rrows])),
        text=[(f"${v:.0f}" if v is not None else "—") + ("" if x["holds_slo"] else " ✗ SLO")
              for v, x in zip(vals, rrows, strict=True)],
        textposition="outside", textfont=dict(color=INK["secondary"]),
        hovertemplate="%{y}<br>$%{x:.0f}/dev-month<extra></extra>"))
    _style(bar, x_title="$/dev-month", y_title="")
    bar.update_layout(showlegend=False, margin=dict(t=16, l=8),
                      height=110 + 44 * len(rrows), yaxis=dict(gridcolor="rgba(0,0,0,0)"))
    st.plotly_chart(bar, width="stretch")
    st.caption("Solid green = holds the SLO · hatched red ✗ = does not (not an option, just cheap).")

    # 4 — the full table; selecting a row opens that run's detail
    st.markdown(f"##### All numbers at {n} developers")
    table = []
    for x in rows:
        hr = slo_headroom(x["median_tps"], x["min_tps"], x["median_ttft"], x["max_ttft"])
        mtok = self_host_usd_per_mtok(x["cost_per_hour"], x["median_tps"], n)
        table.append({
            "model × hardware": _combo_label(x),
            "SLO": "✓ holds" if x["holds_slo"] else "✗ fails",
            "$/dev-month": round(x["cost_per_dev_month"]) if x["cost_per_dev_month"] is not None else None,
            "tok/s": round(x["median_tps"]) if x["median_tps"] is not None else None,
            "TTFT (s)": x["median_ttft"],
            "headroom": f"{hr['tps_ratio']:.1f}×" if hr["tps_ratio"] is not None else "—",
            "$/Mtok": round(mtok, 2) if mtok is not None else None,
            "tok/dev": x["tokens_per_dev"],
        })
    tkey = f"combos-{benchmark}-{n}"
    event = st.dataframe(
        table, hide_index=True, width="stretch", key=tkey,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "$/dev-month": st.column_config.NumberColumn(format="$%d",
                help="GPU list price amortized across the N developers"),
            "tok/s": st.column_config.NumberColumn(help="median per-stream decode throughput"),
            "TTFT (s)": st.column_config.NumberColumn(format="%.2f",
                help="median time to first token (SLO caps it)"),
            "headroom": st.column_config.TextColumn(help="median tok/s ÷ SLO floor (>1× = margin)"),
            "$/Mtok": st.column_config.NumberColumn(format="$%.2f",
                help="self-host cost per million decode tokens at this N"),
            "tok/dev": st.column_config.NumberColumn(help="median tokens one developer's task used"),
        })
    st.caption("Select a row to open the full run detail (timeline & cost, gate outcomes).")
    sel = event.selection.rows if event and event.selection else []
    if sel:
        x = rows[sel[0]]
        st.session_state["selected_run"] = run_key({"family": x["family"], "quant": x["quant"],
                                                    "hardware": x["hardware"], "gpu_count": x["gpus"]})
        try:
            del st.session_state[tkey]   # clear the selection so Back doesn't bounce here
        except KeyError:
            pass
        st.switch_page(_PAGES["detail"])


def run_detail():
    import streamlit as st
    reports = _PAGES["reports"]
    key = st.session_state.get("selected_run")
    rep = next((r for r in reversed(reports) if run_key(r) == key), None) if key else None
    if rep is None:
        st.info("Pick a model × hardware row from a benchmark page to see its run detail.")
        return
    benchmark = rep.get("benchmark")
    if st.button(f"← Back to {benchmark}"):
        back = _PAGES["benchmarks"].get(benchmark)
        if back is not None:
            st.switch_page(back)
    st.title(f"{rep.get('family')}/{rep.get('quant')} · {rep.get('hardware')} · {rep.get('gpu_count')}gpu")
    st.caption(f"{rep.get('benchmark')} · run {rep.get('run_id', '?')}")
    render_detail(rep)


def main():
    import streamlit as st
    st.set_page_config(page_title="Developer AI Lab — benchmarks", layout="wide")
    reports = load_runs()

    pages = [st.Page(overview, title="Overview", icon="📊", default=True)]
    benchmark_pages = {}
    for b in benchmarks(reports):
        pg = st.Page((lambda name=b: benchmark_page(name)), title=b, url_path=b)
        benchmark_pages[b] = pg
        pages.append(pg)
    detail = st.Page(run_detail, title="Run detail", url_path="run-detail", visibility="hidden")
    pages.append(detail)

    _PAGES.clear()
    _PAGES.update(reports=reports, benchmarks=benchmark_pages, detail=detail)
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
