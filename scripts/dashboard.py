"""Streamlit + Plotly dashboard: benchmark → team size (N) → model×hardware. For a benchmark, compare
hardware at a chosen N (holds SLO? · $/dev-month) and drill into a run's per-N results, timeline
& cost, and failing agents. Local only — deps live in requirements-report.txt.

Run: make dashboard   (or: streamlit run scripts/dashboard.py)
"""

import json
import pathlib
import sys
import urllib.parse

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


def families(reports, benchmark):
    """Distinct model families measured for a benchmark, sorted — populates the family filter."""
    return sorted({r.get("family") for r in reports
                   if r.get("benchmark") == benchmark and r.get("family")})


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


# Numeric per-N metrics that average across repeat runs (with a CV). Everything else in
# an entry is taken from the latest run; boolean verdicts are conservative (all must hold).
_AGG_MEAN_KEYS = ("cost_per_dev_month", "median_tps", "median_ttft", "p90_ttft", "p90_tps",
                  "p99_ttft", "p99_tps", "e2e_p50", "e2e_p99", "cost_per_mtok",
                  "cost_per_mtok_output", "tokens_per_hour", "computed_tokens_per_hour",
                  "tokens_per_dev", "prefix_cache_hit_rate")
_AGG_ALL_KEYS = ("holds_slo", "holds_slo_p90", "valid")


def aggregate_entries(entries):
    """Aggregate repeat measurements of the same (combo, N): numeric metrics average and
    carry a sample-CV (the stability protocol of arXiv:2606.11690 §5.8 — the CV exposes
    runs that disagree), boolean verdicts hold only if EVERY repeat holds, and anything
    else comes from the latest run. Pure; entries are in run order (latest last)."""
    import statistics
    out = dict(entries[-1])
    out["repeats"] = len(entries)
    cv = {}
    for k in _AGG_MEAN_KEYS:
        vals = [e[k] for e in entries if e.get(k) is not None]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        out[k] = mean
        if len(vals) >= 2 and mean:
            cv[k] = statistics.stdev(vals) / mean
    for k in _AGG_ALL_KEYS:
        vals = [e[k] for e in entries if e.get(k) is not None]
        if vals:
            out[k] = all(vals)
    if cv:
        out["cv"] = cv
    return out


def combo_rows(reports, benchmark, n):
    """One row per model×hardware combo that ran this benchmark and tested this N —
    sorted by cost_per_dev_month asc (None last). Repeat measurements of the N aggregate
    (mean + CV, conservative SLO); run-level metadata comes from the latest run."""
    grouped = {}
    for r in reports:
        if r.get("benchmark") != benchmark:
            continue
        found = next((d for d in r.get("by_devs", []) if d["devs"] == n), None)
        if found is None:
            continue
        cur = grouped.setdefault(run_key(r), {"rep": r, "entries": []})
        cur["rep"] = r
        cur["entries"].append(found)
    by_key = {}
    for key, cur in grouped.items():
        entry, r = aggregate_entries(cur["entries"]), cur["rep"]
        by_key[key] = {
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
            # Concurrency-aware metrics (arXiv:2606.11690); None on pre-paper reports.
            "cost_per_mtok": entry.get("cost_per_mtok"),
            "cost_per_mtok_output": entry.get("cost_per_mtok_output"),
            "utilization": entry.get("utilization"),
            "underutilization_penalty": entry.get("underutilization_penalty"),
            "holds_slo_p90": entry.get("holds_slo_p90"),
            "p90_tps": entry.get("p90_tps"), "p90_ttft": entry.get("p90_ttft"),
            "repeats": entry.get("repeats", 1), "cv": entry.get("cv"),
        }
    rows = list(by_key.values())
    rows.sort(key=lambda x: (x["cost_per_dev_month"] is None, x["cost_per_dev_month"] or 0))
    return rows


def combo_series(reports, benchmark):
    """One entry per model×hardware combo with its points sorted by devs — for the cross-N
    charts. N measurements are independent, so points MERGE across a combo's runs keyed by
    (combo, N): a new run measuring only new team sizes extends the curve instead of
    erasing the old Ns; the same N measured again is a repeat and aggregates (mean + CV,
    see aggregate_entries)."""
    by_key = {}
    for r in reports:
        if r.get("benchmark") != benchmark:
            continue
        cur = by_key.setdefault(run_key(r), {"rep": r, "by_n": {}})
        cur["rep"] = r   # latest run's metadata (price, ...) wins
        for d in r.get("by_devs", []):
            cur["by_n"].setdefault(d["devs"], []).append(d)
    out = []
    for cur in by_key.values():
        r = cur["rep"]
        agg = {n: aggregate_entries(entries) for n, entries in cur["by_n"].items()}
        pts = [{"devs": d["devs"], "cost_per_dev_month": d["cost_per_dev_month"],
                "median_tps": d["median_tps"], "median_ttft": d.get("median_ttft"),
                "holds_slo": d["holds_slo"],
                # Concurrency-aware metrics (arXiv:2606.11690); None on pre-paper reports.
                "cost_per_mtok": d.get("cost_per_mtok"),
                "cost_per_mtok_output": d.get("cost_per_mtok_output"),
                "tokens_per_hour": d.get("tokens_per_hour"),
                "computed_tokens_per_hour": d.get("computed_tokens_per_hour"),
                "tokens_source": d.get("tokens_source"),
                "utilization": None,
                "repeats": d.get("repeats", 1), "cv": d.get("cv"),
                "holds_slo_p90": d.get("holds_slo_p90")}
               for _, d in sorted(agg.items())]
        # U(N) over the combo's MERGED points, not per report: histories that measured
        # each N in its own run would otherwise show a meaningless 100% at every point.
        # A measured Θmax (saturation probe, latest run's) is the true denominator and
        # works for any number of points; without one the best measured level stands in
        # (a lower bound), which needs ≥2 points — one point is trivially its own best.
        # Numerator: compute-real throughput (cache hits discounted) when captured.
        def achieved(p):
            return p.get("computed_tokens_per_hour") or p.get("tokens_per_hour")
        theta = r.get("theta_max")
        tphs = [achieved(p) for p in pts if achieved(p)]
        basis = None
        if theta and theta.get("tokens_per_hour"):
            denominator, basis = theta["tokens_per_hour"], "theta_max"
        elif len(tphs) >= 2:
            denominator, basis = max(tphs), "best_level"
        if basis:
            for p in pts:
                if achieved(p):
                    p["utilization"] = achieved(p) / denominator
        x = {"family": r.get("family"), "quant": r.get("quant"),
             "hardware": r.get("hardware"), "gpus": r.get("gpu_count")}
        out.append({**x, "label": _combo_label(x), "points": pts,
                    "price_usd_per_gpu_hour": r.get("price_usd_per_gpu_hour"),
                    "theta_max": theta, "utilization_basis": basis,
                    "model_meta": r.get("model_meta")})
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


def _scale_mtok(d, ratio):
    """Rescale the price-derived $/MTok fields by the what-if price ratio. Utilization is
    measured (throughput vs throughput), so it never changes with price."""
    out = dict(d)
    for k in ("cost_per_mtok", "cost_per_mtok_output"):
        if out.get(k) is not None:
            out[k] = out[k] * ratio
    return out


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
        old = s.get("price_usd_per_gpu_hour")
        ratio = (price / old) if old else None
        pts = [{**(_scale_mtok(p, ratio) if ratio else p),
                "cost_per_dev_month": price * s["gpus"] * HOURS_PER_MONTH / p["devs"]}
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
            old = x.get("price_usd_per_gpu_hour")
            scaled = _scale_mtok(x, price / old) if old else dict(x)
            out.append({**scaled, "price_usd_per_gpu_hour": price, "cost_per_hour": cph,
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
    SLO — identity never rides on color alone), direct labels when ≤4 series AND their
    line ends don't collide (overlapping labels are worse than none — the legend always
    carries identity)."""
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
    ends = [(s["points"][-1][value_key], s) for s in series
            if s["points"] and s["points"][-1].get(value_key) is not None]
    all_vals = [p[value_key] for s in series for p in s["points"]
                if p.get(value_key) is not None]
    span = (max(all_vals) - min(all_vals)) if all_vals else 0.0
    # Direct labels only for 1-2 series: with more, reference lines (SLO caps/floors)
    # compress the pixel space and end labels collide — the legend carries identity.
    if len(series) <= 2 and span > 0:
        last_labeled = None
        for y_end, s in sorted(ends, key=lambda t: t[0]):
            # ≥6% of the chart's value span between labels, or they overprint.
            if last_labeled is not None and (y_end - last_labeled) < 0.06 * span:
                continue
            last_labeled = y_end
            # short label: full combo names overflow the right edge on phones
            fig.add_annotation(x=s["points"][-1]["devs"], y=y_end,
                               text=s.get("short", s["label"]), showarrow=False,
                               xanchor="left", xshift=10,
                               font=dict(size=11, color=INK["secondary"]))
    _style(fig, x_title="developers (N)", y_title=y_title)
    fig.update_layout(height=340)
    fig.update_xaxes(tickmode="array",
                     tickvals=sorted({p["devs"] for s in series for p in s["points"]}))
    return fig


def _log_ticks(lo, hi):
    """1-2-5 ticks covering [lo, hi] (money-friendly log scale)."""
    import math
    ticks, exp = [], math.floor(math.log10(lo))
    while 10 ** exp <= hi:
        for m in (1, 2, 5):
            v = m * 10 ** exp
            if lo <= v <= hi:
                ticks.append(v)
        exp += 1
    return ticks or [lo, hi]


def _loglog(fig, series, value_key, extra_values=()):
    """Deterministic log-log axes: explicit ranges and 1-2-5 ticks. Plotly's autorange
    on log axes inherits the linear rangemode and blows up to decades of empty space
    (seen live 2026-07-04: a $0.01-$4 chart scaled to 10k). extra_values (e.g. API
    reference lines) are kept inside the visible range."""
    import math
    xs = [p["devs"] for s in series for p in s["points"]]
    ys = [p[value_key] for s in series for p in s["points"]
          if p.get(value_key) is not None]
    ys += [v for v in extra_values if v and v > 0]
    if not xs or not ys or min(ys) <= 0:
        return fig
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    fig.update_xaxes(type="log", tickmode="array",
                     tickvals=sorted(set(xs)),
                     range=[math.log10(x_lo) - 0.05, math.log10(x_hi) + 0.35])
    fig.update_yaxes(type="log", rangemode="normal", tickmode="array",
                     tickvals=_log_ticks(y_lo * 0.5, y_hi * 2.5),
                     range=[math.log10(y_lo) - 0.25, math.log10(y_hi) + 0.35])
    return fig


def series_with_metric(series, key):
    """Series restricted to points where `key` was measured (dropping empty series) — the
    concurrency-aware charts only draw runs that carry the paper metrics. Pure."""
    out = []
    for s in series:
        pts = [p for p in s["points"] if p.get(key) is not None]
        if pts:
            out.append({**s, "points": pts})
    return out


def mtok_spread(series):
    """Per combo: the spread of the measured effective $/MTok across its team sizes —
    the paper's headline result (17.5-36.3x on identical hardware, driven by load alone).
    Combos with <2 measured points are skipped (no spread to speak of). Pure."""
    out = []
    for s in series:
        vals = [(p["cost_per_mtok"], p["devs"]) for p in s["points"]
                if p.get("cost_per_mtok") is not None]
        if len(vals) < 2:
            continue
        (max_mtok, max_n), (min_mtok, min_n) = max(vals), min(vals)
        out.append({"label": s["label"], "max_mtok": max_mtok, "max_devs": max_n,
                    "min_mtok": min_mtok, "min_devs": min_n,
                    "spread": max_mtok / min_mtok if min_mtok else None})
    return out


def quant_impact(series):
    """Quantization impact: pairs of series that differ in NOTHING but the quant (same
    family, hardware, and GPU count), compared at their common measured team sizes —
    the only comparison that isolates quantization from the GPU (arXiv:2606.11690 finds
    the FP8 gain is architecture-dependent: ~+31% dense vs +69-74% MoE). Series measured
    on different hardware are never paired. Returns [] when no valid pair exists. Pure."""
    import itertools
    by_cfg = {}
    for s in series:
        by_cfg.setdefault((s["family"], s["hardware"], s["gpus"]), []).append(s)
    out = []
    for (family, hardware, gpus), group in sorted(by_cfg.items()):
        for a, b in itertools.combinations(sorted(group, key=lambda s: s["quant"]), 2):
            tph_a = {p["devs"]: p["tokens_per_hour"] for p in a["points"]
                     if p.get("tokens_per_hour")}
            tph_b = {p["devs"]: p["tokens_per_hour"] for p in b["points"]
                     if p.get("tokens_per_hour")}
            common = sorted(set(tph_a) & set(tph_b))
            if not common:
                continue
            out.append({
                "family": family, "hardware": hardware, "gpus": gpus,
                "quant_a": a["quant"], "quant_b": b["quant"],
                "points": [{"devs": n, "tph_a": tph_a[n], "tph_b": tph_b[n],
                            "uplift": tph_b[n] / tph_a[n] - 1} for n in common]})
    return out


def penalty_matrix(series):
    """The underutilization penalty as a combo × N grid (Figure 3 of arXiv:2606.11690):
    each cell is that combo's measured $/MTok at N over its cheapest measured point, so
    the grid shows where the cost cliff lives. None where a combo didn't measure an N.
    Pure; returns {ns, labels, cells}."""
    with_mtok = []
    for s in series:
        vals = {p["devs"]: p["cost_per_mtok"] for p in s["points"]
                if p.get("cost_per_mtok") is not None}
        if vals:
            with_mtok.append((s["label"], vals))
    ns = sorted({n for _, vals in with_mtok for n in vals})
    cells = []
    for _, vals in with_mtok:
        floor = min(vals.values())
        cells.append([(vals[n] / floor if n in vals and floor else None) for n in ns])
    return {"ns": ns, "labels": [lab for lab, _ in with_mtok], "cells": cells}


def sla_table(series):
    """The SLO's price, per combo (arXiv:2606.11690 Table 4): the largest measured team
    size that HOLDS the SLO, the output-$/MTok at that operating point, the saturation
    floor Csat (list price over the probed Θmax output throughput — unreachable under any
    real SLO), and the premium the SLO imposes over that floor. Pure."""
    out = []
    for s in series:
        holding = [p for p in s["points"]
                   if p["holds_slo"] and p.get("cost_per_mtok_output") is not None]
        row = {"label": s["label"], "sla_devs": None, "mtok_at_sla": None,
               "c_sat": None, "premium": None}
        theta_out = (s.get("theta_max") or {}).get("output_tokens_per_hour")
        price, gpus = s.get("price_usd_per_gpu_hour"), s.get("gpus")
        if theta_out and price and gpus:
            row["c_sat"] = price * gpus / theta_out * 1e6
        if holding:
            best = max(holding, key=lambda p: p["devs"])
            row["sla_devs"] = best["devs"]
            row["mtok_at_sla"] = best["cost_per_mtok_output"]
            if row["c_sat"]:
                row["premium"] = row["mtok_at_sla"] / row["c_sat"]
        out.append(row)
    return out


def api_crossover(series, api_price):
    """Per combo: the smallest measured N where the output-token $/MTok drops to or below
    a reference API price (paper §4's crossover) — None if it never does. Pure."""
    out = []
    for s in series:
        n = None
        for p in sorted(s["points"], key=lambda p: p["devs"]):
            v = p.get("cost_per_mtok_output")
            if v is not None and v <= api_price:
                n = p["devs"]
                break
        out.append({"label": s["label"], "crossover_devs": n})
    return out


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

    has_paper = any(d.get("cost_per_mtok") is not None for d in rep.get("by_devs", []))
    rows = []
    for d in rep.get("by_devs", []):
        row = {"devs": d["devs"], "SLO": "✓ holds" if d["holds_slo"] else "✗ fails",
               "$/dev-month": d.get("cost_per_dev_month"), "tok/s": d.get("median_tps"),
               "TTFT (s)": d.get("median_ttft"), "valid": d.get("valid"),
               "gates ok": f"{d.get('agents_ok')}/{d['devs']}"}
        if has_paper:
            p90 = d.get("holds_slo_p90")
            row["SLO p90"] = "—" if p90 is None else ("✓ holds" if p90 else "✗ fails")
            row["p90 tok/s"] = d.get("p90_tps")
            row["p99 TTFT (s)"] = d.get("p99_ttft")
            row["$/Mtok"] = d.get("cost_per_mtok")
            row["$/Mtok out"] = d.get("cost_per_mtok_output")
            row["util %"] = (round(d["utilization"] * 100)
                             if d.get("utilization") is not None else None)
            row["penalty"] = d.get("underutilization_penalty")
            row["cache hit %"] = (round(d["prefix_cache_hit_rate"] * 100)
                                  if d.get("prefix_cache_hit_rate") is not None else None)
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch",
                 column_config={
                     "$/dev-month": st.column_config.NumberColumn(format="$%.0f"),
                     "tok/s": st.column_config.NumberColumn(format="%.1f"),
                     "TTFT (s)": st.column_config.NumberColumn(format="%.2f"),
                     "SLO p90": st.column_config.TextColumn(
                         help="the SLO at the p90 tail — what the median hides"),
                     "p90 tok/s": st.column_config.NumberColumn(format="%.1f",
                         help="decode tok/s at the p90-slowest token time"),
                     "p99 TTFT (s)": st.column_config.NumberColumn(format="%.2f",
                         help="tail time-to-first-token (the paper's SLA percentile)"),
                     "$/Mtok": st.column_config.NumberColumn(format="$%.2f",
                         help="measured effective $ per million served tokens (blended)"),
                     "$/Mtok out": st.column_config.NumberColumn(format="$%.2f",
                         help="all cost on generated tokens — the API-comparable figure"),
                     "util %": st.column_config.NumberColumn(format="%d%%",
                         help="token throughput vs Θmax (or the run's best level)"),
                     "penalty": st.column_config.NumberColumn(format="%.1f×",
                         help="1/U — how much a utilization-naive estimate understates "
                              "cost at this N"),
                     "cache hit %": st.column_config.NumberColumn(format="%d%%",
                         help="vLLM prefix-cache hit rate during the level (agentic "
                              "traffic shares real prefixes)"),
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


_REPO_URL = "https://github.com/dpoblacion/developer-ai-lab"


def overview():
    import streamlit as st
    st.title("Developer AI Lab")
    st.markdown(
        "**Which self-hosted GPU serves a team of N developers at SLO — and at what "
        "\\$/developer.** Real agentic coding sessions (Claude Code against vLLM), "
        "measured end to end: latency SLOs, code-quality gates, and concurrency-aware "
        "per-token economics.")
    reports = _PAGES["reports"]
    bs = benchmarks(reports)
    if not bs:
        st.info("No runs yet — run a benchmark first (`make run`).")
        return
    for b in bs:
        series = combo_series(reports, b)
        best = best_combo_from_series(series)
        ns = devs_for(reports, b)
        with st.container(border=True):
            st.subheader(b)
            c1, c2, c3 = st.columns(3)
            c1.metric("best deal (holds SLO)",
                      f"${best['cost_per_dev_month']:.0f}/dev-mo" if best else "—")
            if best:
                c1.caption(f"{best['label']} @ {best['devs']} devs")
            c2.metric("model × hardware combos", len(series))
            c3.metric("team sizes measured", ", ".join(str(n) for n in ns))
            page = _PAGES["benchmarks"].get(b)
            if page is not None:
                st.page_link(page, label=f"Explore {b} →", icon=":material/monitoring:")
    st.caption(f"Everything here is measured, never estimated — methodology in the "
               f"[repository]({_REPO_URL}) (docs/methodology.md), per-token economics "
               f"after [arXiv:2606.11690](https://arxiv.org/abs/2606.11690).")


def benchmark_page(benchmark):
    import streamlit as st
    import plotly.graph_objects as go
    st.title(benchmark)
    reports = _PAGES["reports"]
    ns = devs_for(reports, benchmark)
    if not ns:
        st.info("No runs for this benchmark yet.")
        return
    all_series = combo_series(reports, benchmark)
    floor = benchmark_min_tps(reports, benchmark)
    ttft_cap = benchmark_max_ttft(reports, benchmark)

    # Colors are assigned over the FULL combo set so a combo keeps its hue when the family
    # filter hides others (color follows the entity, never its rank).
    colors = assign_series_colors([s["label"] for s in all_series])

    # Family filter: governs every figure and table on the page. Only shown when there's
    # more than one family to choose between.
    fams = families(reports, benchmark)
    selected_fams = fams
    if len(fams) > 1:
        selected_fams = st.multiselect("Model family", fams, default=fams,
                                       key=f"fam-{benchmark}") or fams
    series = [s for s in all_series if s["family"] in selected_fams]

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
                    f"**\\${best['cost_per_dev_month']:.0f}/dev-month** @ {best['devs']} devs "
                    f"(holds the SLO{f': ≥ {floor:.0f} tok/s' if floor is not None else ''}){tag}.")
    else:
        st.markdown("**No measured combo holds the SLO yet.**")

    # 2 — the analysis charts: every measured N (the selector below does NOT affect these)
    st.subheader("Analysis across team sizes")
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

    # 2b — per-token economics (arXiv:2606.11690): only drawn for runs that measured
    # them (server-side token counters), so pre-paper reports keep the old page.
    mtok_series = series_with_metric(series, "cost_per_mtok")
    if mtok_series:
        st.divider()
        st.subheader("Per-token economics — concurrency-aware")
        sources = {p["tokens_source"] for s in mtok_series for p in s["points"]
                   if p.get("tokens_source")}
        provenance = {
            frozenset({"server"}): "Token counts come from vLLM's server-side counters.",
            frozenset({"client"}): "Token counts come from client-reported usage "
                                   "(what the server processed, as billed to each agent).",
        }.get(frozenset(sources),
              "Token counts come from vLLM's server-side counters where captured, "
              "otherwise from client-reported usage (each level's tokens_source says which).")
        st.caption("Methodology from [arXiv:2606.11690](https://arxiv.org/abs/2606.11690): "
                   "utilization is an *output* of the measurement, never an assumed input. "
                   + provenance)

        # Cost spread across load (the paper's Figure 1 takeaway), as a table.
        st.markdown("**Cost spread across team sizes** — same hardware; the spread is "
                    "pure load")
        st.dataframe(
            [{"model × hardware": sp["label"],
              "costliest": f"${sp['max_mtok']:.3f}/MTok @ N={sp['max_devs']}",
              "cheapest": f"${sp['min_mtok']:.3f}/MTok @ N={sp['min_devs']}",
              "spread": f"{sp['spread']:.1f}×"}
             for sp in mtok_spread(mtok_series)], hide_index=True, width="stretch")

        col_out, col_mtok, col_util = st.columns(3)
        out_series = series_with_metric(series, "cost_per_mtok_output")
        col_out.markdown("**$/MTok, output tokens — vs API prices** (the paper's "
                         "canonical Ceff, Fig. 5; log-log)")
        tiers_raw = col_out.text_input(
            "Reference API prices ($/MTok output, comma-separated)", value="4",
            key=f"api-{benchmark}",
            help="Per-token APIs bill output 5-6× input; self-hosting is indifferent. "
                 "One dashed line per price (the paper draws three vendor tiers).")
        tiers = sorted({float(t) for t in tiers_raw.replace(";", ",").split(",")
                        if t.strip().replace(".", "", 1).isdigit() and float(t) > 0})
        out_fig = _curve_fig(out_series, colors, "cost_per_mtok_output",
                             "measured $/MTok (output only)", "$%{y:.2f}/MTok out")
        _loglog(out_fig, out_series, "cost_per_mtok_output", extra_values=tiers)
        for tier in tiers:
            out_fig.add_hline(y=tier, line_dash="dot", line_color=INK["secondary"],
                              annotation_text=f"API ${tier:g}/MTok",
                              annotation_font_color=INK["secondary"])
        col_out.plotly_chart(out_fig, width="stretch")
        crossings = api_crossover(out_series, min(tiers)) if tiers else []
        crossed = [c for c in crossings if c["crossover_devs"] is not None]
        never = [c["label"] for c in crossings if c["crossover_devs"] is None]
        if crossed:
            col_out.caption(f"Beats the lowest tier (${min(tiers):g}) from: " + "; ".join(
                f"{c['label']} at **N≥{c['crossover_devs']}**" for c in crossed) + ".")
        if never:
            col_out.caption("Never beats it at the measured team sizes: "
                            + ", ".join(never) + ".")

        col_mtok.markdown("**Effective $/MTok (blended)** — all served tokens; the "
                          "agentic mix is prompt-heavy (log-log)")
        mtok_fig = _curve_fig(mtok_series, colors, "cost_per_mtok",
                              "measured $/MTok (all served tokens)", "$%{y:.2f}/MTok")
        _loglog(mtok_fig, mtok_series, "cost_per_mtok")
        col_mtok.plotly_chart(mtok_fig, width="stretch")

        util_series = series_with_metric(series, "utilization")
        col_util.markdown("**Utilization** — U = Θachieved/Θmax; "
                          "the gap is headroom you pay for (penalty = 1/U)")
        util_fig = _curve_fig(util_series, colors, "utilization",
                              "utilization", "%{y:.0%}")
        util_fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
        col_util.plotly_chart(util_fig, width="stretch")
        bases = {s.get("utilization_basis") for s in util_series}
        if bases == {"theta_max"}:
            col_util.caption("Against each combo's measured Θmax (raw-saturation probe).")
        elif "theta_max" in bases:
            col_util.caption("Against measured Θmax where probed; otherwise the combo's "
                             "best measured level (a lower bound on true saturation).")
        else:
            col_util.caption("Against the combo's best measured level — a lower bound on "
                             "true saturation, since the largest N may not saturate the "
                             "server.")

        # The underutilization penalty as a grid (paper Fig. 3): where the cliff lives.
        pm = penalty_matrix(mtok_series)
        if pm["labels"]:
            st.markdown("**Underutilization penalty by team size** — each cell: measured "
                        "\\$/MTok at N over that combo's cheapest point (paper Fig. 3)")
            heat = go.Figure(go.Heatmap(
                z=pm["cells"], x=[f"N={n}" for n in pm["ns"]], y=pm["labels"],
                colorscale=[[0, "#f4f8fd"], [1, "#2a78d6"]],   # sequential, one hue
                text=[[f"{v:.1f}×" if v is not None else "" for v in row]
                      for row in pm["cells"]],
                texttemplate="%{text}", textfont=dict(color=INK["secondary"], size=12),
                hovertemplate="%{y} · %{x}: %{text}<extra></extra>",
                showscale=False, xgap=2, ygap=2))
            _style(heat, x_title="", y_title="")
            heat.update_layout(height=80 + 42 * len(pm["labels"]),
                               margin=dict(t=8, l=8), yaxis=dict(autorange="reversed"))
            st.plotly_chart(heat, width="stretch")

        # The SLO's price (paper Fig. 4 + Table 4): the operating point an SLO-bound
        # operator can actually ship, vs the (unreachable) saturation floor.
        sla_rows = [r for r in sla_table(mtok_series)
                    if r["sla_devs"] is not None or r["c_sat"] is not None]
        if sla_rows:
            st.markdown("**The SLO's price** — the cheapest *shippable* operating point "
                        "vs the saturation floor Csat (unreachable under any real SLO)")
            paired = [r for r in sla_rows
                      if r["mtok_at_sla"] is not None and r["c_sat"] is not None]
            if paired:
                bars = go.Figure()
                bars.add_trace(go.Bar(
                    name="at SLO (shippable)", x=[r["label"] for r in paired],
                    y=[r["mtok_at_sla"] for r in paired],
                    marker=dict(color=CATEGORICAL[0], cornerradius=4),
                    text=[f"N={r['sla_devs']}" for r in paired], textposition="outside",
                    textfont=dict(color=INK["secondary"]),
                    hovertemplate="%{x}<br>at SLO: $%{y:.2f}/MTok<extra></extra>"))
                bars.add_trace(go.Bar(
                    name="Csat (no-SLO floor)", x=[r["label"] for r in paired],
                    y=[r["c_sat"] for r in paired],
                    marker=dict(color=INK["muted"], cornerradius=4,
                                pattern=dict(shape="/")),
                    hovertemplate="%{x}<br>Csat: $%{y:.2f}/MTok<extra></extra>"))
                _style(bars, x_title="", y_title="$/MTok (output)")
                bars.update_layout(barmode="group", bargroupgap=0.08, height=320,
                                   margin=dict(t=32))
                st.plotly_chart(bars, width="stretch")
            st.dataframe(
                [{"model × hardware": r["label"],
                  "max N holding SLO": r["sla_devs"],
                  "$/MTok out @ SLO": r["mtok_at_sla"],
                  "Csat ($/MTok out)": r["c_sat"],
                  "SLO premium": (f"{r['premium']:.2f}×" if r["premium"] else "—")}
                 for r in sla_rows], hide_index=True, width="stretch",
                column_config={
                    "$/MTok out @ SLO": st.column_config.NumberColumn(format="$%.2f"),
                    "Csat ($/MTok out)": st.column_config.NumberColumn(
                        format="$%.2f", help="list price ÷ probed Θmax output throughput"),
                    "SLO premium": st.column_config.TextColumn(
                        help="$/MTok at the SLO-feasible point over the saturation floor "
                             "— what honoring the SLO costs (paper Table 4: 1.13-1.91×)"),
                })
            st.caption("Csat comes from the shape-matched saturation probe. A premium "
                       "below 1× means the operating point out-produced the probe's "
                       "output rate — the probe's shape was more prompt-heavy than the "
                       "workload's effective (cache-discounted) compute mix.")

        # Quantization impact: only same-family × same-hardware × same-GPU-count pairs
        # isolate the quant (comparing across GPUs would confound the two).
        st.markdown("**Quantization impact** — same model, same hardware, different "
                    "quant (paper Fig. 2)")
        pairs = quant_impact(series)
        if pairs:
            for pr in pairs:
                st.markdown(f"*{pr['family']} × {pr['hardware']}-{pr['gpus']}gpu*")
                xs = [f"N={p['devs']}" for p in pr["points"]]
                qbar = go.Figure()
                qbar.add_trace(go.Bar(
                    name=pr["quant_a"], x=xs, y=[p["tph_a"] for p in pr["points"]],
                    marker=dict(color=CATEGORICAL[1], cornerradius=4),
                    hovertemplate=f"{pr['quant_a']} · %{{x}}: %{{y:,.0f}} tok/h<extra></extra>"))
                qbar.add_trace(go.Bar(
                    name=pr["quant_b"], x=xs, y=[p["tph_b"] for p in pr["points"]],
                    marker=dict(color=CATEGORICAL[2], cornerradius=4),
                    text=[f"{p['uplift']:+.0%}" for p in pr["points"]],
                    textposition="outside", textfont=dict(color=INK["secondary"]),
                    hovertemplate=f"{pr['quant_b']} · %{{x}}: %{{y:,.0f}} tok/h<extra></extra>"))
                _style(qbar, x_title="", y_title="tokens/hour served")
                qbar.update_layout(barmode="group", bargroupgap=0.08, height=300,
                                   margin=dict(t=32))
                st.plotly_chart(qbar, width="stretch")
            st.caption("Labels: the second quant's tokens/hour uplift at each common N. "
                       "arXiv:2606.11690 finds the FP8 gain is architecture-dependent "
                       "(~+31% dense vs +69-74% MoE) — these pairs are the harness's "
                       "equivalent, on real agentic load.")
        else:
            st.caption("No comparable pair measured yet: isolating quantization requires "
                       "the same model family in two quants on identical hardware and "
                       "GPU count, at a common team size.")

    # 3 — drill into one team size: the selector sits WITH the ranking + table it controls
    st.divider()
    st.subheader("Drill into one team size")
    n = st.selectbox("Developers (N)", ns, index=len(ns) - 1)
    rows = [x for x in combo_rows(reports, benchmark, n) if x["family"] in selected_fams]
    if not rows:
        st.info(f"No model × hardware measured at {n} devs for the selected family.")
        return
    if overrides:
        rows = reprice_rows(rows, overrides, n)
    for line in compare_narrative(rows, n):
        # Streamlit treats $…$ as LaTeX — escape dollars or the sentence renders mangled.
        st.markdown("- " + line.replace("$", "\\$"))

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
    has_paper = any(x.get("cost_per_mtok") is not None for x in rows)
    table = []
    for x in rows:
        hr = slo_headroom(x["median_tps"], x["min_tps"], x["median_ttft"], x["max_ttft"])
        # Prefer the measured effective $/MTok (server-side counters); older runs fall
        # back to the estimate from median decode tok/s × N.
        mtok = x.get("cost_per_mtok")
        if mtok is None:
            mtok = self_host_usd_per_mtok(x["cost_per_hour"], x["median_tps"], n)
        row = {
            "model × hardware": _combo_label(x),
            "SLO": "✓ holds" if x["holds_slo"] else "✗ fails",
            "$/dev-month": round(x["cost_per_dev_month"]) if x["cost_per_dev_month"] is not None else None,
            "tok/s": round(x["median_tps"]) if x["median_tps"] is not None else None,
            "TTFT (s)": x["median_ttft"],
            "headroom": f"{hr['tps_ratio']:.1f}×" if hr["tps_ratio"] is not None else "—",
            "$/Mtok": round(mtok, 2) if mtok is not None else None,
            "tok/dev": x["tokens_per_dev"],
        }
        if has_paper:
            p90 = x.get("holds_slo_p90")
            row["SLO p90"] = "—" if p90 is None else ("✓ holds" if p90 else "✗ fails")
            row["$/Mtok out"] = (round(x["cost_per_mtok_output"], 2)
                                 if x.get("cost_per_mtok_output") is not None else None)
            row["util %"] = (round(x["utilization"] * 100)
                             if x.get("utilization") is not None else None)
        if any(r.get("repeats", 1) > 1 for r in rows):
            cv = (x.get("cv") or {}).get("cost_per_mtok")
            row["stability"] = (f"n={x.get('repeats', 1)}"
                                + (f", ±{cv:.1%}" if cv is not None else ""))
        # Per-row navigation: st.dataframe cannot embed buttons, but a LinkColumn can
        # deep-link into the detail page via query params.
        row["detail"] = "run-detail?" + urllib.parse.urlencode(
            {"family": x["family"], "quant": x["quant"],
             "hardware": x["hardware"], "gpus": x["gpus"]})
        table.append(row)
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "detail": st.column_config.LinkColumn(
                "detail", display_text="open →",
                help="Full run detail: timeline & cost, gate outcomes."),
            "$/dev-month": st.column_config.NumberColumn(format="$%d",
                help="GPU list price amortized across the N developers"),
            "tok/s": st.column_config.NumberColumn(help="median per-stream decode throughput"),
            "TTFT (s)": st.column_config.NumberColumn(format="%.2f",
                help="median time to first token (SLO caps it)"),
            "headroom": st.column_config.TextColumn(help="median tok/s ÷ SLO floor (>1× = margin)"),
            "$/Mtok": st.column_config.NumberColumn(format="$%.2f",
                help="effective $ per million served tokens at this N — measured from vLLM's "
                     "counters (older runs: estimated from median decode tok/s × N)"),
            "tok/dev": st.column_config.NumberColumn(help="median tokens one developer's task used"),
            "SLO p90": st.column_config.TextColumn(
                help="the same SLO evaluated at the p90 tail — what the median hides"),
            "$/Mtok out": st.column_config.NumberColumn(format="$%.2f",
                help="all cost assigned to generated tokens — compare against API output pricing"),
            "util %": st.column_config.NumberColumn(format="%d%%",
                help="token throughput vs Θmax (or the combo's best level); "
                     "1/U = headroom you pay for"),
            "stability": st.column_config.TextColumn(
                help="repeat measurements aggregated (mean); ± is the run-to-run CV on "
                     "$/MTok (arXiv:2606.11690 §5.8 reports ≤0.31%)"),
        })
    st.caption("The *detail* column opens that combo's full run detail "
               "(timeline & cost, gate outcomes).")


def run_detail():
    import streamlit as st
    reports = _PAGES["reports"]
    # Deep link from the ranking table's LinkColumn (query params) wins; the
    # session-state path remains for programmatic navigation.
    qp = st.query_params
    if all(k in qp for k in ("family", "quant", "hardware", "gpus")):
        try:
            key = (qp["family"], qp["quant"], qp["hardware"], int(qp["gpus"]))
        except ValueError:
            key = None
    else:
        key = st.session_state.get("selected_run")
    rep = next((r for r in reversed(reports) if run_key(r) == key), None) if key else None
    if rep is None:
        st.info("Open a *detail* link from a benchmark page to see a run's detail.")
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
    st.set_page_config(page_title="Developer AI Lab", page_icon=":material/speed:",
                       layout="wide")
    reports = load_runs()

    pages = [st.Page(overview, title="Overview", icon=":material/home:", default=True)]
    benchmark_pages = {}
    for b in benchmarks(reports):
        pg = st.Page((lambda name=b: benchmark_page(name)), title=b, url_path=b,
                     icon=":material/monitoring:")
        benchmark_pages[b] = pg
        pages.append(pg)
    detail = st.Page(run_detail, title="Run detail", url_path="run-detail", visibility="hidden")
    pages.append(detail)

    _PAGES.clear()
    _PAGES.update(reports=reports, benchmarks=benchmark_pages, detail=detail)
    nav = {"": pages[:1], "Benchmarks": list(benchmark_pages.values()) + [detail]}
    st.navigation(nav).run()


if __name__ == "__main__":
    main()
