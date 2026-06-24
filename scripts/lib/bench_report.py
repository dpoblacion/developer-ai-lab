"""Assemble a benchmark run's report: flat per-N shape (N devs vs. throughput/latency).

Each dev_record is one team size: does the median stream hold SLO, cost per developer.
"""
import statistics

from scripts.lib.timeline import cost_breakdown

HOURS_PER_MONTH = 730          # the $/dev-month amortization factor (docs/methodology.md)
MIN_VALID_WALL_S = 30.0        # below this median wall-time the agents did no real work


def agent_record(index, hard_score, generation, gate_timed_out=False):
    """One agent's outcome: gate results (output kept only on failure) + token/wall rollup.
    gate_timed_out marks an ENVIRONMENTAL gate timeout (scored failed, but distinguishable
    from a genuine gate failure in the report)."""
    gates = (hard_score or {}).get("gates", {}).get("results", [])
    slim = []
    for g in gates:
        entry = {"id": g["id"], "passed": g["passed"], "exit_code": g.get("exit_code")}
        if not g["passed"] and g.get("output"):
            entry["output"] = g["output"]
        slim.append(entry)
    passed = bool(gates) and all(g["passed"] for g in gates)
    phases = (generation or {}).get("phases", [])
    tokens = {
        "prompt": sum(p.get("prompt_tokens", 0) for p in phases),
        "completion": sum(p.get("completion_tokens", 0) for p in phases),
        "total": sum(p.get("total_tokens", 0) for p in phases),
    }
    wall_time = sum(p.get("wall_time", 0.0) for p in phases)
    record = {"agent": index, "passed": passed, "gates": slim,
              "tokens": tokens, "wall_time": wall_time}
    if gate_timed_out:
        record["passed"] = False
        record["gate_timeout"] = True
    return record


def dev_record(devs, latency, agent_records, slo, cost_per_hour):
    """One team-size (N) answer: does the median stream hold SLO, cost per dev, validity."""
    if devs < 1:
        raise ValueError(f"devs must be >= 1, got {devs}")
    holds = latency["median_ttft"] <= slo["max_ttft"] and latency["median_tps"] >= slo["min_tps"]
    walls = [a["wall_time"] for a in agent_records]
    toks = [a["tokens"]["total"] for a in agent_records]
    median_wall = statistics.median(walls) if walls else 0.0
    return {
        "devs": devs, "holds_slo": holds,
        "median_ttft": latency["median_ttft"], "median_tps": latency["median_tps"],
        "cost_per_dev_hour": cost_per_hour / devs,
        "cost_per_dev_month": cost_per_hour / devs * HOURS_PER_MONTH,
        "valid": median_wall >= MIN_VALID_WALL_S,           # agents did real work (not a 3s garbage exit)
        "agents_ok": sum(1 for a in agent_records if a["passed"]),
        "tokens_per_dev": int(statistics.median(toks)) if toks else 0,
        "agents": agent_records,
    }


def _date_from_run_id(run_id):
    """'20260628-101636' -> '2026-06-28'; '' if run_id is not a YYYYMMDD-... timestamp."""
    if len(run_id) >= 8 and run_id[:8].isdigit():
        return f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}"
    return ""


def build_report(family, quant, model, hardware, benchmark, gpu_count, price_usd_per_gpu_hour,
                 run_id, dev_records, slo, timeline_segments=None):
    """The full report.json payload for one run: config axes, per-N dev_records (by_devs),
    token totals, and — when a timeline is available — the billed-time breakdown and the
    measured pod_cost_usd (wall-clock × list price)."""
    cost_per_hour = price_usd_per_gpu_hour * gpu_count
    tokens = {"prompt": 0, "completion": 0, "total": 0}
    for d in dev_records:
        for a in d["agents"]:
            for k in tokens:
                tokens[k] += a["tokens"][k]
    report = {"run_id": run_id, "date": _date_from_run_id(run_id),
              "benchmark": benchmark, "family": family, "quant": quant, "model": model,
              "hardware": hardware, "gpu_count": gpu_count,
              "price_usd_per_gpu_hour": price_usd_per_gpu_hour, "cost_per_hour": cost_per_hour,
              "slo": slo, "by_devs": dev_records, "tokens": tokens}
    if timeline_segments is not None:
        breakdown = cost_breakdown(timeline_segments, cost_per_hour)
        report["timeline"] = breakdown
        report["pod_cost_usd"] = sum(r["cost_usd"] for r in breakdown)
    return report
