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
    # The SLO gate is judged at the scenario's percentile (arXiv:2606.11690 expresses its
    # SLA at p99); default is the median. Falls back to the median when the requested
    # tail was not measured.
    pct = slo.get("percentile", "p50")
    key = {"p50": "median", "p90": "p90", "p99": "p99"}.get(pct, "median")
    ttft = latency.get(f"{key}_ttft", latency["median_ttft"])
    tps = latency.get(f"{key}_tps", latency["median_tps"])
    holds = ttft <= slo["max_ttft"] and tps >= slo["min_tps"]
    walls = [a["wall_time"] for a in agent_records]
    toks = [a["tokens"]["total"] for a in agent_records]
    median_wall = statistics.median(walls) if walls else 0.0
    record = {
        "devs": devs, "holds_slo": holds, "slo_percentile": pct,
        "median_ttft": latency["median_ttft"], "median_tps": latency["median_tps"],
        "cost_per_dev_hour": cost_per_hour / devs,
        "cost_per_dev_month": cost_per_hour / devs * HOURS_PER_MONTH,
        "valid": median_wall >= MIN_VALID_WALL_S,           # agents did real work (not a 3s garbage exit)
        "agents_ok": sum(1 for a in agent_records if a["passed"]),
        "tokens_per_dev": int(statistics.median(toks)) if toks else 0,
        "agents": agent_records,
    }
    if "p90_ttft" in latency:
        # The tails the median hides: informational gate at the same SLO thresholds
        # (arXiv:2606.11690 conditions cost on tail latency, not p50).
        record["p90_ttft"] = latency["p90_ttft"]
        record["p90_tps"] = latency["p90_tps"]
        record["holds_slo_p90"] = (latency["p90_ttft"] <= slo["max_ttft"]
                                   and latency["p90_tps"] >= slo["min_tps"])
    for k in ("p99_ttft", "p99_tps", "e2e_p50", "e2e_p99"):
        if k in latency:
            record[k] = latency[k]
    # Token provenance for the per-token economics: vLLM's own counters when the level
    # captured them ("server"), else the agents' client-reported usage ("client") — LiteLLM
    # usage is what the server processed, timed over the slowest agent (the level's span).
    server = latency.get("server_tokens")
    duration = latency.get("duration_s") or 0.0
    source = "server" if server and duration > 0 else None
    if source is None and agent_records:
        prompt = sum(a["tokens"]["prompt"] for a in agent_records)
        generation = sum(a["tokens"]["completion"] for a in agent_records)
        duration = max((a.get("wall_time", 0.0) for a in agent_records), default=0.0)
        if duration > 0 and (prompt + generation) > 0:
            server = {"prompt": prompt, "generation": generation}
            source = "client"
    if source:
        # Effective $/MTok (Ceff): list price over the token throughput this level actually
        # achieved. Blended over all served tokens, plus the all-cost-on-output figure for
        # hand-comparison against per-token API pricing (which bills output 5-6x input).
        total_tph = (server["prompt"] + server["generation"]) / duration * 3600
        gen_tph = server["generation"] / duration * 3600
        record["server_tokens"] = server
        record["tokens_source"] = source
        record["tokens_per_hour"] = total_tph
        if total_tph > 0:
            record["cost_per_mtok"] = cost_per_hour / total_tph * 1e6
        if gen_tph > 0:
            record["cost_per_mtok_output"] = cost_per_hour / gen_tph * 1e6
    cache = latency.get("prefix_cache")
    if cache and cache.get("hit_rate") is not None:
        record["prefix_cache_hit_rate"] = cache["hit_rate"]
        if source and duration > 0:
            # Compute-real throughput: cache-hit prompt tokens are served nearly for
            # free, so utilization must discount them or a cache-heavy workload exceeds
            # a no-cache Θmax probe (live 2026-07-04: U>1 at a 96% hit rate). The $/MTok
            # economics above stay on SERVED tokens — that is what the deployment
            # delivers; this field only anchors U.
            computed = max(0, server["prompt"] - cache["hits"]) + server["generation"]
            record["computed_tokens_per_hour"] = computed / duration * 3600
    return record


def apply_utilization(dev_records, theta_max_tph=None):
    """U(N) = Θachieved/Θmax plus the 1/U underutilization penalty (arXiv:2606.11690).
    With a measured Θmax (saturation probe) this is the paper's true utilization and is
    meaningful for any number of levels; without one, the best measured level stands in
    (a lower bound on Θmax), which requires ≥2 levels — a single level is trivially its
    own best, and a meaningless 100% would mislead. Mutates the records in place; returns
    the basis used ("theta_max" | "best_level" | None)."""
    def achieved(d):
        # Compute-real throughput when cache accounting exists (the probe never hits
        # the cache, so compute-vs-compute is the coherent comparison); served otherwise.
        return d.get("computed_tokens_per_hour") or d.get("tokens_per_hour")

    with_tph = [d for d in dev_records if achieved(d)]
    if theta_max_tph:
        denominator, basis = theta_max_tph, "theta_max"
    elif len(with_tph) >= 2:
        denominator, basis = max(achieved(d) for d in with_tph), "best_level"
    else:
        return None
    for d in with_tph:
        d["utilization"] = achieved(d) / denominator
        d["underutilization_penalty"] = denominator / achieved(d)
    return basis if with_tph else None


def _date_from_run_id(run_id):
    """'20260628-101636' -> '2026-06-28'; '' if run_id is not a YYYYMMDD-... timestamp."""
    if len(run_id) >= 8 and run_id[:8].isdigit():
        return f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}"
    return ""


def build_report(family, quant, model, hardware, benchmark, gpu_count, price_usd_per_gpu_hour,
                 run_id, dev_records, slo, timeline_segments=None, model_meta=None,
                 theta_max=None, truncated=None):
    """The full report.json payload for one run: config axes, per-N dev_records (by_devs),
    token totals, and — when a timeline is available — the billed-time breakdown and the
    measured pod_cost_usd (wall-clock × list price)."""
    cost_per_hour = price_usd_per_gpu_hour * gpu_count
    basis = apply_utilization(dev_records,
                              theta_max_tph=(theta_max or {}).get("tokens_per_hour"))
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
    if model_meta:
        report["model_meta"] = model_meta
    if theta_max:
        report["theta_max"] = theta_max
    if basis:
        report["utilization_basis"] = basis
    if truncated:
        # A watchdog abort: the completed levels are real measurements — kept, marked.
        report["truncated"] = truncated
    if timeline_segments is not None:
        breakdown = cost_breakdown(timeline_segments, cost_per_hour)
        report["timeline"] = breakdown
        report["pod_cost_usd"] = sum(r["cost_usd"] for r in breakdown)
    return report
