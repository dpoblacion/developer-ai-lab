"""Capacity metrics from vLLM's Prometheus /metrics endpoint.

Server-side and tunnel-independent: medians come from vLLM's own histograms. We diff two
snapshots (before/after a concurrency level) so each level's latency is isolated.
"""

import re

_LINE = re.compile(r'^(?P<name>[a-zA-Z_:][\w:]*)_bucket\{(?P<labels>[^}]*)\}\s+(?P<val>[\d.eE+-]+)')
_SAMPLE = re.compile(r'^(?P<name>[a-zA-Z_:][\w:]*?)(?:\{[^}]*\})?\s+(?P<val>[\d.eE+-]+)$')


def counter_total(text, metric):
    """Sum of a counter metric's samples across label sets (exact name match)."""
    total = 0.0
    for line in text.splitlines():
        m = _SAMPLE.match(line.strip())
        if m and m.group("name") == metric:
            total += float(m.group("val"))
    return total


def histogram_buckets(text, metric):
    """Cumulative (le, count) buckets for a histogram metric, sorted by le (+Inf -> inf)."""
    out = {}
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if not m or m.group("name") != metric:
            continue
        le_m = re.search(r'le="([^"]+)"', m.group("labels"))
        if not le_m:
            continue
        le = float("inf") if le_m.group(1) in ("+Inf", "Inf") else float(le_m.group(1))
        out[le] = float(m.group("val"))
    return sorted(out.items())


def quantile_from_buckets(buckets, q):
    """Prometheus-style interpolated quantile from cumulative (le, count) buckets."""
    if not buckets:
        return 0.0
    total = buckets[-1][1]
    if total == 0:
        return 0.0
    rank = q * total
    prev_le, prev_c = 0.0, 0.0
    for le, c in buckets:
        if c >= rank:
            if le == float("inf"):
                return prev_le
            if c == prev_c:
                return le
            return prev_le + (le - prev_le) * (rank - prev_c) / (c - prev_c)
        prev_le, prev_c = le, c
    return buckets[-1][0]


def level_prefix_cache(before_text, after_text):
    """Prefix-cache activity for the level (token-level counter diffs). Agentic traffic
    shares real prefixes (system prompts, tool loops), so the cache is live here — the
    hit rate bounds how these numbers compare to cache-free protocols (arXiv:2606.11690
    §5.7: real hits cut saturation cost by 20-22%). hit_rate is None with no queries."""
    def diff(base):
        # Name drift across vLLM versions: 0.11 spells vllm:prefix_cache_queries_total;
        # other lines use a gpu_ prefix and/or drop the _total suffix. (_created
        # companions are timestamps and never match these exact names.)
        for metric in (f"vllm:prefix_cache_{base}_total", f"vllm:gpu_prefix_cache_{base}_total",
                       f"vllm:gpu_prefix_cache_{base}", f"vllm:prefix_cache_{base}"):
            delta = counter_total(after_text, metric) - counter_total(before_text, metric)
            if delta or counter_total(after_text, metric):
                return int(delta)
        return 0
    queries = diff("queries")
    hits = diff("hits")
    return {"queries": queries, "hits": hits,
            "hit_rate": (hits / queries) if queries > 0 else None}


def level_metrics(before_text, after_text, duration_s):
    """Everything a level's dev_record needs from the two /metrics snapshots: latency
    quantiles, server-side token counts, prefix-cache activity, and the level's wall
    duration."""
    out = level_latency(before_text, after_text)
    out["server_tokens"] = level_tokens(before_text, after_text)
    out["prefix_cache"] = level_prefix_cache(before_text, after_text)
    out["duration_s"] = duration_s
    return out


def level_tokens(before_text, after_text):
    """Server-side prompt/generation token counts for the requests served between the two
    snapshots (counter diffs — the ground truth the effective $/MTok is computed from)."""
    return {
        "prompt": int(counter_total(after_text, "vllm:prompt_tokens_total")
                      - counter_total(before_text, "vllm:prompt_tokens_total")),
        "generation": int(counter_total(after_text, "vllm:generation_tokens_total")
                          - counter_total(before_text, "vllm:generation_tokens_total")),
    }


def _diff(before, after):
    """after - before, aligned by le (assumes the same bucket boundaries)."""
    b = dict(before)
    return sorted((le, c - b.get(le, 0.0)) for le, c in after)


def level_latency(before_text, after_text):
    """Latency quantiles for the requests served between the two snapshots: TTFT (s),
    decode throughput (tok/s) and end-to-end request latency (s) at the median and the
    p90/p99 tails. Medians alone let a large minority of requests violate SLO unseen;
    arXiv:2606.11690 measures P50/P90/P99 and expresses its example SLA at p99."""
    ttft = _diff(histogram_buckets(before_text, "vllm:time_to_first_token_seconds"),
                 histogram_buckets(after_text, "vllm:time_to_first_token_seconds"))
    tpot = _diff(histogram_buckets(before_text, "vllm:time_per_output_token_seconds"),
                 histogram_buckets(after_text, "vllm:time_per_output_token_seconds"))
    e2e = _diff(histogram_buckets(before_text, "vllm:e2e_request_latency_seconds"),
                histogram_buckets(after_text, "vllm:e2e_request_latency_seconds"))

    def tps(q):
        t = quantile_from_buckets(tpot, q)
        return (1.0 / t) if t > 0 else 0.0

    return {
        "median_ttft": quantile_from_buckets(ttft, 0.5),
        "median_tps": tps(0.5),
        "p90_ttft": quantile_from_buckets(ttft, 0.9),
        "p90_tps": tps(0.9),
        "p99_ttft": quantile_from_buckets(ttft, 0.99),
        "p99_tps": tps(0.99),
        "e2e_p50": quantile_from_buckets(e2e, 0.5),
        "e2e_p99": quantile_from_buckets(e2e, 0.99),
    }
