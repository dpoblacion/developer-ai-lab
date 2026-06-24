"""Capacity metrics from vLLM's Prometheus /metrics endpoint.

Server-side and tunnel-independent: medians come from vLLM's own histograms. We diff two
snapshots (before/after a concurrency level) so each level's latency is isolated.
"""

import re

_LINE = re.compile(r'^(?P<name>[a-zA-Z_:][\w:]*)_bucket\{(?P<labels>[^}]*)\}\s+(?P<val>[\d.eE+-]+)')


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


def _diff(before, after):
    """after - before, aligned by le (assumes the same bucket boundaries)."""
    b = dict(before)
    return sorted((le, c - b.get(le, 0.0)) for le, c in after)


def level_latency(before_text, after_text):
    """Median TTFT (s) and median decode throughput (tok/s) for the requests served between
    the two snapshots."""
    ttft = _diff(histogram_buckets(before_text, "vllm:time_to_first_token_seconds"),
                 histogram_buckets(after_text, "vllm:time_to_first_token_seconds"))
    tpot = _diff(histogram_buckets(before_text, "vllm:time_per_output_token_seconds"),
                 histogram_buckets(after_text, "vllm:time_per_output_token_seconds"))
    median_ttft = quantile_from_buckets(ttft, 0.5)
    median_tpot = quantile_from_buckets(tpot, 0.5)
    median_tps = (1.0 / median_tpot) if median_tpot > 0 else 0.0
    return {"median_ttft": median_ttft, "median_tps": median_tps}
