"""SLO evaluation for the concurrency sweep.

A "developer" is served acceptably at a given concurrency when the typical (median)
stream meets the SLO: TTFT below ``max_ttft`` and decode throughput above ``min_tps``.
The knee is the highest concurrency whose whole prefix still passes — i.e. the maximum
number of concurrent streams the node sustains at SLO.

Pure functions over already-collected samples, so they unit-test offline.
"""

import statistics


def summarize_level(concurrency, samples):
    """Aggregate per-stream samples ({ttft, decode_tps}) for one concurrency level."""
    ttfts = [s["ttft"] for s in samples]
    tps = [s["decode_tps"] for s in samples]
    return {
        "concurrency": concurrency,
        "streams": len(samples),
        "ttft_median": statistics.median(ttfts),
        "ttft_max": max(ttfts),
        "tps_median": statistics.median(tps),
        "tps_min": min(tps),
    }


def evaluate_slo(summary, slo):
    """True if the median stream meets the SLO at this level."""
    return (summary["ttft_median"] <= slo["max_ttft"]
            and summary["tps_median"] >= slo["min_tps"])


def find_knee(summaries, slo):
    """Largest concurrency whose prefix (this level and all lower) passes the SLO.

    Returns 0 if even the smallest level fails.
    """
    knee = 0
    for summary in sorted(summaries, key=lambda s: s["concurrency"]):
        if not evaluate_slo(summary, slo):
            break
        knee = summary["concurrency"]
    return knee
