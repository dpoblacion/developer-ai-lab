"""Per-step timing for a benchmark run. A Timeline records contiguous segments via mark()
(each mark closes the previous segment and opens the next), so the segments cover the billed
pod window with no gaps. Pure: the clock is injected, so it unit-tests without real waiting."""

import time


class Timeline:
    """Records contiguous (label, seconds) segments. ``now`` is injected for testing; ``on_close``
    (optional) is called with each segment as it closes — used to log step durations live."""

    def __init__(self, now=time.time, on_close=None):
        self._now = now
        self._on_close = on_close
        self.segments = []
        self._open = None  # (label, start_time)

    def mark(self, label):
        t = self._now()
        if self._open is not None:
            self._close(t)
        self._open = (label, t)

    def stop(self):
        if self._open is not None:
            self._close(self._now())

    def _close(self, t):
        label, start = self._open
        self._open = None
        seg = {"label": label, "seconds": t - start}
        self.segments.append(seg)
        if self._on_close:
            self._on_close(seg)


def cost_breakdown(segments, cost_per_hour):
    """Aggregate segments by label (first-occurrence order) into per-step cost rows.

    Each row: {label, seconds, cost_usd, pct}. cost_usd = seconds/3600 * cost_per_hour;
    pct = fraction of total seconds (0.0 when total is 0)."""
    order, agg = [], {}
    for s in segments:
        if s["label"] not in agg:
            agg[s["label"]] = 0.0
            order.append(s["label"])
        agg[s["label"]] += s["seconds"]
    total = sum(agg.values())
    return [{"label": label, "seconds": agg[label],
             "cost_usd": agg[label] / 3600 * cost_per_hour,
             "pct": (agg[label] / total) if total else 0.0}
            for label in order]


def fmt_duration(secs):
    """Human elapsed: '7s' / '5m12s'."""
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"
