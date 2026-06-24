import unittest

from scripts.lib.timeline import Timeline, cost_breakdown, fmt_duration


class FakeClock:
    """Deterministic clock: returns the next value on each call."""
    def __init__(self, times):
        self.times = list(times)
        self.i = 0

    def __call__(self):
        t = self.times[self.i]
        self.i += 1
        return t


class TimelineTest(unittest.TestCase):
    def test_marks_produce_contiguous_segments(self):
        # now() called at: mark a=0, mark b=10, mark c=10, stop=40
        tl = Timeline(now=FakeClock([0, 10, 10, 40]))
        tl.mark("a")
        tl.mark("b")     # closes a (10s)
        tl.mark("c")     # closes b (0s)
        tl.stop()        # closes c (30s)
        self.assertEqual(tl.segments,
                         [{"label": "a", "seconds": 10},
                          {"label": "b", "seconds": 0},
                          {"label": "c", "seconds": 30}])

    def test_stop_is_idempotent_and_noop_when_nothing_open(self):
        tl = Timeline(now=FakeClock([0, 5]))
        tl.mark("a")
        tl.stop()        # closes a (5s)
        tl.stop()        # no-op, no clock call beyond what's needed
        self.assertEqual(tl.segments, [{"label": "a", "seconds": 5}])

    def test_on_close_fires_per_segment(self):
        seen = []
        tl = Timeline(now=FakeClock([0, 3, 8]), on_close=seen.append)
        tl.mark("a")
        tl.mark("b")     # closes a
        tl.stop()        # closes b
        self.assertEqual([s["label"] for s in seen], ["a", "b"])


class CostBreakdownTest(unittest.TestCase):
    def test_aggregates_by_label_with_cost_and_pct(self):
        segs = [{"label": "generation", "seconds": 10},
                {"label": "gates", "seconds": 5},
                {"label": "generation", "seconds": 20}]
        rows = cost_breakdown(segs, cost_per_hour=3600.0)  # $1/sec
        self.assertEqual([r["label"] for r in rows], ["generation", "gates"])
        gen, gates = rows
        self.assertEqual(gen["seconds"], 30)
        self.assertAlmostEqual(gen["cost_usd"], 30.0)
        self.assertAlmostEqual(gen["pct"], 30 / 35)
        self.assertAlmostEqual(gates["cost_usd"], 5.0)

    def test_empty_segments(self):
        self.assertEqual(cost_breakdown([], 100.0), [])

    def test_zero_total_seconds_pct_is_zero(self):
        rows = cost_breakdown([{"label": "x", "seconds": 0}], 100.0)
        self.assertEqual(rows[0]["pct"], 0.0)


class FmtDurationTest(unittest.TestCase):
    def test_seconds_only(self):
        self.assertEqual(fmt_duration(7), "7s")

    def test_minutes_and_seconds_zero_padded(self):
        self.assertEqual(fmt_duration(312), "5m12s")
