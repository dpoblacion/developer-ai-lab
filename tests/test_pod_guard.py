import os
import tempfile
import unittest

from scripts.lib import pod_guard


class StateFileTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "sub", "active-pods.json")

    def test_read_missing_returns_empty(self):
        self.assertEqual(pod_guard.read_state(self.path), [])

    def test_read_corrupt_returns_empty(self):
        os.makedirs(os.path.dirname(self.path))
        open(self.path, "w").write("not json{")
        self.assertEqual(pod_guard.read_state(self.path), [])

    def test_add_then_read_roundtrip(self):
        e = {"pod_id": "p1", "created_at": 10.0, "owner_pid": 123, "label": "x"}
        pod_guard.add_entry(self.path, e)
        self.assertEqual(pod_guard.read_state(self.path), [e])

    def test_remove_entry(self):
        pod_guard.add_entry(self.path, {"pod_id": "p1", "created_at": 1.0, "owner_pid": 1, "label": "a"})
        pod_guard.add_entry(self.path, {"pod_id": "p2", "created_at": 2.0, "owner_pid": 2, "label": "b"})
        pod_guard.remove_entry(self.path, "p1")
        self.assertEqual([e["pod_id"] for e in pod_guard.read_state(self.path)], ["p2"])


class SelectionTest(unittest.TestCase):
    def test_select_orphans_picks_dead_pids(self):
        entries = [
            {"pod_id": "p1", "owner_pid": 11, "created_at": 0, "label": ""},
            {"pod_id": "p2", "owner_pid": 22, "created_at": 0, "label": ""},
        ]
        alive = {11}.__contains__
        orphans = pod_guard.select_orphans(entries, alive)
        self.assertEqual([e["pod_id"] for e in orphans], ["p2"])

    def test_reap_all_when_age_zero(self):
        self.assertEqual(
            sorted(pod_guard.select_to_reap(["a", "b"], {}, now=100.0, reap_age=0)),
            ["a", "b"])

    def test_reap_by_age_and_unknown(self):
        created = {"old": 0.0, "young": 90.0}  # now=100
        got = pod_guard.select_to_reap(["old", "young", "unknown"], created, now=100.0, reap_age=30)
        self.assertEqual(sorted(got), ["old", "unknown"])


class AbortReasonTest(unittest.TestCase):
    def call(self, now, last, phase_start=0.0, run_start=0.0, stall=300, max_phase=720, max_run=3600):
        return pod_guard.abort_reason(now, last, phase_start, run_start,
                                      stall=stall, max_phase=max_phase, max_run=max_run)

    def test_ok_when_recent_progress(self):
        self.assertIsNone(self.call(now=100, last=50))

    def test_stall_trips(self):
        self.assertEqual(self.call(now=400, last=50), "stall")

    def test_phase_ceiling_trips(self):
        self.assertEqual(self.call(now=800, last=799), "max_phase")

    def test_max_run_wins(self):
        self.assertEqual(self.call(now=4000, last=3999, max_phase=720), "max_run")

    def test_no_phase_ceiling_when_none(self):
        self.assertIsNone(self.call(now=800, last=799, max_phase=None))
