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
