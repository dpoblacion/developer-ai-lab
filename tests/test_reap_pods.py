import unittest

from scripts.reap_pods import choose


class ChooseTest(unittest.TestCase):
    def test_age_zero_kills_all(self):
        pods = [{"id": "a"}, {"id": "b"}]
        self.assertEqual(sorted(choose(pods, [], now=100.0, reap_age=0)), ["a", "b"])

    def test_age_filter_uses_state_created_at(self):
        pods = [{"id": "old"}, {"id": "young"}]
        state = [{"pod_id": "old", "created_at": 0.0}, {"pod_id": "young", "created_at": 95.0}]
        self.assertEqual(choose(pods, state, now=100.0, reap_age=30), ["old"])
