import os
import tempfile
import unittest

from scripts.lib import pod_guard
from scripts.lib.orchestrator import BenchContext, create_and_track
from scripts.lib.timeline import fmt_duration
from tests.support import make_guard


class _FakeGuard:
    def __init__(self):
        self.terminate_calls = 0

    def terminate_all(self):
        self.terminate_calls += 1


def _ctx(guard, keep=False):
    return BenchContext(ip="1.2.3.4", port=22, key="k", pod_id="p1",
                        composed_path="configs/_composed.yaml", scenario_path="s.yaml",
                        vllm_cfg={"served_model_name": "m"}, served="m",
                        hardware="l40s", price_usd_per_gpu_hour=1.14,
                        out_dir=None, guard=guard, keep=keep)


class FmtDurationTest(unittest.TestCase):
    def test_seconds(self):
        self.assertEqual(fmt_duration(7), "7s")

    def test_minutes(self):
        self.assertEqual(fmt_duration(312), "5m12s")


class ExcludesTest(unittest.TestCase):
    def test_dotenv_is_never_rsynced_to_the_pod(self):
        # .env holds RUNPOD_API_KEY (create/destroy pods = spend on the account) + HF_TOKEN.
        # It must not land on a third-party ephemeral pod. HF_TOKEN reaches the pod via the
        # launch env instead (bench_sdd.vllm_launch_cmd).
        from scripts.lib.orchestrator import EXCLUDES
        self.assertIn(".env", EXCLUDES)


class CreateAndTrackTest(unittest.TestCase):
    """create → track must be atomic w.r.t. SIGINT/SIGTERM: a signal landing right after
    create_pod returns must still terminate the (now tracked) pod instead of leaking it."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "active-pods.json")
        self.killed = []
        self.guard = make_guard(self.killed, state_path=self.path)
        self.spec = {"image": "img", "gpu_type_ids": ["NVIDIA L40S"]}

    def test_happy_path_creates_and_tracks(self):
        class Runpod:
            def create_pod(self, **kw):
                return {"id": "p9"}
        pod = create_and_track(Runpod(), self.spec, "ssh-rsa KEY", self.guard)
        self.assertEqual(pod["id"], "p9")
        self.assertEqual([e["pod_id"] for e in pod_guard.read_state(self.path)], ["p9"])
        self.assertEqual(self.killed, [])

    def test_signal_during_create_still_terminates_the_new_pod(self):
        guard = self.guard
        class Runpod:
            def create_pod(self, **kw):
                guard._on_signal()   # SIGINT lands while create_pod is in flight
                return {"id": "p9"}
        with self.assertRaises(SystemExit):
            create_and_track(Runpod(), self.spec, "ssh-rsa KEY", guard)
        self.assertEqual(self.killed, ["p9"])
        self.assertEqual(pod_guard.read_state(self.path), [])

    def test_watchdog_abort_during_create_terminates_and_raises(self):
        # The watchdog THREAD (not a signal) can abort mid-create: terminate_all runs
        # against an empty _tracked and would never fire again. create_and_track must
        # not hand a billing pod to provision() in that state.
        guard = self.guard
        class Runpod:
            def create_pod(self, **kw):
                guard.aborted = "max_run"    # watchdog fired while create was in flight
                return {"id": "p9"}
        with self.assertRaises(pod_guard.PodGuardAborted):
            create_and_track(Runpod(), self.spec, "ssh-rsa KEY", guard)
        self.assertEqual(self.killed, ["p9"])


class StopPodTest(unittest.TestCase):
    def test_terminates_once_and_is_idempotent(self):
        g = _FakeGuard()
        ctx = _ctx(g)
        ctx.stop_pod()
        ctx.stop_pod()
        self.assertEqual(g.terminate_calls, 1)

    def test_noop_under_keep(self):
        g = _FakeGuard()
        ctx = _ctx(g, keep=True)
        ctx.stop_pod()
        self.assertEqual(g.terminate_calls, 0)
