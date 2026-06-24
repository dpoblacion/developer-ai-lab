import os
import tempfile
import unittest

from scripts.lib import pod_guard
from tests.support import make_guard


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


class SelectUntrackedTest(unittest.TestCase):
    """A pod can be created and lost before track() records it (e.g. create_pod raising
    after RunPod allocated). It is invisible to the state-file reaper; the API sweep finds
    it by our name prefix."""

    def test_flags_dail_pods_missing_from_state(self):
        pods = [{"id": "a", "name": "dail-qwen3-coder-30b-fp8"},
                {"id": "b", "name": "someone-elses-pod"},
                {"id": "c", "name": "dail-glm-5.2"}]
        entries = [{"pod_id": "c", "created_at": 0, "owner_pid": 1, "label": "x"}]
        self.assertEqual(pod_guard.select_untracked(pods, entries), ["a"])

    def test_pod_without_name_is_not_ours(self):
        self.assertEqual(pod_guard.select_untracked([{"id": "x"}], []), [])


class ApiSweepTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "active-pods.json")
        self.killed = []

    def test_init_sweeps_untracked_dail_pods(self):
        make_guard(self.killed, state_path=self.path,
                   list_pods_fn=lambda: [{"id": "ghost", "name": "dail-x"},
                                         {"id": "other", "name": "not-ours"}])
        self.assertEqual(self.killed, ["ghost"])

    def test_api_failure_never_blocks_the_run(self):
        def boom():
            raise RuntimeError("API down")
        make_guard(self.killed, state_path=self.path, list_pods_fn=boom)  # must not raise
        self.assertEqual(self.killed, [])


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


class StallForDevsTest(unittest.TestCase):
    """Generation stall budget scales with team size: N agents share one vLLM, so a queued
    straggler can legitimately go minutes without writing a file. A fixed 480s false-aborts
    a large-N level where 15/16 agents finished and the last is still waiting in the queue."""

    def test_small_teams_keep_the_base_budget(self):
        self.assertEqual(pod_guard.stall_for_devs(4, base=480), 480)
        self.assertEqual(pod_guard.stall_for_devs(1, base=480), 480)

    def test_budget_scales_linearly_with_team_size(self):
        self.assertEqual(pod_guard.stall_for_devs(8, base=480), 960)
        self.assertEqual(pod_guard.stall_for_devs(16, base=480), 1920)
        self.assertEqual(pod_guard.stall_for_devs(32, base=480), 3840)


class PhaseStallOverrideTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "active-pods.json")
        self.clock = FakeClock()

    def _guard(self):
        return make_guard([], state_path=self.path, clock=self.clock, max_run=10_000)

    def test_phase_accepts_a_stall_override(self):
        g = self._guard()
        g.track("p1", progress_fn=lambda: 0)
        g.phase("generation", stall=1920)
        self.clock.t = 1900          # under the scaled budget → no abort
        self.assertIsNone(g._evaluate(self.clock.t))
        self.clock.t = 1921          # over it → stall
        self.assertEqual(g._evaluate(self.clock.t), "stall")

    def test_phase_without_override_uses_the_default(self):
        g = self._guard()
        g.track("p1", progress_fn=lambda: 0)
        g.phase("generation")
        self.clock.t = pod_guard.STALL_GEN + 1
        self.assertEqual(g._evaluate(self.clock.t), "stall")


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


class FakeClock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


class PodGuardTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "active-pods.json")
        self.killed = []
        self.clock = FakeClock()

    def guard(self, **kw):
        return make_guard(self.killed, state_path=self.path,
                          clock=self.clock, is_alive=lambda p: False, **kw)

    def test_terminate_all_idempotent(self):
        g = self.guard()
        g.track("p1", progress_fn=lambda: 0)
        g.terminate_all()
        g.terminate_all()
        self.assertEqual(self.killed, ["p1"])

    def test_stall_evaluates_to_terminate(self):
        g = self.guard(max_run=10_000)
        g.track("p1", progress_fn=lambda: 0)   # progress never changes
        g.phase("generation")                  # stall=STALL_GEN default
        self.clock.t = pod_guard.STALL_GEN + 1
        self.assertEqual(g._evaluate(self.clock.t), "stall")
        self.assertEqual(self.killed, ["p1"])
        with self.assertRaises(pod_guard.PodGuardAborted):
            g.raise_if_aborted()

    def test_progress_resets_stall(self):
        counter = {"n": 0}
        g = self.guard(max_run=10_000)
        g.track("p1", progress_fn=lambda: counter["n"])
        g.phase("generation")
        counter["n"] = 1                       # progress advanced
        self.clock.t = pod_guard.STALL_GEN - 1
        self.assertIsNone(g._evaluate(self.clock.t))
        self.assertEqual(self.killed, [])

    def test_reap_orphans_on_init_kills_dead_owner(self):
        pod_guard.add_entry(self.path, {"pod_id": "ghost", "created_at": 0,
                                        "owner_pid": 999999, "label": "old"})
        self.guard()  # is_alive=False -> ghost reaped
        self.assertIn("ghost", self.killed)
        self.assertEqual(pod_guard.read_state(self.path), [])

    def test_provisioning_phase_has_no_ceiling(self):
        # Guard starts in provisioning (no max_phase ceiling).
        # Advancing clock past MAX_STARTUP should NOT abort the pod even when
        # stall is kept at bay via heartbeat (as the orchestrator does each poll).
        g = self.guard(max_run=10_000)
        g.track("p1", progress_fn=lambda: 0)
        self.clock.t = pod_guard.MAX_STARTUP + 100
        g.heartbeat()  # simulate the orchestrator's ready-poll heartbeat
        self.assertIsNone(g._evaluate(self.clock.t))
        self.assertEqual(self.killed, [])
        # Switching to startup resets the phase clock; now the ceiling IS active.
        g.phase("startup")
        self.clock.t += pod_guard.MAX_STARTUP + 1
        self.assertEqual(g._evaluate(self.clock.t), "max_phase")
        self.assertIn("p1", self.killed)

    def test_set_progress_repoints_sampler_no_new_state_entry(self):
        counter = {"n": 0}
        g = self.guard(max_run=10_000)
        g.track("p1", progress_fn=lambda: counter["n"])
        # State file has exactly one entry for p1
        self.assertEqual(len(pod_guard.read_state(self.path)), 1)
        # Re-point the sampler to a different function
        new_fn = lambda: 999
        g.set_progress("p1", new_fn)
        # _sample() now uses the new fn
        self.assertEqual(g._sample(), 999)
        # State file still has exactly one entry (no duplicate)
        self.assertEqual(len(pod_guard.read_state(self.path)), 1)
        self.assertEqual(pod_guard.read_state(self.path)[0]["pod_id"], "p1")


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "active-pods.json")

    def test_release_prevents_termination_and_clears_state(self):
        terminated = []
        g = pod_guard.PodGuard("t", terminate_fn=terminated.append,
                               state_path=self.path, clock=lambda: 0.0)
        g.track("pod-1")
        g.release("pod-1")
        g.terminate_all()
        self.assertEqual(terminated, [])                       # not terminated
        self.assertEqual(pod_guard.read_state(self.path), [])  # state entry removed

    def test_release_unknown_pod_is_noop(self):
        terminated = []
        g = pod_guard.PodGuard("t", terminate_fn=terminated.append,
                               state_path=self.path, clock=lambda: 0.0)
        g.release("nope")          # no such tracked pod — must not raise
        self.assertEqual(terminated, [])


class DeferredSignalsTest(unittest.TestCase):
    """A SIGINT/SIGTERM landing between create_pod and guard.track would leak a billing pod:
    _on_signal only terminates *tracked* pods. deferred_signals() makes that window atomic —
    the signal is held until the section exits, by which time the pod is tracked."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "active-pods.json")
        self.killed = []
        self.guard = make_guard(self.killed, state_path=self.path)

    def test_signal_inside_section_is_deferred_then_terminates_tracked_pod(self):
        with self.assertRaises(SystemExit):
            with self.guard.deferred_signals():
                self.guard._on_signal()          # SIGINT lands mid-create: must NOT raise here
                self.guard.track("pod-1")        # ... so the pod still gets tracked
        self.assertEqual(self.killed, ["pod-1"])
        self.assertEqual(pod_guard.read_state(self.path), [])

    def test_no_signal_section_is_noop(self):
        with self.guard.deferred_signals():
            self.guard.track("pod-1")
        self.assertEqual(self.killed, [])
        self.assertEqual([e["pod_id"] for e in pod_guard.read_state(self.path)], ["pod-1"])

    def test_signal_outside_section_still_raises_immediately(self):
        self.guard.track("pod-1")
        with self.assertRaises(SystemExit):
            self.guard._on_signal()
        self.assertEqual(self.killed, ["pod-1"])


class TrackAfterAbortTest(unittest.TestCase):
    """If the watchdog aborts while create_pod is in flight (nothing tracked yet),
    terminate_all is a no-op and `not self.aborted` stops it from ever firing again —
    the pod tracked right after would bill until the run notices. track() must
    terminate immediately when the guard is already aborted."""

    def test_track_on_aborted_guard_terminates_the_pod_immediately(self):
        path = os.path.join(tempfile.mkdtemp(), "pods.json")
        killed = []
        g = make_guard(killed, state_path=path)
        g.aborted = "stall"                  # watchdog fired mid-create
        g.track("pod-late")
        self.assertEqual(killed, ["pod-late"])
        self.assertEqual(pod_guard.read_state(path), [])


class GenerationPhaseTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "active-pods.json")

    def test_generation_phase_has_no_wallclock_ceiling(self):
        # A long generation run (well past startup's 720s) with advancing progress must NOT
        # be aborted: the generation phase is bounded by stall + MAX_RUN, not max_phase.
        clock = [0.0]
        progress = [0]
        g = pod_guard.PodGuard("c", terminate_fn=lambda _id: None,
                               state_path=self.path, clock=lambda: clock[0])
        g.track("pod-1", progress_fn=lambda: progress[0])
        g.phase("generation")
        for t in range(60, 1020, 60):   # up to 17 min, past the 720s startup ceiling
            clock[0] = float(t)
            progress[0] += 1            # progress advances each poll (no stall)
            self.assertIsNone(g._evaluate(clock[0]))
