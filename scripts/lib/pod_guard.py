"""Central watchdog + reaper for paid RunPod pods.

Pure decision functions (read_state/select_*/abort_reason) are unit-tested with no network
or threads. ``PodGuard`` drives them over a real pod lifecycle and is injected with a
clock / terminate_fn / progress samplers so its watchdog is testable deterministically.
"""

import atexit
import contextlib
import json
import os
import signal
import threading
import time


def read_state(path):
    """Return the list of tracked-pod entries, or [] if missing/corrupt."""
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return []


def write_state(path, entries):
    """Write entries as JSON, creating parent dirs."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f)


def add_entry(path, entry):
    # Single-writer assumption: one orchestrator run at a time. Concurrent runs could
    # clobber each other's entries because this is an unlocked read-modify-write. The
    # backstop is `make reap` (REAP_AGE=0 terminates all tracked pods via the API).
    entries = read_state(path)
    entries.append(entry)
    write_state(path, entries)


def remove_entry(path, pod_id):
    # See add_entry: single-writer assumption applies here too.
    entries = [e for e in read_state(path) if e.get("pod_id") != pod_id]
    write_state(path, entries)


def select_orphans(entries, is_alive):
    """Entries whose owner process is no longer alive."""
    return [e for e in entries if not is_alive(e.get("owner_pid"))]


def select_untracked(pods, entries, prefix="dail-"):
    """Our pods (by name prefix) with no state entry: created but lost before track()
    could record them (e.g. create_pod raising after RunPod allocated the pod). Relies on
    the same single-writer assumption as add_entry — no other orchestrator is mid-create."""
    tracked = {e.get("pod_id") for e in entries}
    return [p["id"] for p in pods
            if p.get("name", "").startswith(prefix) and p["id"] not in tracked]


def select_to_reap(pod_ids, created_at_by_id, now, reap_age):
    """Pod ids to terminate. reap_age<=0 -> all; else age>=reap_age, unknown treated ancient."""
    if reap_age <= 0:
        return list(pod_ids)
    out = []
    for pid in pod_ids:
        created = created_at_by_id.get(pid)
        if created is None or (now - created) >= reap_age:
            out.append(pid)
    return out


def abort_reason(now, last_progress_at, phase_started_at, run_started_at,
                 *, stall, max_phase, max_run):
    """Return why the pod should be killed now, or None. Global ceiling wins first."""
    if now - run_started_at >= max_run:
        return "max_run"
    if max_phase is not None and now - phase_started_at >= max_phase:
        return "max_phase"
    if now - last_progress_at >= stall:
        return "stall"
    return None


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


STALL_STARTUP = _env_int("STALL_STARTUP", 300)
MAX_STARTUP = _env_int("MAX_STARTUP", 720)
STALL_GEN = _env_int("STALL_GEN", 480)
MAX_RUN = _env_int("MAX_RUN", 3600)


def stall_for_devs(n, base=STALL_GEN):
    """Generation stall budget scaled by team size. N agents share one vLLM instance, so a
    queued straggler's throughput falls ~1/N and it can legitimately go minutes without
    writing a file while the earlier agents finish. Scale the budget ~linearly (base per 4
    devs) so a large-N level isn't false-aborted by the last agent waiting in the queue."""
    return base * max(1, -(-n // 4))   # base × ceil(n/4)
REAP_AGE = _env_int("REAP_AGE", 0)
DEFAULT_STATE_PATH = os.path.expanduser("~/.cache/dail/active-pods.json")
POLL_INTERVAL = 10

PHASES = {
    "provisioning": {"stall": MAX_STARTUP, "max_phase": None},
    "startup": {"stall": STALL_STARTUP, "max_phase": MAX_STARTUP},
    "generation": {"stall": STALL_GEN, "max_phase": None},
}


def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


class PodGuardAborted(Exception):
    pass


class PodGuard:
    def __init__(self, label, terminate_fn, *, state_path=DEFAULT_STATE_PATH,
                 clock=time.time, is_alive=_pid_alive, max_run=MAX_RUN,
                 list_pods_fn=None):
        self.label = label
        self._terminate_fn = terminate_fn
        self.state_path = state_path
        self._clock = clock
        self._is_alive = is_alive
        self._list_pods_fn = list_pods_fn
        self.max_run = max_run
        self._tracked = {}          # pod_id -> progress_fn
        self._terminated = set()
        self.aborted = None
        self._run_started_at = clock()
        self._phase_started_at = self._run_started_at
        self._last_progress_at = self._run_started_at
        self._last_progress = None
        self._policy = PHASES["provisioning"]
        self._stop = threading.Event()
        self._thread = None
        self._deferring = False
        self._pending_signal = False
        self.reap_orphans()

    def reap_orphans(self):
        entries = read_state(self.state_path)
        for e in select_orphans(entries, self._is_alive):
            self._terminate(e["pod_id"])
            remove_entry(self.state_path, e["pod_id"])
        if self._list_pods_fn is None:
            return
        try:
            pods = self._list_pods_fn()
        except Exception:
            return  # the API sweep is a best-effort safety net; never block a run on it
        for pod_id in select_untracked(pods, read_state(self.state_path)):
            self._terminate(pod_id)

    def track(self, pod_id, progress_fn=lambda: 0):
        self._tracked[pod_id] = progress_fn
        add_entry(self.state_path, {"pod_id": pod_id, "created_at": self._clock(),
                                    "owner_pid": os.getpid(), "label": self.label})
        now = self._clock()
        self._last_progress = progress_fn()
        self._last_progress_at = now
        if self.aborted:
            # The watchdog fired while create_pod was in flight: its terminate_all ran
            # against an empty _tracked (no-op) and `not self.aborted` stops it from ever
            # firing again — terminate the late-tracked pod now or it bills unnoticed.
            self.terminate_all()

    def set_progress(self, pod_id, progress_fn):
        """Re-point a tracked pod's progress sampler (no new state entry)."""
        self._tracked[pod_id] = progress_fn

    def phase(self, name, stall=None):
        """Enter a watchdog phase. `stall` overrides the phase's default stall budget (used
        to scale the generation budget with team size — see stall_for_devs)."""
        policy = PHASES[name]
        self._policy = {"stall": policy["stall"] if stall is None else stall,
                        "max_phase": policy["max_phase"]}
        now = self._clock()
        self._phase_started_at = now
        self._last_progress_at = now

    def heartbeat(self):
        self._last_progress_at = self._clock()

    def _sample(self):
        return max((fn() for fn in self._tracked.values()), default=0)

    def _evaluate(self, now):
        sample = self._sample()
        if self._last_progress is None or sample != self._last_progress:
            self._last_progress = sample
            self._last_progress_at = now
        reason = abort_reason(now, self._last_progress_at, self._phase_started_at,
                              self._run_started_at, stall=self._policy["stall"],
                              max_phase=self._policy["max_phase"], max_run=self.max_run)
        if reason and not self.aborted:
            self.aborted = reason
            self.terminate_all()
        return reason

    def raise_if_aborted(self):
        if self.aborted:
            raise PodGuardAborted(self.aborted)

    def _terminate(self, pod_id):
        if pod_id in self._terminated:
            return
        self._terminated.add(pod_id)
        try:
            self._terminate_fn(pod_id)
        except Exception:
            pass

    def terminate_all(self):
        for pod_id in list(self._tracked):
            self._terminate(pod_id)
            remove_entry(self.state_path, pod_id)

    def release(self, pod_id):
        """Stop guarding a pod so it survives (used by --keep): drop it from tracking and the
        state file. After release, neither terminate_all/__exit__ nor the orphan reaper
        terminates it."""
        self._tracked.pop(pod_id, None)
        remove_entry(self.state_path, pod_id)

    def _loop(self):
        while not self._stop.wait(POLL_INTERVAL):
            self._evaluate(self._clock())

    def __enter__(self):
        self._prev = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        for s in self._prev:
            signal.signal(s, self._on_signal)
        atexit.register(self.terminate_all)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _on_signal(self, *_):
        if self._deferring:
            self._pending_signal = True
            return
        self.terminate_all()
        raise SystemExit(1)

    @contextlib.contextmanager
    def deferred_signals(self):
        """Hold SIGINT/SIGTERM across a critical section (create_pod -> track): a signal
        landing in that window would otherwise terminate only *tracked* pods and leak the
        one just created. On exit the held signal fires: terminate_all + SystemExit."""
        self._deferring = True
        self._pending_signal = False
        try:
            yield
        finally:
            self._deferring = False
            if self._pending_signal:
                self.terminate_all()
                raise SystemExit(1)

    def __exit__(self, *exc):
        atexit.unregister(self.terminate_all)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=POLL_INTERVAL + 1)
        self.terminate_all()
        for s, h in getattr(self, "_prev", {}).items():
            signal.signal(s, h)
        return False
