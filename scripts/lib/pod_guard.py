"""Central watchdog + reaper for paid RunPod pods.

Pure decision functions (read_state/select_*/abort_reason) are unit-tested with no network
or threads. ``PodGuard`` drives them over a real pod lifecycle and is injected with a
clock / terminate_fn / progress samplers so its watchdog is testable deterministically.
"""

import json
import os


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f)


def add_entry(path, entry):
    entries = read_state(path)
    entries.append(entry)
    write_state(path, entries)


def remove_entry(path, pod_id):
    entries = [e for e in read_state(path) if e.get("pod_id") != pod_id]
    write_state(path, entries)


def select_orphans(entries, is_alive):
    """Entries whose owner process is no longer alive."""
    return [e for e in entries if not is_alive(e.get("owner_pid"))]


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


import atexit
import signal
import threading
import time


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


STALL_STARTUP = _env_int("STALL_STARTUP", 300)
MAX_STARTUP = _env_int("MAX_STARTUP", 720)
STALL_GEN = _env_int("STALL_GEN", 480)
MAX_RUN = _env_int("MAX_RUN", 3600)
REAP_AGE = _env_int("REAP_AGE", 0)
DEFAULT_STATE_PATH = os.path.expanduser("~/.cache/dail/active-pods.json")
POLL_INTERVAL = 10

PHASES = {
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
                 clock=time.time, is_alive=_pid_alive, max_run=MAX_RUN):
        self.label = label
        self._terminate_fn = terminate_fn
        self.state_path = state_path
        self._clock = clock
        self._is_alive = is_alive
        self.max_run = max_run
        self._tracked = {}          # pod_id -> progress_fn
        self._terminated = set()
        self.aborted = None
        self._run_started_at = clock()
        self._phase_started_at = self._run_started_at
        self._last_progress_at = self._run_started_at
        self._last_progress = None
        self._policy = PHASES["startup"]
        self._stop = threading.Event()
        self._thread = None
        self.reap_orphans()

    def reap_orphans(self):
        entries = read_state(self.state_path)
        for e in select_orphans(entries, self._is_alive):
            self._terminate(e["pod_id"])
            remove_entry(self.state_path, e["pod_id"])

    def track(self, pod_id, progress_fn=lambda: 0):
        self._tracked[pod_id] = progress_fn
        add_entry(self.state_path, {"pod_id": pod_id, "created_at": self._clock(),
                                    "owner_pid": os.getpid(), "label": self.label})
        now = self._clock()
        self._last_progress = progress_fn()
        self._last_progress_at = now

    def phase(self, name):
        self._policy = PHASES[name]
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
