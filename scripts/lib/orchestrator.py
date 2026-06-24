"""Shared orchestration toolkit for benchmark runs.

A run is: provision a pod (the shared prologue, under a PodGuard) then hand a BenchContext to the
benchmark runner's run(ctx) (bench_sdd.run). Everything up to "the pod is up and the repo is on
it" lives here.
"""

import contextlib
import dataclasses
import os
import subprocess
import time

from scripts.lib.pod_guard import PodGuard
from scripts.lib.runpod_pod import (
    build_create_kwargs, is_ready, ssh_endpoint, ssh_run_cmd, rsync_up_cmd)
from scripts.lib.timeline import Timeline, fmt_duration

REMOTE_DIR = "/workspace/developer-ai-lab"
# .env holds RUNPOD_API_KEY (spend on the account) + HF_TOKEN — never ship it to a
# third-party ephemeral pod. HF_TOKEN reaches the pod via the launch env instead
# (bench_sdd.vllm_launch_cmd); RUNPOD_API_KEY has no reason to be on the pod at all.
EXCLUDES = [".git", ".env", "results", "__pycache__", ".venv", "docs"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cmd(cmd, timeout=None, label=None):
    """Run a subprocess, raising on failure. Logs a short label unless SDD_VERBOSE=1."""
    if os.getenv("SDD_VERBOSE"):
        log("+ " + " ".join(cmd))
    else:
        log("+ " + (label or " ".join(cmd[:2]) + " …"))
    subprocess.run(cmd, check=True, timeout=timeout)


def create_with_fallback(runpod, spec, public_key):
    """Try each gpu_type_id until one accepts the pod; raise if none do."""
    last = None
    for gpu in spec["gpu_type_ids"]:
        try:
            log(f"creating pod on {gpu} ...")
            return runpod.create_pod(**build_create_kwargs(spec, gpu, public_key))
        except Exception as exc:  # bad id or no capacity — try the next candidate
            log(f"  {gpu} unavailable: {exc}")
            last = exc
    raise SystemExit(f"No GPU candidate available: {last}")


def create_and_track(runpod, spec, public_key, guard):
    """Create the pod and register it with the guard atomically w.r.t. SIGINT/SIGTERM: a
    signal landing between create_pod returning and guard.track would leak a billing pod
    (only tracked pods are terminated on signal). If the watchdog aborted mid-create,
    track() has already terminated the pod — raise instead of handing it to provision."""
    with guard.deferred_signals():
        pod = create_with_fallback(runpod, spec, public_key)
        guard.track(pod["id"], progress_fn=lambda: 0)
    guard.raise_if_aborted()
    return pod


def wait_ready(runpod, pod_id, retries=180, nap=5):  # 180 × 5s = 15 min ceiling
    for _ in range(retries):
        pod = runpod.get_pod(pod_id)
        if is_ready(pod):
            return pod
        time.sleep(nap)
    raise SystemExit("Pod did not become ready in time")


def wait_for_ssh(ip, port, key, retries=24, nap=5):  # 24 × 5s = 2 min ceiling
    """Probe SSH with an idempotent mkdir until sshd accepts (ports up != sshd ready)."""
    cmd = ssh_run_cmd(ip, port, key, f"mkdir -p {REMOTE_DIR}")
    for _ in range(retries):
        if subprocess.run(cmd, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0:
            return
        time.sleep(nap)
    raise SystemExit("SSH did not become available")


def pod_progress(ip, port, key):
    """Monotonic-ish token sampled over SSH: vllm.log line count + HF dir bytes (0 on
    error/timeout). Feeds the PodGuard watchdog so a long in-pod run isn't false-aborted."""
    try:
        out = subprocess.run(ssh_run_cmd(ip, port, key,
            "wc -l < /workspace/vllm.log 2>/dev/null; "
            "du -sb /workspace/huggingface 2>/dev/null | cut -f1"),
            capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return 0
    nums = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    return sum(nums)


@dataclasses.dataclass
class BenchContext:
    """Everything the runner needs once the pod is provisioned and the repo is on it."""
    ip: str
    port: int
    key: str
    pod_id: str
    composed_path: str
    scenario_path: str
    vllm_cfg: dict
    served: str
    hardware: str
    price_usd_per_gpu_hour: float
    out_dir: object
    guard: object
    family: str = ""
    quant: str = ""
    run_id: str = ""
    timeline: object = None
    keep: bool = False
    _stopped: bool = False

    def stop_pod(self):
        """Terminate the pod now to stop billing early (no-op under --keep). Idempotent."""
        if self.keep or self._stopped:
            return
        self._stopped = True
        self.guard.terminate_all()


@contextlib.contextmanager
def provision(runpod, *, pod_spec, vllm_cfg, key, pub, scenario_path, composed_path,
              out_dir, keep, label, hardware, price_usd_per_gpu_hour, family="", quant="",
              run_id=""):
    """Shared prologue under a PodGuard: create pod (GPU fallback) -> track -> wait ready ->
    SSH endpoint -> wait_for_ssh -> rsync repo up -> set the progress sampler. Yields a
    BenchContext. On exit, terminates the pod via the guard unless keep (then release) or the
    runner already stop_pod()'d it."""
    t_start = time.time()
    timeline = Timeline(on_close=lambda seg: log(f"  ⏱ {seg['label']}: {fmt_duration(seg['seconds'])}"))
    with PodGuard(label=label, terminate_fn=runpod.terminate_pod,
                  list_pods_fn=getattr(runpod, "get_pods", None)) as guard:
        timeline.mark("create→ready")
        log("STEP 1: creating pod")
        pod = create_and_track(runpod, pod_spec, pub, guard)
        pod_id = pod["id"]
        log(f"  pod {pod_id} created; waiting until ready ...")
        pod = wait_ready(runpod, pod_id)
        ip, port = ssh_endpoint(pod)
        log(f"  pod ready at {ip}:{port}  ({fmt_duration(time.time() - t_start)})")
        guard.set_progress(pod_id, lambda: pod_progress(ip, port, key))
        guard.phase("startup")

        log("STEP 2: pushing repo to the pod")
        timeline.mark("rsync")
        wait_for_ssh(ip, port, key)
        run_cmd(rsync_up_cmd(ip, port, key, "./", REMOTE_DIR + "/", EXCLUDES),
                timeout=300, label="rsync repo -> pod")

        ctx = BenchContext(ip=ip, port=port, key=key, pod_id=pod_id,
                           composed_path=composed_path, scenario_path=scenario_path,
                           vllm_cfg=vllm_cfg, served=vllm_cfg["served_model_name"],
                           hardware=hardware, price_usd_per_gpu_hour=price_usd_per_gpu_hour,
                           out_dir=out_dir, guard=guard, keep=keep, timeline=timeline,
                           family=family, quant=quant, run_id=run_id)
        try:
            yield ctx
        finally:
            if keep:
                guard.release(pod_id)
                log(f"pod {pod_id} left running (--keep). Terminate it to stop billing.")
        # guard.__exit__ terminates the pod here unless it was released (keep) or already stopped.
    log(f"total wall-clock: {fmt_duration(time.time() - t_start)}")
