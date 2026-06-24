"""SDD benchmark handler. The pod serves vLLM only; the agent + gates run locally in a Docker
toolchain container reaching the pod's vLLM over an SSH tunnel (LiteLLM runs in the container).
The pod is stopped as soon as generation ends; gates score with the pod down."""

import json
import os
import pathlib
import subprocess
import sys
import threading
import time

import yaml

from scripts.lib.compose import DEFAULT_SLO
from scripts.lib.orchestrator import REMOTE_DIR, log, run_cmd
from scripts.lib.runpod_pod import ssh_run_cmd
from scripts.lib.sdd_cmds import ssh_tunnel_cmd, docker_run_cmd
from scripts.lib.vllm_metrics import level_latency
from scripts.lib import bench_report
from scripts.lib import pod_guard
from scripts.lib import report_terminal

BASE_IMAGE = "dail-toolchain"

# Fatal patterns for the vllm.log startup scan. Matched only on non-WARNING/INFO lines:
# vLLM emits benign FP8 warnings ("Config file not found ...") that would otherwise
# false-trip 'not found' and abort a healthy (still-loading) startup. 'Killed' catches the
# kernel OOM-killer, 'out of memory' the CUDA/torch OOMs that may lack a traceback line.
VLLM_FATAL_PATTERNS = ("not found|Traceback|RuntimeError|ValueError|No available"
                       "|out of memory|OutOfMemoryError|Killed")


def vllm_fail_scan_cmd(log_path="/workspace/vllm.log", exit_path="/workspace/vllm.exit"):
    """Shell pipeline (runs on the pod) that prints the log tail iff vLLM died — non-empty
    output means the run should fail fast instead of burning MAX_STARTUP of GPU. Two
    signals: the exit sentinel (written by vllm_launch_cmd when the chain dies — catches
    ANY death, pattern-independent) and the fatal-pattern grep (catches fatal logs from a
    still-running process)."""
    return (f"if [ -f {exit_path} ]; then "
            f"echo \"vLLM chain exited (code $(cat {exit_path}))\"; tail -25 {log_path}; "
            f"else grep -vE 'WARNING|INFO' {log_path} "
            f"| grep -qE '{VLLM_FATAL_PATTERNS}' "
            f"&& tail -25 {log_path} || true; fi")


def vllm_launch_cmd(remote_dir, composed_path, log_path="/workspace/vllm.log",
                    exit_path="/workspace/vllm.exit", hf_token=""):
    """Detached setup+vLLM chain for the pod. stdin from /dev/null so ssh returns
    immediately; all output to the log. When the chain dies (setup failure OR vllm
    exiting, cleanly or not) its exit code lands in the sentinel that vllm_fail_scan_cmd
    flags. hf_token is exported inline (the .env is NOT rsync'd — see orchestrator.EXCLUDES)
    so HF downloads still authenticate without shipping RUNPOD_API_KEY to the pod."""
    hf = f"export HF_TOKEN={hf_token}; " if hf_token else ""
    return (f"rm -f {exit_path}; "
            f"({hf}cd {remote_dir} && INSTALL_CLAUDE=0 ./scripts/setup_pod.sh {composed_path} "
            f"&& ./scripts/start_vllm.sh {composed_path}; echo $? > {exit_path}) "
            f"</dev/null >{log_path} 2>&1 &")


SCAN_TIMEOUT = 20


def _scan_pod_log(ip, port, key, runner=subprocess.run):
    """Run the fail-scan on the pod; '' when the ssh exec itself stalls — a slow scan must
    read as 'nothing found this poll', not crash a healthy still-loading startup."""
    try:
        out = runner(ssh_run_cmd(ip, port, key, vllm_fail_scan_cmd()),
                     capture_output=True, text=True, timeout=SCAN_TIMEOUT)
    except subprocess.TimeoutExpired:
        return ""
    return out.stdout


def toolchain_image(scenario_path):
    """Pick the toolchain image for a scenario: its overlay (dail-toolchain-<name>) when the
    scenario ships its own Dockerfile, else the scenario-agnostic base."""
    scenario_dir = pathlib.Path(scenario_path).parent
    if (scenario_dir / "Dockerfile").exists():
        return f"{BASE_IMAGE}-{scenario_dir.name}"
    return BASE_IMAGE


def _docker_image_exists(image):
    return subprocess.run(["docker", "image", "inspect", image],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def require_toolchain_image(scenario_path, exists=_docker_image_exists):
    """Fail fast (before any paid pod) if the scenario's toolchain image isn't built locally —
    without it the agent containers exit instantly and the run yields an all-zeros report. The
    `exists` check is injected so this is unit-tested without Docker. Returns the image name."""
    image = toolchain_image(scenario_path)
    if exists(image):
        return image
    scenario_dir = pathlib.Path(scenario_path).parent
    dockerfile = (f"{scenario_dir}/Dockerfile" if image != BASE_IMAGE
                  else "infra/toolchain/Dockerfile")
    raise SystemExit(f"toolchain image '{image}' not found locally — agents can't run without it.\n"
                     f"Build it:  docker build -t {image} -f {dockerfile} .")


def _gen_progress(out_dir):
    """Largest mtime under the SDD out dir; advances as phases write files."""
    latest = 0.0
    for root, _dirs, files in os.walk(out_dir):
        for f in files:
            try:
                latest = max(latest, os.path.getmtime(os.path.join(root, f)))
            except OSError:
                pass
    return latest


def _curl(url, timeout):
    """curl -sf with a hard timeout (both curl's -m and the subprocess); None on timeout.
    A half-open SSH tunnel must never hang a poll loop."""
    try:
        return subprocess.run(["curl", "-sf", "-m", str(timeout), url],
                              capture_output=True, text=True, timeout=timeout + 5)
    except subprocess.TimeoutExpired:
        return None


def _url_ok(url, timeout=10):
    """True iff the URL answers within the timeout."""
    out = _curl(url, timeout)
    return out is not None and out.returncode == 0


def _metrics_snapshot(url="http://localhost:8000/metrics", timeout=20):
    """vLLM /metrics over the tunnel (localhost:8000). '' on failure/timeout (level latency
    then 0)."""
    out = _curl(url, timeout)
    return out.stdout if out is not None and out.returncode == 0 else ""


def _docker_rm_f(names):
    """Force-remove containers by name, best-effort. SIGKILLing the `docker run` client
    does NOT stop the container (and --rm never fires) — only the daemon can, and a
    surviving container's name would collide with a later run."""
    try:
        subprocess.run(["docker", "rm", "-f", *names],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    except subprocess.TimeoutExpired:
        log(f"  (docker rm -f {' '.join(names)} timed out)")


def _container_name(kind, run_id, n, agent):
    """Container name unique per run: an orphan left by a timeout/kill must not collide
    with the next run's identically-shaped name."""
    parts = ["dail", kind, run_id, f"n{n}", str(agent)]
    return "-".join(p for p in parts if p)


def _stdout_sink(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def _spawn_prefixed(cmd, prefix, sink=_stdout_sink):
    """Popen cmd with every output line (stdout+stderr) tagged '[prefix] …' via a drain
    thread — N concurrent agents share the console, and untagged interleaving reads as
    turn-taking. Draining also stops the pipe from filling and blocking the container.
    The thread is exposed as proc.pump_thread (daemon; ends when the pipe closes)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)

    def _pump():
        for line in proc.stdout:
            sink(f"[{prefix}] {line}")
        proc.stdout.close()

    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    proc.pump_thread = t
    return proc


def _wait_procs(procs, guard, poll=10, names=(), rm=_docker_rm_f):
    """Wait for the agent containers, polling the guard: on a watchdog abort (the pod is
    already terminated) kill the survivors — terminate (SIGTERM, proxied into the
    container), then kill + reap, then force-remove the named containers — and raise,
    instead of waiting forever on agents retrying a dead endpoint. Wall-clock ceilings
    stay the guard's job (stall / MAX_RUN)."""
    pending = list(procs)
    while pending:
        try:
            pending[0].wait(timeout=poll)
            pending.pop(0)
        except subprocess.TimeoutExpired:
            if guard.aborted:
                for q in pending:
                    q.terminate()
                for q in pending:
                    try:
                        q.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        q.kill()
                        q.wait()          # reap the SIGKILLed docker client
                if names:
                    rm(list(names))       # the daemon-side containers outlive the clients
                guard.raise_if_aborted()
    # The abort can land with every proc already dead (they exit instantly once the pod is
    # gone): no wait() ever times out, so check once more or the truncated level would be
    # scored and reported as a success.
    guard.raise_if_aborted()


GATES_TIMEOUT = 900


def _run_with_timeout(cmd, timeout, label, runner=subprocess.run, cleanup=None):
    """Run cmd with a hard timeout; on expiry log, run the cleanup hook (e.g. force-remove
    the container the killed docker client left behind) and return False."""
    try:
        runner(cmd, timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        log(f"  {label} timed out after {timeout}s")
        if cleanup:
            cleanup()
        return False


def _load_json(path):
    """Parse a JSON file; {} on missing or invalid."""
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _save_vllm_log(ctx):
    """Pull the pod's vLLM log to results/ before the pod is terminated. On a startup failure
    the fail-fast tail keeps only 25 lines (the generic 'Engine core init failed' traceback);
    the real root cause is higher up and is lost once the pod dies. Best-effort: never masks
    the original failure."""
    try:
        out = subprocess.run(ssh_run_cmd(ctx.ip, ctx.port, ctx.key,
                             "tail -c 500000 /workspace/vllm.log"),
                             capture_output=True, text=True, timeout=30)
        ctx.out_dir.mkdir(parents=True, exist_ok=True)
        dest = ctx.out_dir / "vllm-startup-fail.log"
        dest.write_text(out.stdout)
        log(f"  saved vLLM startup log -> {dest}")
    except Exception as exc:  # diagnostics must never hide the real error
        log(f"  (could not save vLLM startup log: {exc})")


def _agent_env(served, out_subdir):
    return {
        "SDD_OUT_DIR": out_subdir,
        "ANTHROPIC_BASE_URL": "http://localhost:4000",
        "ANTHROPIC_AUTH_TOKEN": "sk-dev-lab",
        "ANTHROPIC_MODEL": "dev-model",
        "LITELLM_UPSTREAM_MODEL": f"hosted_vllm/{served}",
        "VLLM_BASE": "http://localhost:8011/v1",
        "DESTREAM_UPSTREAM": "http://host.docker.internal:8000",
        "PYTHONPATH": "/repo",
        "MODEL": served,
    }


# sleep 8: crude warmup so the destream proxy + LiteLLM bind their ports before the agent's
# first request; if they are slower on a cold container, Claude Code's own retries cover it.
_GEN = ("set -e; python3 -m scripts.destream_proxy > /tmp/destream.log 2>&1 & "
        "litellm --config /repo/infra/litellm/config.yaml --port 4000 > /tmp/litellm.log 2>&1 & "
        "sleep 8; python3 -m scripts.run_sdd_scenario /repo/{scenario}")


def _generate_level(ctx, image, mounts_for, n):
    """Run the task with n concurrent agents (pod up); return (latency_dict, agent_dirs).
    Gate scoring is deferred to _score_level so it never bills GPU."""
    before = _metrics_snapshot()
    ctx.timeline.mark("generation")
    procs, dirs, names = [], [], []
    for i in range(n):
        agent_dir = ctx.out_dir / "artifacts" / f"n{n}" / f"agent{i}"
        agent_dir.mkdir(parents=True, exist_ok=True)
        dirs.append(agent_dir)
        name = _container_name("agent", ctx.run_id, n, i)
        names.append(name)
        cmd = docker_run_cmd(image, ["bash", "-lc", _GEN.format(scenario=ctx.scenario_path)],
                             mounts=mounts_for(agent_dir), env=_agent_env(ctx.served, "/out"),
                             workdir="/repo", name=name)
        procs.append(_spawn_prefixed(cmd, f"agent{i}"))
    _wait_procs(procs, ctx.guard, names=names)  # guard-aware: an abort kills the survivors
    after = _metrics_snapshot()
    return level_latency(before, after), dirs


def _score_level(ctx, image, mounts_for, n, dirs):
    """Gate-score a level's agent workspaces. Runs with the pod DOWN: gates are local
    CPU-only containers (host_gateway=False) and must not bill idle GPU."""
    records = []
    for i, agent_dir in enumerate(dirs):
        gname = _container_name("gates", ctx.run_id, n, agent_dir.name)
        ok = _run_with_timeout(docker_run_cmd(
            image, ["bash", "-lc", f"python3 -m scripts.run_gates /repo/{ctx.scenario_path} /out/workspace"],
            mounts=mounts_for(agent_dir), env={"PYTHONPATH": "/repo", "MODEL": ctx.served},
            workdir="/repo", name=gname, host_gateway=False),
            GATES_TIMEOUT, f"gates n{n} {agent_dir.name}",
            cleanup=lambda gname=gname: _docker_rm_f([gname]))
        hard = _load_json(agent_dir / "hard-score.json") if ok else {}
        gen = _load_json(agent_dir / "generation.json")
        records.append(bench_report.agent_record(i, hard, gen, gate_timed_out=not ok))
    return records


def run(ctx):
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    guard = ctx.guard
    image = toolchain_image(ctx.scenario_path)
    tunnel = None
    try:
        # Detach setup + vLLM (subshell, stdin from /dev/null) so ssh returns immediately
        # instead of hanging on the backgrounded process's channel. Errors land in vllm.log,
        # which the health-poll below checks (fail-fast).
        ctx.timeline.mark("vLLM startup")
        run_cmd(ssh_run_cmd(ctx.ip, ctx.port, ctx.key,
                vllm_launch_cmd(REMOTE_DIR, ctx.composed_path,
                                hf_token=os.environ.get("HF_TOKEN", ""))),
                timeout=120, label="ssh: setup_pod + start_vllm (detached)")

        log("waiting for vLLM to serve (via SSH tunnel); fails fast on pod errors")
        # stderr -> DEVNULL: until vLLM binds :8000 the tunnel logs a benign "channel N: open
        # failed" on every poll — pure noise. Tunnel death is detected via poll() below.
        tunnel = subprocess.Popen(ssh_tunnel_cmd(ctx.ip, ctx.port, ctx.key), stderr=subprocess.DEVNULL)
        reachable = False
        for i in range(360):  # backstop ceiling; guard MAX_STARTUP (~12 min) governs
            guard.raise_if_aborted()
            if _url_ok("http://localhost:8000/health"):
                reachable = True
                break
            if i % 12 == 11:  # ~every 60s: progress + check vLLM didn't fail to start
                found = _scan_pod_log(ctx.ip, ctx.port, ctx.key)
                if found.strip():
                    log("FAILED: vLLM failed to start on the pod:\n" + found)
                    _save_vllm_log(ctx)
                    raise SystemExit("vLLM startup failed")
                log(f"  still waiting for vLLM ... ({(i + 1) * 5 // 60} min)")
            if tunnel.poll() is not None:  # tunnel died — reopen it
                tunnel = subprocess.Popen(ssh_tunnel_cmd(ctx.ip, ctx.port, ctx.key),
                                          stderr=subprocess.DEVNULL)
            time.sleep(5)
        if not reachable:
            _save_vllm_log(ctx)
            raise SystemExit(
                "FAILED: vLLM did not serve within the backstop window (guard MAX_STARTUP governs)")
        log("  vLLM reachable via tunnel.")

        # Optional diagnostic (SDD_VLLM_PROBE=1): hit vLLM directly through the tunnel with a
        # tool, streaming + non-streaming, to isolate the vLLM tool parser from LiteLLM + Claude.
        if os.getenv("SDD_VLLM_PROBE") == "1":
            ctx.timeline.mark("probe")
            log("probing vLLM tool parsing directly (SDD_VLLM_PROBE=1)")
            vllm_probe = subprocess.run(
                [sys.executable, "-m", "scripts.probe_vllm_tools",
                 "http://localhost:8000/v1", ctx.served, str(ctx.out_dir / "vllm_probe.json")],
                capture_output=True, text=True)
            log(vllm_probe.stdout + (vllm_probe.stderr or ""))

        scenario = yaml.safe_load(pathlib.Path(ctx.scenario_path).read_text())
        devs = scenario.get("devs") or [1]
        slo = scenario.get("slo") or DEFAULT_SLO
        repo = str(pathlib.Path.cwd())

        def mounts_for(agent_dir):
            # No docker.sock: the benchmark never runs docker inside the agent container,
            # and mounting the host socket would give an LLM-driven shell host-root.
            return [(repo, "/repo"), (str(agent_dir), "/out")]

        guard.set_progress(ctx.pod_id, lambda: _gen_progress(str(ctx.out_dir)))
        cost_per_hour = ctx.price_usd_per_gpu_hour * ctx.vllm_cfg.get("tensor_parallel_size", 1)
        generated = []
        for n in sorted(devs):
            guard.raise_if_aborted()
            # Re-enter the generation phase per level with a stall budget scaled to N: on a
            # slow GPU a queued straggler can go minutes without writing while the rest
            # finish, which a fixed budget would false-abort.
            guard.phase("generation", stall=pod_guard.stall_for_devs(n))
            log(f"testing {n} concurrent developer(s)")
            latency, dirs = _generate_level(ctx, image, mounts_for, n)
            generated.append((n, latency, dirs))
    finally:
        if tunnel:
            tunnel.terminate()

    ctx.timeline.stop()
    ctx.stop_pod()  # capacity measured; gates are local CPU work and score with the pod down
    dev_records = []
    for n, latency, dirs in generated:
        records = _score_level(ctx, image, mounts_for, n, dirs)
        rec = bench_report.dev_record(n, latency, records, slo, cost_per_hour)
        dev_records.append(rec)
        log(f"  n={n}: holds_slo={rec['holds_slo']} ${rec['cost_per_dev_month']:.0f}/dev-mo "
            f"tps={rec['median_tps']:.1f} valid={rec['valid']} gates_ok={rec['agents_ok']}/{n}")
    report = bench_report.build_report(
        ctx.family, ctx.quant, ctx.served, ctx.hardware,
        pathlib.Path(ctx.scenario_path).parent.name,
        ctx.vllm_cfg.get("tensor_parallel_size", 1), ctx.price_usd_per_gpu_hour,
        ctx.run_id, dev_records, slo, timeline_segments=ctx.timeline.segments)
    (ctx.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    log(f"DONE. report -> {ctx.out_dir}/report.json")
    report_terminal.render(report)
