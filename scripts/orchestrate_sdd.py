"""SDD benchmark, agent-local. The GPU pod serves vLLM only; the agent, its build-fix
loop, and the gates run in a local toolchain container that reaches the pod's vLLM over
an SSH tunnel (LiteLLM runs locally in the container). The pod stops as soon as generation
ends; gates run with the pod down.

Requires (.env): RUNPOD_API_KEY, SSH_KEY_PATH. Local Docker + the dail-toolchain image
(build: docker build -t dail-toolchain -f infra/toolchain/Dockerfile .). Scenarios that
need extra runtime (e.g. todo-app's .NET SDK) ship benchmarks/scenarios/<name>/Dockerfile,
an overlay on dail-toolchain; the orchestrator then uses dail-toolchain-<name>, which you
build alongside the base (docker build -t dail-toolchain-<name> -f <that Dockerfile> .).

Usage: python -m scripts.orchestrate_sdd [--config configs/qwen3coder.yaml]
       [--spec infra/runpod/pod.yaml] [--scenario benchmarks/scenarios/todo-app/scenario.yaml]
"""

import argparse
import os
import pathlib
import subprocess
import sys
import time

from scripts.lib.dotenv import load_dotenv
from scripts.lib.pod_guard import PodGuard, PodGuardAborted
from scripts.lib.runpod_pod import (
    build_create_kwargs, is_ready, ssh_endpoint, ssh_run_cmd, rsync_up_cmd)
from scripts.lib.sdd_cmds import ssh_tunnel_cmd, docker_run_cmd

REMOTE_DIR = "/workspace/developer-ai-lab"
EXCLUDES = [".git", "results", "__pycache__", ".venv", "docs/superpowers"]
BASE_IMAGE = "dail-toolchain"


def toolchain_image(scenario_path):
    """Pick the toolchain image for a scenario: its overlay (dail-toolchain-<name>) when the
    scenario ships its own Dockerfile, else the scenario-agnostic base."""
    scenario_dir = pathlib.Path(scenario_path).parent
    if (scenario_dir / "Dockerfile").exists():
        return f"{BASE_IMAGE}-{scenario_dir.name}"
    return BASE_IMAGE


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run(cmd, timeout=None):
    log("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=timeout)  # TimeoutExpired -> finally stops the pod


def _startup_progress(ip, port, key):
    """A monotonic-ish token: pod vllm.log line count + HF dir bytes. 0 on SSH error."""
    out = subprocess.run(ssh_run_cmd(ip, port, key,
        "wc -l < /workspace/vllm.log 2>/dev/null; du -sb /workspace/huggingface 2>/dev/null | cut -f1"),
        capture_output=True, text=True)
    nums = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    return sum(nums)


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


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="infra/runpod/pod.yaml")
    ap.add_argument("--config", default="configs/qwen3coder.yaml")
    ap.add_argument("--scenario", default="benchmarks/scenarios/todo-app/scenario.yaml")
    ap.add_argument("--key", default=os.environ.get("SSH_KEY_PATH"))
    args = ap.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key or not args.key:
        raise SystemExit("Set RUNPOD_API_KEY and SSH_KEY_PATH in .env")
    args.key = os.path.expanduser(args.key)
    image = toolchain_image(args.scenario)

    import yaml
    import runpod
    runpod.api_key = api_key
    spec = yaml.safe_load(pathlib.Path(args.spec).read_text())
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    served = cfg["served_model_name"]
    pub = pathlib.Path(args.key + ".pub").read_text().strip()

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = (pathlib.Path("results") / run_id / "sdd").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = str(pathlib.Path.cwd())
    mounts = [(repo, "/repo"), (str(out_dir), "/out"),
              ("/var/run/docker.sock", "/var/run/docker.sock")]

    pod = None
    tunnel = None
    try:
        log("STEP 1/5: creating vLLM-only pod")
        for gpu in spec["gpu_type_ids"]:
            try:
                pod = runpod.create_pod(**build_create_kwargs(spec, gpu, pub))
                break
            except Exception as exc:
                log(f"  {gpu} unavailable: {exc}")
        if not pod:
            raise SystemExit("FAILED step 1: no GPU candidate available")
        pod_id = pod["id"]
        log(f"  pod {pod_id} created; waiting for it to be ready ...")

        with PodGuard(label=f"sdd-{cfg['served_model_name']}",
                      terminate_fn=runpod.terminate_pod) as guard:
            guard.track(pod["id"], progress_fn=lambda: 0)

            ip = port = None
            for _ in range(180):  # ~15 min
                guard.heartbeat()
                p = runpod.get_pod(pod_id)
                if is_ready(p):
                    ip, port = ssh_endpoint(p)
                    break
                time.sleep(5)
            if not ip:
                raise SystemExit("FAILED step 1: pod did not become ready in 15 min")
            log(f"  pod ready at {ip}:{port}")

            guard.set_progress(pod["id"], lambda: _startup_progress(ip, port, args.key))
            guard.phase("startup")

            log("STEP 2/5: pushing repo + launching setup/vLLM on the pod")
            for _ in range(24):
                if subprocess.run(ssh_run_cmd(ip, port, args.key, f"mkdir -p {REMOTE_DIR}"),
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                    break
                time.sleep(5)
            _run(rsync_up_cmd(ip, port, args.key, "./", REMOTE_DIR + "/", EXCLUDES), timeout=300)
            # Detach setup + vLLM (subshell, stdin from /dev/null) so ssh returns immediately
            # instead of hanging on the backgrounded process's channel. Errors land in vllm.log,
            # which the health-poll below checks (fail-fast).
            _run(ssh_run_cmd(ip, port, args.key,
                 f"(cd {REMOTE_DIR} && INSTALL_CLAUDE=0 ./scripts/setup_pod.sh {args.config} "
                 f"&& ./scripts/start_vllm.sh {args.config}) </dev/null >/workspace/vllm.log 2>&1 &"),
                 timeout=120)

            log("STEP 3/5: waiting for vLLM to serve (via SSH tunnel); fails fast on pod errors")
            tunnel = subprocess.Popen(ssh_tunnel_cmd(ip, port, args.key))
            reachable = False
            for i in range(360):  # backstop ceiling; guard MAX_STARTUP (~12 min) governs
                guard.raise_if_aborted()
                if subprocess.run(["curl", "-sf", "http://localhost:8000/health"],
                                  stdout=subprocess.DEVNULL).returncode == 0:
                    reachable = True
                    break
                if i % 12 == 11:  # ~every 60s: progress + check vLLM didn't fail to start
                    # Match fatal patterns only on non-WARNING/INFO lines: vLLM emits benign FP8
                    # warnings like "Config file not found at ..." that would otherwise false-trip
                    # 'not found' and abort a healthy (still-loading) startup.
                    chk = subprocess.run(ssh_run_cmd(ip, port, args.key,
                        "grep -vE 'WARNING|INFO' /workspace/vllm.log "
                        "| grep -qE 'not found|Traceback|RuntimeError|No available' "
                        "&& tail -25 /workspace/vllm.log || true"),
                        capture_output=True, text=True)
                    if chk.stdout.strip():
                        log("FAILED step 3: vLLM failed to start on the pod:\n" + chk.stdout)
                        raise SystemExit("vLLM startup failed")
                    log(f"  still waiting for vLLM ... ({(i + 1) * 5 // 60} min)")
                if tunnel.poll() is not None:  # tunnel died — reopen it
                    tunnel = subprocess.Popen(ssh_tunnel_cmd(ip, port, args.key))
                time.sleep(5)
            if not reachable:
                raise SystemExit("FAILED step 3: vLLM did not serve within backstop window (guard MAX_STARTUP governs)")
            log("  vLLM reachable via tunnel.")

            # Optional diagnostic (SDD_VLLM_PROBE=1): hit vLLM directly through the tunnel with a
            # tool, streaming + non-streaming, to see whether vLLM populates tool_calls or leaves
            # the call in content — isolates the vLLM tool parser from LiteLLM + Claude Code.
            if os.getenv("SDD_VLLM_PROBE") == "1":
                log("STEP 3b: probing vLLM tool parsing directly (SDD_VLLM_PROBE=1)")
                probe = subprocess.run(
                    [sys.executable, "-m", "scripts.probe_vllm_tools",
                     "http://localhost:8000/v1", served, str(out_dir / "vllm_probe.json")],
                    capture_output=True, text=True)
                log(probe.stdout + (probe.stderr or ""))

            # 4. generation in the toolchain container. Chain: Claude Code -> LiteLLM -> destream
            # proxy -> tunnel -> pod vLLM. The proxy forces the upstream vLLM call non-streaming
            # (vLLM's qwen3 streaming tool parser leaks tool calls into content; non-streaming is
            # correct) and re-emits SSE, so Claude Code still streams with tool calls intact.
            guard.phase("generation")
            guard.set_progress(pod["id"], lambda: _gen_progress(str(out_dir)))
            env = {
                "SDD_OUT_DIR": "/out",
                "ANTHROPIC_BASE_URL": "http://localhost:4000",
                "ANTHROPIC_AUTH_TOKEN": "sk-dev-lab",
                "ANTHROPIC_MODEL": "dev-model",
                "LITELLM_UPSTREAM_MODEL": f"hosted_vllm/{served}",  # -> /chat/completions, not /responses
                "VLLM_BASE": "http://localhost:8011/v1",            # LiteLLM -> destream proxy
                "DESTREAM_UPSTREAM": "http://host.docker.internal:8000",  # proxy -> pod vLLM (tunnel)
                "PYTHONPATH": "/repo",
                "MODEL": served,
            }
            log("STEP 4/5: generation in the toolchain container (agent builds the app)")
            gen = ("set -e; "
                   "python3 -m scripts.destream_proxy > /tmp/destream.log 2>&1 & "
                   "litellm --config /repo/infra/litellm/config.yaml --port 4000 > /tmp/litellm.log 2>&1 & "
                   "sleep 8; "
                   f"python3 -m scripts.run_sdd_scenario /repo/{args.scenario}")
            _run(docker_run_cmd(image, ["bash", "-lc", gen], mounts=mounts, env=env,
                                workdir="/repo", name="dail-sdd-gen"),
                 timeout=int(os.getenv("SDD_GEN_TIMEOUT", "3600")))
        # PodGuard.__exit__ terminates the pod; runpod.terminate_pod not called here.
    finally:
        if tunnel:
            tunnel.terminate()

    log("STEP 5/5: scoring gates (pod stopped)")
    _run(docker_run_cmd(
        image, ["bash", "-lc", f"python3 -m scripts.run_gates /repo/{args.scenario} /out/workspace"],
        mounts=mounts, env={"PYTHONPATH": "/repo", "MODEL": served},
        workdir="/repo", name="dail-sdd-gates", host_gateway=False), timeout=600)
    log(f"DONE. Results under {out_dir}")


if __name__ == "__main__":
    main()
