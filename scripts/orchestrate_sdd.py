"""SDD benchmark, agent-local. The GPU pod serves vLLM only; the agent, its build-fix
loop, and the gates run in a local toolchain container that reaches the pod's vLLM over
an SSH tunnel (LiteLLM runs locally in the container). The pod stops as soon as generation
ends; gates run with the pod down.

Requires (.env): RUNPOD_API_KEY, SSH_KEY_PATH. Local Docker + the dail-toolchain image
(build: docker build -t dail-toolchain -f infra/toolchain/Dockerfile .).

Usage: python -m scripts.orchestrate_sdd [--config configs/qwen3coder.yaml]
       [--spec infra/runpod/pod.yaml] [--scenario benchmarks/scenarios/todo-app/scenario.yaml]
"""

import argparse
import os
import pathlib
import subprocess
import time

from scripts.lib.dotenv import load_dotenv
from scripts.lib.runpod_pod import (
    build_create_kwargs, is_ready, ssh_endpoint, ssh_run_cmd, rsync_up_cmd)
from scripts.lib.sdd_cmds import ssh_tunnel_cmd, docker_run_cmd

REMOTE_DIR = "/workspace/developer-ai-lab"
EXCLUDES = [".git", "results", "__pycache__", ".venv", "docs/superpowers"]
IMAGE = "dail-toolchain"


def _run(cmd):
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


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
        # 1. vLLM-only pod
        for gpu in spec["gpu_type_ids"]:
            try:
                pod = runpod.create_pod(**build_create_kwargs(spec, gpu, pub))
                break
            except Exception as exc:
                print(f"  {gpu} unavailable: {exc}")
        if not pod:
            raise SystemExit("No GPU candidate available")
        pod_id = pod["id"]
        print(f"Pod {pod_id} created.")
        ip = port = None
        for _ in range(180):
            p = runpod.get_pod(pod_id)
            if is_ready(p):
                ip, port = ssh_endpoint(p)
                break
            time.sleep(5)
        if not ip:
            raise SystemExit("Pod did not become ready")

        # 2. push repo + start vLLM (inference only; no claude/.NET on the pod)
        for _ in range(24):
            if subprocess.run(ssh_run_cmd(ip, port, args.key, f"mkdir -p {REMOTE_DIR}"),
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                break
            time.sleep(5)
        _run(rsync_up_cmd(ip, port, args.key, "./", REMOTE_DIR + "/", EXCLUDES))
        # Detach setup + vLLM (subshell, stdin from /dev/null) so ssh returns immediately
        # instead of hanging on the backgrounded process's channel. Errors land in vllm.log,
        # which the health-poll below checks (fail-fast).
        _run(ssh_run_cmd(ip, port, args.key,
             f"(cd {REMOTE_DIR} && INSTALL_CLAUDE=0 ./scripts/setup_pod.sh {args.config} "
             f"&& ./scripts/start_vllm.sh {args.config}) </dev/null >/workspace/vllm.log 2>&1 &"))

        # 3. tunnel to vLLM and wait until reachable (fail fast if vLLM died on the pod)
        tunnel = subprocess.Popen(ssh_tunnel_cmd(ip, port, args.key))
        reachable = False
        for i in range(360):
            if subprocess.run(["curl", "-sf", "http://localhost:8000/health"],
                              stdout=subprocess.DEVNULL).returncode == 0:
                reachable = True
                break
            if i % 12 == 11:  # ~every 60s, check vLLM didn't fail to start on the pod
                chk = subprocess.run(ssh_run_cmd(ip, port, args.key,
                    "grep -qE 'not found|Traceback|RuntimeError|No available' /workspace/vllm.log "
                    "&& tail -25 /workspace/vllm.log || true"),
                    capture_output=True, text=True)
                if chk.stdout.strip():
                    print("vLLM failed to start on the pod:\n" + chk.stdout)
                    raise SystemExit("vLLM startup failed")
            if tunnel.poll() is not None:  # tunnel died — reopen it
                tunnel = subprocess.Popen(ssh_tunnel_cmd(ip, port, args.key))
            time.sleep(5)
        if not reachable:
            raise SystemExit("vLLM did not become reachable via the tunnel")
        print("vLLM reachable via tunnel.")

        # 4. generation in the toolchain container (LiteLLM local -> tunnel -> pod vLLM)
        env = {
            "SDD_OUT_DIR": "/out",
            "ANTHROPIC_BASE_URL": "http://localhost:4000",
            "ANTHROPIC_AUTH_TOKEN": "sk-dev-lab",
            "ANTHROPIC_MODEL": "dev-model",
            "LITELLM_UPSTREAM_MODEL": f"hosted_vllm/{served}",  # -> /chat/completions, not /responses
            "VLLM_BASE": "http://host.docker.internal:8000/v1",
            "PYTHONPATH": "/repo",
            "MODEL": served,
        }
        gen = ("set -e; "
               "litellm --config /repo/infra/litellm/config.yaml --port 4000 > /tmp/litellm.log 2>&1 & "
               "sleep 8; "
               f"python3 -m scripts.run_sdd_scenario /repo/{args.scenario}")
        _run(docker_run_cmd(IMAGE, ["bash", "-lc", gen], mounts=mounts, env=env,
                            workdir="/repo", name="dail-sdd-gen"))
    finally:
        if tunnel:
            tunnel.terminate()
        if pod:
            print(f"Terminating pod {pod['id']} ...")
            runpod.terminate_pod(pod["id"])

    # 5. gates with the pod down, in the container
    _run(docker_run_cmd(
        IMAGE, ["bash", "-lc", f"python3 -m scripts.run_gates /repo/{args.scenario} /out/workspace"],
        mounts=mounts, env={"PYTHONPATH": "/repo", "MODEL": served},
        workdir="/repo", name="dail-sdd-gates", host_gateway=False))
    print(f"Done. Results under {out_dir}")


if __name__ == "__main__":
    main()
