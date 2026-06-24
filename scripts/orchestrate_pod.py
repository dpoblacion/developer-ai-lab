"""Drive a full benchmark run on a RunPod GPU pod, end to end, from the local machine.

Creates a pod from a spec, waits until it's ready, rsyncs this repo up, runs setup + the
benchmark over SSH, rsyncs results back, then terminates the pod so per-second billing
stops. The in-pod stack is unchanged (make setup / make pod). No git on the pod.

Requires: RUNPOD_API_KEY in the environment, and an SSH public key registered with the
RunPod account (the matching private key passed via --key).

Usage:
  python -m scripts.orchestrate_pod [--spec infra/runpod/pod.yaml] \
      [--config configs/qwen3coder.yaml] [--key ~/.ssh/id_ed25519] [--keep]
"""

import argparse
import os
import pathlib
import subprocess
import time

from scripts.lib.dotenv import load_dotenv
from scripts.lib.runpod_pod import (
    build_create_kwargs, is_ready, ssh_endpoint,
    ssh_run_cmd, rsync_up_cmd, rsync_down_cmd)

REMOTE_DIR = "/workspace/developer-ai-lab"
EXCLUDES = [".git", "results", "__pycache__", ".venv", "docs/superpowers"]


def _run(cmd):
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def create_with_fallback(runpod, spec):
    last = None
    for gpu in spec["gpu_type_ids"]:
        try:
            print(f"Creating pod on {gpu} ...")
            return runpod.create_pod(**build_create_kwargs(spec, gpu))
        except Exception as exc:  # bad id or no capacity — try the next candidate
            print(f"  {gpu} unavailable: {exc}")
            last = exc
    raise SystemExit(f"No GPU candidate available: {last}")


def wait_ready(runpod, pod_id, retries=180, nap=5):
    for _ in range(retries):
        pod = runpod.get_pod(pod_id)
        if is_ready(pod):
            return pod
        time.sleep(nap)
    raise SystemExit("Pod did not become ready in time")


def wait_for_ssh(ip, port, key, retries=24, nap=5):
    """Probe SSH with an idempotent mkdir until sshd accepts (ports up != sshd ready)."""
    cmd = ssh_run_cmd(ip, port, key, f"mkdir -p {REMOTE_DIR}")
    for _ in range(retries):
        if subprocess.run(cmd).returncode == 0:
            return
        time.sleep(nap)
    raise SystemExit("SSH did not become available")


def main():
    load_dotenv()  # local secrets from .env (RUNPOD_API_KEY, SSH_KEY_PATH); env wins

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="infra/runpod/pod.yaml")
    ap.add_argument("--config", default="configs/qwen3coder.yaml")
    ap.add_argument("--key", default=os.environ.get("SSH_KEY_PATH"))
    ap.add_argument("--keep", action="store_true", help="do not terminate the pod at the end")
    args = ap.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY is not set (put it in .env or export it).")
    if not args.key:
        raise SystemExit("SSH key not set — set SSH_KEY_PATH in .env or pass --key.")
    args.key = os.path.expanduser(args.key)  # ~ is not expanded when passed as an argv item

    import yaml
    import runpod
    runpod.api_key = api_key

    spec = yaml.safe_load(pathlib.Path(args.spec).read_text())

    pod = create_with_fallback(runpod, spec)
    pod_id = pod["id"]
    print(f"Pod {pod_id} created.")

    try:
        pod = wait_ready(runpod, pod_id)
        ip, port = ssh_endpoint(pod)
        print(f"SSH at {ip}:{port}")
        wait_for_ssh(ip, port, args.key)  # mkdir -p {REMOTE_DIR} happens as the probe
        _run(rsync_up_cmd(ip, port, args.key, "./", REMOTE_DIR + "/", EXCLUDES))

        remote = (f"cd {REMOTE_DIR} && make setup CONFIG={args.config} "
                  f"&& make pod CONFIG={args.config}")
        _run(ssh_run_cmd(ip, port, args.key, remote))

        pathlib.Path("results").mkdir(exist_ok=True)
        _run(rsync_down_cmd(ip, port, args.key, REMOTE_DIR + "/results/", "results/"))
        print("Results pulled to ./results/")
    finally:
        if args.keep:
            print(f"Pod {pod_id} left running (--keep). Terminate it to stop billing.")
        else:
            print(f"Terminating pod {pod_id} ...")
            runpod.terminate_pod(pod_id)


if __name__ == "__main__":
    main()
