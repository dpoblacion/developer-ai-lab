"""Download a model's weights to a RunPod network volume using a CHEAP single pod, so a
later expensive multi-GPU run finds them cached (no re-download). The download uses no GPU
— it's network + disk — so this pod is a fraction of the serving cost.

Usage: python -m scripts.prefetch_to_volume <hf-repo-id>
Needs in .env: RUNPOD_API_KEY, SSH_KEY_PATH, DAIL_NETWORK_VOLUME_ID, DAIL_DATA_CENTER_ID,
optional HF_TOKEN. The GPU used for the pod is PREFETCH_GPU (default a cheap-ish one in the
volume's data center). PAID, but tiny vs a full serving run.
"""
import os
import subprocess
import sys

from scripts.lib.dotenv import load_dotenv
from scripts.lib.orchestrator import (
    create_and_track, log, pod_progress, wait_for_ssh, wait_ready)
from scripts.lib.pod_guard import PodGuard
from scripts.lib.runpod_pod import ssh_endpoint, ssh_run_cmd

# Base image just needs python3 + pip to run huggingface_hub; CUDA version is irrelevant
# for a download, so reuse the pinned toolchain-ish pytorch base.
PREFETCH_IMAGE = "runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404"
DOWNLOAD_TIMEOUT = 7200   # 2h ceiling for the ssh download call


def hf_download_cmd(repo, cache_dir, hf_token):
    """Remote shell that installs hf libs and snapshot_downloads `repo` into `cache_dir`
    (the network-volume-backed HF cache). hf_transfer speeds the multi-hundred-GB pull."""
    token = f"HF_TOKEN={hf_token} " if hf_token else ""
    py = (f"from huggingface_hub import snapshot_download; "
          f"snapshot_download('{repo}', cache_dir='{cache_dir}')")
    # --break-system-packages: the base image's system python is PEP-668 externally-managed;
    # this pod is ephemeral (download only), so installing into it is fine.
    return (f"pip install -q --break-system-packages hf_transfer huggingface_hub && "
            f"HF_HUB_ENABLE_HF_TRANSFER=1 {token}python3 -c \"{py}\"")


def prefetch_spec(gpu, network_volume_id, data_center_id):
    """A cheap single-GPU pod pinned to the volume's data center. The weights land on the
    network volume (/workspace), so the container disk only needs room for the hf libs."""
    return {
        "name": "dail-prefetch",
        "image": PREFETCH_IMAGE,
        "gpu_type_ids": [gpu],
        "gpu_count": 1,
        "container_disk_gb": 30,
        "network_volume_id": network_volume_id,
        "data_center_id": data_center_id,
    }


def main():
    load_dotenv()
    repo = sys.argv[1] if len(sys.argv) > 1 else "zai-org/GLM-5.2-FP8"
    api_key = os.environ.get("RUNPOD_API_KEY")
    key = os.environ.get("SSH_KEY_PATH")
    vol = os.environ.get("DAIL_NETWORK_VOLUME_ID")
    dc = os.environ.get("DAIL_DATA_CENTER_ID")
    if not (api_key and key and vol and dc):
        raise SystemExit("Set RUNPOD_API_KEY, SSH_KEY_PATH, DAIL_NETWORK_VOLUME_ID, "
                         "DAIL_DATA_CENTER_ID in .env")
    key = os.path.expanduser(key)
    pub = open(key + ".pub").read().strip()
    hf = os.environ.get("HF_TOKEN", "")
    gpu = os.environ.get("PREFETCH_GPU", "NVIDIA H100 NVL")

    import runpod
    runpod.api_key = api_key
    spec = prefetch_spec(gpu, vol, dc)

    with PodGuard(label="dail-prefetch", terminate_fn=runpod.terminate_pod,
                  list_pods_fn=getattr(runpod, "get_pods", None),
                  max_run=DOWNLOAD_TIMEOUT + 600) as guard:
        log(f"creating cheap prefetch pod ({gpu}) in {dc} on volume {vol}")
        pod = create_and_track(runpod, spec, pub, guard)
        pod = wait_ready(runpod, pod["id"])
        ip, port = ssh_endpoint(pod)
        wait_for_ssh(ip, port, key)
        # Progress = bytes in the HF cache; keeps the watchdog from false-aborting the
        # long download (it's genuinely making progress the whole time).
        guard.set_progress(pod["id"], lambda: pod_progress(ip, port, key))
        guard.phase("generation")   # no max_phase; bounded by stall (progress) + max_run
        log(f"downloading {repo} -> /workspace/huggingface (no GPU used; this is the cheap part)")
        r = subprocess.run(ssh_run_cmd(ip, port, key, hf_download_cmd(repo, "/workspace/huggingface", hf)),
                           timeout=DOWNLOAD_TIMEOUT)
        if r.returncode != 0:
            raise SystemExit("download failed")
    log(f"DONE — {repo} cached on volume {vol}. GLM runs there now start in minutes.")


if __name__ == "__main__":
    main()
