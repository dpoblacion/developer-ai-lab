"""Pure helpers for the RunPod pod orchestrator.

No ``runpod`` SDK import here, so these unit-test offline. The orchestrator
(scripts/run_benchmark.py via scripts/lib/orchestrator.py) imports the SDK and uses
these to build create args, detect readiness, and find the SSH endpoint.
"""


def build_create_kwargs(spec, gpu_type_id=None, public_key=None):
    """Map a pod spec (configs/hardware/<hw>.yaml) to runpod.create_pod kwargs.

    ``public_key`` (contents of an .pub) is injected as the PUBLIC_KEY env var, which
    RunPod images write into authorized_keys on boot — that's what authorizes our SSH.
    """
    env = dict(spec.get("env", {}))
    if public_key:
        env["PUBLIC_KEY"] = public_key
    return {
        "name": spec.get("name", "developer-ai-lab"),
        "image_name": spec["image"],
        "gpu_type_id": gpu_type_id or spec["gpu_type_ids"][0],
        "gpu_count": spec.get("gpu_count", 1),
        "container_disk_in_gb": spec.get("container_disk_gb", 50),
        "volume_in_gb": spec.get("volume_gb", 0),
        "volume_mount_path": spec.get("volume_mount_path", "/workspace"),
        "ports": spec.get("ports", "8000/http,22/tcp"),
        "support_public_ip": True,
        "start_ssh": True,
        # Optional persistent weight cache: a RunPod network volume mounted at
        # volume_mount_path (/workspace), so /workspace/huggingface survives across pods and
        # a huge model (GLM's 744GB) is downloaded once. The volume is pinned to one data
        # center, so the pod must be created there. Both are None unless configured.
        "network_volume_id": spec.get("network_volume_id"),
        "data_center_id": spec.get("data_center_id"),
        "env": env,
    }


def is_ready(pod):
    """A pod is ready once its runtime exists and its ports are populated."""
    runtime = pod.get("runtime")
    return bool(runtime and runtime.get("ports"))


def ssh_endpoint(pod):
    """Return (ip, public_port) for full SSH (public port 22), or None."""
    runtime = pod.get("runtime") or {}
    for port in runtime.get("ports", []):
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            return port["ip"], port["publicPort"]
    return None


def ssh_opts(port, key_path):
    """Common ssh flags for a RunPod pod.

    Host-key checking is intentionally disabled: pods are ephemeral with rotating IPs,
    so known-hosts would churn. Acceptable for a throwaway benchmark pod — do NOT copy
    this into a context where MITM protection matters. LogLevel=ERROR silences the
    "Warning: Permanently added ... to the list of known hosts" line that disabling
    known-hosts would otherwise print on every connection, while keeping real errors.
    """
    return ["-p", str(port), "-i", key_path,
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "PreferredAuthentications=publickey", "-o", "PasswordAuthentication=no",
            "-o", "ConnectTimeout=10", "-o", "LogLevel=ERROR"]


def ssh_run_cmd(ip, port, key_path, remote_cmd):
    """ssh command list that runs ``remote_cmd`` on the pod."""
    return ["ssh", *ssh_opts(port, key_path), f"root@{ip}", remote_cmd]


def rsync_up_cmd(ip, port, key_path, local_dir, remote_dir, excludes=()):
    """rsync command list to push local_dir -> pod:remote_dir.

    Uses --delete (mirror). Safe because it is scoped to remote_dir and the things that
    must survive a re-push live outside it or are excluded: model weights cache in
    download_dir (/workspace/huggingface, outside) and results/ (in excludes). Keep it
    that way — moving download_dir inside remote_dir would wipe weights on every push.
    """
    cmd = ["rsync", "-az", "--delete"]
    for exclude in excludes:
        cmd += ["--exclude", exclude]
    cmd += ["-e", "ssh " + " ".join(ssh_opts(port, key_path)),
            local_dir, f"root@{ip}:{remote_dir}"]
    return cmd
