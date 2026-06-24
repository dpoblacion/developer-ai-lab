"""Pure command builders for the SDD local-agent orchestrator (offline-testable).

The orchestrator runs the agent + gates in a local toolchain container and reaches the
pod's vLLM over an SSH tunnel; these build the exact ssh/docker command lines.
"""

_SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
             "-o", "PreferredAuthentications=publickey", "-o", "PasswordAuthentication=no"]


def ssh_tunnel_cmd(ip, port, key_path, local=8000, remote=8000):
    """Background SSH local-forward: localhost:local -> pod:remote (vLLM)."""
    return ["ssh", "-N", "-L", f"{local}:localhost:{remote}",
            "-p", str(port), "-i", key_path, *_SSH_OPTS, f"root@{ip}"]


def docker_run_cmd(image, command, mounts=(), env=None, workdir=None,
                   name="dail-sdd", host_gateway=True):
    """docker run for the toolchain container.

    mounts: list of (host_path, container_path). host_gateway adds host.docker.internal
    so the container can reach the host's SSH tunnel to the pod's vLLM.
    """
    cmd = ["docker", "run", "--rm", "--name", name]
    for host, container in mounts:
        cmd += ["-v", f"{host}:{container}"]
    if host_gateway:
        cmd += ["--add-host", "host.docker.internal:host-gateway"]
    if workdir:
        cmd += ["-w", workdir]
    for key, value in (env or {}).items():
        cmd += ["-e", f"{key}={value}"]
    cmd.append(image)
    cmd += list(command)
    return cmd
