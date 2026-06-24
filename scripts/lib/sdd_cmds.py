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


def docker_run_cmd(image, workspace, env, command, name="dail-sdd"):
    """docker run for the toolchain container: workspace + docker socket + host gateway."""
    cmd = ["docker", "run", "--rm", "--name", name,
           "-v", f"{workspace}:/workspace",
           "-v", "/var/run/docker.sock:/var/run/docker.sock",
           "--add-host", "host.docker.internal:host-gateway",
           "-w", "/workspace"]
    for key, value in env.items():
        cmd += ["-e", f"{key}={value}"]
    cmd.append(image)
    cmd += list(command)
    return cmd
