# Scenario-agnostic toolchain for the SDD local-agent benchmark: runs Claude Code against a
# tunneled vLLM and executes the Python harness (generation + gates). The repo is mounted at
# run time (not baked), so iteration doesn't need a rebuild. Pinned base for reproducibility.
#
# Scenario-specific runtimes (e.g. the .NET SDK the todo-app gates build/score with) live in
# a per-scenario overlay image that FROMs this one — see benchmarks/scenarios/<name>/Dockerfile.
# The orchestrator picks the overlay automatically when a scenario ships its own Dockerfile.
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git python3 python3-venv python3-pip ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs docker.io \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI. ENV PATH covers non-login shells; the profile.d snippet covers LOGIN
# shells (the orchestrator runs `bash -lc`, and debian's /etc/profile resets PATH, which
# would otherwise drop /root/.local/bin and hide the claude binary).
RUN curl -fsSL https://claude.ai/install.sh | bash \
    && echo 'export PATH=/root/.local/bin:$PATH' > /etc/profile.d/claude-path.sh
ENV PATH="/root/.local/bin:${PATH}"

# Python deps for the harness (litellm proxy, yaml, aiohttp, ...)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --break-system-packages -r /tmp/requirements.txt

WORKDIR /repo
