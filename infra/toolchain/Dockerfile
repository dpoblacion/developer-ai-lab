# Scenario-agnostic toolchain for the SDD local-agent benchmark: runs Claude Code against a
# tunneled vLLM and executes the Python harness (generation + gates). The repo is mounted at
# run time (not baked), so iteration doesn't need a rebuild. Pinned base for reproducibility.
#
# Scenario-specific runtimes live in a per-scenario overlay image that FROMs this one —
# see benchmarks/<name>/Dockerfile. The orchestrator picks the overlay automatically when a
# scenario ships its own Dockerfile.
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git python3 python3-venv python3-pip ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Gate deps preinstalled: `npx tsc` / `npx vitest` must resolve OFFLINE at gate time. If an
# agent skipped `npm install`, npx would otherwise hit the network mid-gate — and a flaky
# download would fail the agent for environmental reasons, polluting the validity filter.
RUN npm install -g typescript@5 vitest@3

# Claude Code CLI. ENV PATH covers non-login shells; the profile.d snippet covers LOGIN
# shells (the orchestrator runs `bash -lc`, and debian's /etc/profile resets PATH, which
# would otherwise drop /root/.local/bin and hide the claude binary).
RUN curl -fsSL https://claude.ai/install.sh | bash \
    && echo 'export PATH=/root/.local/bin:$PATH' > /etc/profile.d/claude-path.sh
ENV PATH="/root/.local/bin:${PATH}"

# Python deps for the harness (litellm proxy, yaml, ...)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --break-system-packages -r /tmp/requirements.txt

WORKDIR /repo
