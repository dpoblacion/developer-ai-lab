# Toolchain for the SDD local-agent benchmark: builds + scores the generated app, and
# runs Claude Code against a tunneled vLLM. The repo is mounted at run time (not baked),
# so iteration doesn't need a rebuild. Pinned base for reproducibility.
FROM mcr.microsoft.com/dotnet/sdk:10.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git python3 python3-venv python3-pip ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs docker.io \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/root/.local/bin:${PATH}"

# Python deps for the harness (litellm proxy, yaml, aiohttp, ...)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --break-system-packages -r /tmp/requirements.txt

WORKDIR /repo
