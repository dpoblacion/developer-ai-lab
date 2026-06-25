#!/usr/bin/env bash
# Prepare a fresh pod: a clean venv with the Python deps + pinned vLLM, plus the
# Claude Code CLI. A venv avoids the pod's externally-managed system Python (PEP 668)
# and its distutils packages that pip can't cleanly upgrade. Run once per pod.
#
# Usage: scripts/setup_pod.sh [configs/qwen3coder.yaml]
set -euo pipefail

CONFIG=${1:-configs/qwen3coder.yaml}
VENV=/workspace/venv

# The claude installer drops its binary here; have it on PATH so the post-install check sees it.
export PATH="$HOME/.local/bin:$PATH"

echo "== venv ($VENV) =="
# Clean venv (NOT --system-site-packages): vllm pins a consistent torch + transformers
# set. With the pinned vllm matching the image's CUDA (see vllm_version), the torch it
# pulls (2.8.0/cu128) works with the host driver, and transformers won't clash with the
# image's newer copy.
python3 -m venv "$VENV"
PIP="$VENV/bin/pip"
$PIP install --quiet --upgrade pip

# A vLLM-only pod needs just pyyaml (for config parsing here + in start_vllm.sh); vLLM pulls
# the rest of its serving stack below. We deliberately do NOT install requirements.txt — that
# carries litellm[proxy] and its heavy transitive deps (polars, boto3, cryptography, …) which
# run in the LOCAL toolchain container, never on the pod, and bloated startup past 10 min.
echo "== Python deps (pyyaml; vLLM pulls its own stack) =="
$PIP install pyyaml

VLLM_VERSION=$("$VENV/bin/python" -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('vllm_version', ''))")

# Optional per-model pip constraints (e.g. pinning transformers for an older vLLM).
CONSTRAINTS=()
while IFS= read -r line; do [ -n "$line" ] && CONSTRAINTS+=("$line"); done < <("$VENV/bin/python" -c "import yaml; [print(c) for c in (yaml.safe_load(open('$CONFIG')).get('pip_constraints') or [])]")

if [ -n "$VLLM_VERSION" ]; then
  echo "== vLLM $VLLM_VERSION ${CONSTRAINTS[*]:-} =="
  $PIP install "vllm==$VLLM_VERSION" "${CONSTRAINTS[@]+"${CONSTRAINTS[@]}"}"
else
  echo "== vLLM (latest) ${CONSTRAINTS[*]:-} =="
  $PIP install vllm "${CONSTRAINTS[@]+"${CONSTRAINTS[@]}"}"
fi

# Skip on an inference-only pod (SDD runs the agent locally): INSTALL_CLAUDE=0.
if [ "${INSTALL_CLAUDE:-1}" = "1" ]; then
  echo "== Claude Code CLI =="
  if ! command -v claude >/dev/null 2>&1; then
    curl -fsSL https://claude.ai/install.sh | bash
  fi
fi

echo "== Versions =="
"$VENV/bin/vllm" --version || echo "WARN: vllm not installed in venv"
if [ "${INSTALL_CLAUDE:-1}" = "1" ]; then
  (command -v claude >/dev/null && claude --version) || echo "WARN: claude not on PATH"
fi
echo "Setup complete."
