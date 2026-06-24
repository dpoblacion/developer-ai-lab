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

echo "== Python deps =="
$PIP install -r requirements.txt

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

echo "== Claude Code CLI =="
if ! command -v claude >/dev/null 2>&1; then
  curl -fsSL https://claude.ai/install.sh | bash
fi

echo "== Versions =="
"$VENV/bin/vllm" --version || echo "WARN: vllm not installed in venv"
(command -v claude >/dev/null && claude --version) || echo "WARN: claude not on PATH"
echo "Setup complete."
