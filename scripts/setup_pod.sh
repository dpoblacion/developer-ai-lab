#!/usr/bin/env bash
# Prepare a fresh pod: install Python deps, the pinned vLLM for this model, and the
# Claude Code CLI. The base image (e.g. runpod/pytorch) ships none of these.
# Run once per pod, before run_pod.sh.
#
# Usage: scripts/setup_pod.sh [configs/qwen3coder.yaml]
set -euo pipefail

CONFIG=${1:-configs/qwen3coder.yaml}

echo "== Python deps =="
pip install -r requirements.txt

VLLM_VERSION=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('vllm_version', ''))")
if [ -n "$VLLM_VERSION" ]; then
  echo "== vLLM $VLLM_VERSION =="
  pip install "vllm==$VLLM_VERSION"
else
  echo "== vLLM (latest) =="
  pip install vllm
fi

echo "== Claude Code CLI =="
if ! command -v claude >/dev/null 2>&1; then
  curl -fsSL https://claude.ai/install.sh | bash
fi

echo "== Versions =="
command -v vllm >/dev/null && vllm --version || echo "WARN: vllm not on PATH"
command -v claude >/dev/null && claude --version || echo "WARN: claude not on PATH (open a new shell or check PATH)"
echo "Setup complete."
