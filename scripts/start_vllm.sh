#!/usr/bin/env bash
# Start vLLM from a model config (configs/*.yaml). Config-driven and model-independent:
# the arg list is built by scripts/lib/vllm_args.py, so adding a model is a new YAML.
set -euo pipefail

CONFIG=${1:-configs/qwen3coder.yaml}

# Export the config's env block (e.g. VLLM_USE_FLASHINFER_SAMPLER).
while IFS='=' read -r key value; do
  [ -n "$key" ] && export "$key=$value"
done < <(python3 -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); [print(f'{k}={v}') for k, v in (cfg.get('env') or {}).items()]")

# Ensure the model download dir exists.
DOWNLOAD_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['download_dir'])")
mkdir -p "$DOWNLOAD_DIR"

# Build the vllm serve arg list and launch.
ARGS=()
while IFS= read -r line; do ARGS+=("$line"); done < <(python3 -m scripts.lib.vllm_args "$CONFIG")

exec vllm serve "${ARGS[@]}"
