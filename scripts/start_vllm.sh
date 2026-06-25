#!/usr/bin/env bash
# Start vLLM from a model config (configs/*.yaml). Config-driven and model-independent:
# the arg list is built by scripts/lib/vllm_args.py, so adding a model is a new YAML.
set -euo pipefail

CONFIG=${1:-configs/qwen3coder.yaml}

# vllm + pyyaml live in the setup_pod venv; put it on PATH if present (run_pod.sh also
# does this, but the SDD orchestrator calls start_vllm.sh directly).
[ -d /workspace/venv ] && export PATH="/workspace/venv/bin:$PATH"

# Authenticate Hugging Face downloads when a token is available. Anonymous downloads share
# the Resolver rate limit (3k requests / 5 min per IP) and 429-stall partway through a ~30GB
# model; a token raises it to 5k/5min and scopes it per-account. Taken from the environment
# or the rsync'd .env (the token itself is never echoed); falls back to anonymous if absent.
if [ -z "${HF_TOKEN:-}" ] && [ -f .env ]; then
  HF_TOKEN=$(sed -n 's/^HF_TOKEN=//p' .env | head -1)
fi
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN
  echo "HF auth: token present"
else
  echo "HF auth: anonymous (no HF_TOKEN — downloads may hit rate limits)"
fi

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
