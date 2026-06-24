#!/usr/bin/env bash
# One-shot pod runbook. Brings up the full stack, verifies it, runs the concurrency
# sweep, and stops. vLLM downloads the model on first start (to download_dir); for a
# large/expensive model you can pre-stage weights with scripts/prefetch_model.sh.
#
#   vLLM (serves model) -> LiteLLM (Anthropic API) -> Claude Code
#
# Usage: scripts/run_pod.sh [configs/glm5.2.yaml]
# Prereq: `claude` CLI installed; deps installed (pip install -r requirements.txt).
set -euo pipefail

CONFIG=${1:-configs/glm5.2.yaml}

# Defaults (ANTHROPIC_*, SLO_*, CONCURRENCY, HW_* ...).
set -a && . ./config.env && set +a

# Tie the served-model name across vLLM, the raw sweep, and the LiteLLM upstream.
SERVED=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['served_model_name'])")
export MODEL="$SERVED"
export LITELLM_UPSTREAM_MODEL="openai/$SERVED"

LOG_DIR=${LOG_DIR:-/workspace/logs}
mkdir -p "$LOG_DIR"

wait_for() {  # name url retries sleep
  local name=$1 url=$2 retries=$3 nap=$4
  echo "Waiting for $name ..."
  for _ in $(seq 1 "$retries"); do
    curl -sf "$url" >/dev/null 2>&1 && { echo "$name ready."; return 0; }
    sleep "$nap"
  done
  echo "ERROR: $name not ready ($url)"; return 1
}

echo "== Starting vLLM ($SERVED) =="
scripts/start_vllm.sh "$CONFIG" > "$LOG_DIR/vllm.log" 2>&1 &
wait_for vLLM http://localhost:8000/health 720 5 || { tail -50 "$LOG_DIR/vllm.log"; exit 1; }

echo "== Starting LiteLLM =="
infra/litellm/start-litellm.sh > "$LOG_DIR/litellm.log" 2>&1 &
wait_for LiteLLM http://localhost:4000/health/readiness 60 2 || { tail -50 "$LOG_DIR/litellm.log"; exit 1; }

echo "== Smoke: 1 developer via Claude Code =="
./scripts/smoke_agent.sh

echo "== Concurrency sweep (SLO) =="
python3 -m scripts.run_concurrency_slo

echo
echo "Done. Results are under results/. Copy them locally, then DESTROY the pod."
