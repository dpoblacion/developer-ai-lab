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

# Use the venv that setup_pod.sh created (vllm, python deps) and the claude CLI install dir.
export PATH="/workspace/venv/bin:$HOME/.local/bin:$PATH"

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
VLLM_PID=$!
echo "Waiting for vLLM (pid $VLLM_PID) ..."
# Generous ceiling (~60 min) for first-boot weight downloads of large models; the PID
# check below makes a crash bail immediately, so a high ceiling costs nothing on failure.
for _ in $(seq 1 720); do
  curl -sf http://localhost:8000/health >/dev/null 2>&1 && { echo "vLLM ready."; break; }
  # Fail fast if vLLM died (e.g. driver/CUDA error) instead of waiting out the timeout.
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "ERROR: vLLM exited during startup"; tail -60 "$LOG_DIR/vllm.log"; exit 1; }
  sleep 5
done
curl -sf http://localhost:8000/health >/dev/null 2>&1 || { echo "ERROR: vLLM not ready (timeout)"; tail -60 "$LOG_DIR/vllm.log"; exit 1; }

echo "== Starting LiteLLM =="
infra/litellm/start-litellm.sh > "$LOG_DIR/litellm.log" 2>&1 &
wait_for LiteLLM http://localhost:4000/health/readiness 60 2 || { tail -50 "$LOG_DIR/litellm.log"; exit 1; }

echo "== Smoke: 1 developer via Claude Code =="
./scripts/smoke_agent.sh

echo "== Concurrency sweep (SLO) =="
python3 -m scripts.run_concurrency_slo

echo
echo "Done. Results are under results/. Copy them locally, then DESTROY the pod."
