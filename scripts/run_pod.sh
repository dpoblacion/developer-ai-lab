#!/usr/bin/env bash
# One-shot pod runbook. Brings up the full stack, verifies it, runs the concurrency
# sweep, and stops. vLLM downloads the model on first start (to download_dir); for a
# large/expensive model you can pre-stage weights with scripts/prefetch_model.sh.
#
#   vLLM (serves model) -> LiteLLM (Anthropic API) -> Claude Code
#
# Usage: scripts/run_pod.sh [configs/glm5.2.yaml]
# Prereq: run scripts/setup_pod.sh first (installs vLLM + deps in a venv + the claude CLI).
set -euo pipefail

CONFIG=${1:-configs/glm5.2.yaml}
SCENARIO=${SCENARIO:-benchmarks/concurrency/scenario.yaml}

# Defaults (ANTHROPIC_*, BASE_URL, ...).
set -a && . ./config.env && set +a

# Export sweep params from the benchmark def for run_concurrency_slo.
# These override any sweep vars that may have been in config.env.
export CONCURRENCY=$(python3 -c "import yaml;print(','.join(str(x) for x in yaml.safe_load(open('$SCENARIO'))['levels']))")
export MAX_TOKENS=$(python3 -c "import yaml;print(yaml.safe_load(open('$SCENARIO')).get('max_tokens',512))")
export SLO_MAX_TTFT=$(python3 -c "import yaml;print(yaml.safe_load(open('$SCENARIO'))['slo']['max_ttft'])")
export SLO_MIN_TPS=$(python3 -c "import yaml;print(yaml.safe_load(open('$SCENARIO'))['slo']['min_tps'])")

# Use the venv that setup_pod.sh created (vllm, python deps) and the claude CLI install dir.
export PATH="/workspace/venv/bin:$HOME/.local/bin:$PATH"

# On the pod, LiteLLM reaches vLLM on localhost (the SDD container overrides this).
export VLLM_BASE="${VLLM_BASE:-http://localhost:8000/v1}"

# Tie the served-model name across vLLM, the raw sweep, and the LiteLLM upstream.
SERVED=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['served_model_name'])")
export MODEL="$SERVED"
# hosted_vllm/ routes to vLLM's /chat/completions (openai/ uses /responses, which needs
# Harmony for tool calls and breaks Claude Code's tools on non-GPT-OSS models).
export LITELLM_UPSTREAM_MODEL="hosted_vllm/$SERVED"

LOG_DIR=${LOG_DIR:-/workspace/logs}
mkdir -p "$LOG_DIR"

# Stage timing so each step's duration (and the total) is visible. $SECONDS = seconds
# since this script started; _since echoes the delta from a captured value.
_since() { echo "$(( SECONDS - ${1:-0} ))s"; }

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
_t=$SECONDS
scripts/start_vllm.sh "$CONFIG" > "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
echo "Waiting for vLLM (pid $VLLM_PID) — downloads + loads the model, can take several min ..."
# Generous ceiling (~60 min) for first-boot weight downloads of large models; the PID
# check below makes a crash bail immediately, so a high ceiling costs nothing on failure.
for _ in $(seq 1 720); do
  curl -sf http://localhost:8000/health >/dev/null 2>&1 && { echo "   ↳ vLLM ready in $(_since $_t)"; break; }
  # Fail fast if vLLM died (e.g. driver/CUDA error) instead of waiting out the timeout.
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "ERROR: vLLM exited during startup"; tail -60 "$LOG_DIR/vllm.log"; exit 1; }
  sleep 5
done
curl -sf http://localhost:8000/health >/dev/null 2>&1 || { echo "ERROR: vLLM not ready (timeout)"; tail -60 "$LOG_DIR/vllm.log"; exit 1; }

echo "== Starting LiteLLM =="
_t=$SECONDS
infra/litellm/start-litellm.sh > "$LOG_DIR/litellm.log" 2>&1 &
wait_for LiteLLM http://localhost:4000/health/readiness 60 2 || { tail -50 "$LOG_DIR/litellm.log"; exit 1; }
echo "   ↳ LiteLLM ready in $(_since $_t)"

echo "== Smoke: 1 developer via Claude Code =="
_t=$SECONDS
./scripts/smoke_agent.sh
echo "   ↳ smoke in $(_since $_t)"

echo "== Concurrency sweep (SLO) =="
_t=$SECONDS
python3 -m scripts.run_concurrency_slo
echo "   ↳ sweep in $(_since $_t)"

echo
echo "Done in ${SECONDS}s (pod run, excl. setup). Results under results/ — copy them, then DESTROY the pod."
