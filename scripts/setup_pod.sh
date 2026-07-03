#!/usr/bin/env bash
# Prepare a fresh pod: a clean venv with the Python deps + pinned vLLM, plus the
# Claude Code CLI. A venv avoids the pod's externally-managed system Python (PEP 668)
# and its distutils packages that pip can't cleanly upgrade. Run once per pod.
#
# Usage: scripts/setup_pod.sh [configs/_composed.yaml]
set -euo pipefail

CONFIG=${1:-configs/_composed.yaml}
VENV=/workspace/venv

# Stage timing: pip runs quiet (-q) to drop the repetitive "Downloading ..." spam, so each
# step prints its own duration instead — the process stays visible and timed. $SECONDS is
# bash's seconds-since-script-start; _since echoes the delta from a captured value.
_since() { echo "$(( SECONDS - ${1:-0} ))s"; }

# The claude installer drops its binary here; have it on PATH so the post-install check sees it.
export PATH="$HOME/.local/bin:$PATH"

# Camino A — image-provided vLLM (e.g. vllm/vllm-openai): reuse it instead of building a venv
# and reinstalling. Duplicating the ~4min vLLM install would waste time and risk a version
# clash with the image's torch/CUDA. We only need pyyaml on the system python for our
# config-parsing helpers (start_vllm.sh's vllm_args). The pytorch base images have no vllm on
# PATH, so this branch is skipped there and the normal venv install below runs unchanged.
if command -v vllm >/dev/null 2>&1; then
  echo "== vLLM already in the image ($(vllm --version 2>/dev/null | head -1)) — reusing it, no venv =="
  python3 -c "import yaml" 2>/dev/null \
    || python3 -m pip install -q --break-system-packages pyyaml 2>/dev/null \
    || python3 -m pip install -q pyyaml
  echo "Setup complete in ${SECONDS}s (image-provided vLLM)."
  exit 0
fi

echo "== venv ($VENV) =="
# Clean venv (NOT --system-site-packages): vllm pins a consistent torch + transformers
# set. With the pinned vllm matching the image's CUDA (see vllm_version), the torch it
# pulls (2.8.0/cu128) works with the host driver, and transformers won't clash with the
# image's newer copy.
python3 -m venv "$VENV"
PIP="$VENV/bin/pip"
$PIP install --quiet --upgrade pip

# Python deps depend on what this pod runs (INSTALL_CLAUDE is the signal):
# - vLLM-only pod (INSTALL_CLAUDE=0, the SDD benchmark): LiteLLM + the agent + gates run in
#   the LOCAL toolchain container, so the pod needs only pyyaml (config parsing); vLLM pulls
#   its own serving stack below. Keeps startup lean (requirements.txt's litellm[proxy] drags
#   in polars/boto3/cryptography and bloated startup past 10 min).
# - full-stack pod (INSTALL_CLAUDE=1, hand-driven debug pods only): LiteLLM and Claude Code
#   run ON the pod, so it needs the full requirements.
if [ "${INSTALL_CLAUDE:-1}" = "1" ]; then
  echo "== Python deps (full stack: requirements.txt) =="
  _t=$SECONDS; $PIP install -q -r requirements.txt; echo "   ↳ done in $(_since $_t)"
else
  echo "== Python deps (vLLM-only: pyyaml) =="
  _t=$SECONDS; $PIP install -q pyyaml; echo "   ↳ done in $(_since $_t)"
fi

VLLM_VERSION=$("$VENV/bin/python" -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('vllm_version', ''))")

# Optional per-model pip constraints (e.g. pinning transformers for an older vLLM).
CONSTRAINTS=()
while IFS= read -r line; do [ -n "$line" ] && CONSTRAINTS+=("$line"); done < <("$VENV/bin/python" -c "import yaml; [print(c) for c in (yaml.safe_load(open('$CONFIG')).get('pip_constraints') or [])]")

if [ -n "$VLLM_VERSION" ]; then
  echo "== vLLM $VLLM_VERSION ${CONSTRAINTS[*]:-} (pulls torch etc., ~2-3 min) =="
  _t=$SECONDS; $PIP install -q "vllm==$VLLM_VERSION" "${CONSTRAINTS[@]+"${CONSTRAINTS[@]}"}"; echo "   ↳ done in $(_since $_t)"
else
  echo "== vLLM (latest) ${CONSTRAINTS[*]:-} (pulls torch etc., ~2-3 min) =="
  _t=$SECONDS; $PIP install -q vllm "${CONSTRAINTS[@]+"${CONSTRAINTS[@]}"}"; echo "   ↳ done in $(_since $_t)"
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
echo "Setup complete in ${SECONDS}s."
