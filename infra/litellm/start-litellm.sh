#!/usr/bin/env bash
# Start the LiteLLM proxy in front of vLLM. Requires vLLM running on :8000.
set -euo pipefail
exec litellm --config "$(dirname "$0")/config.yaml" --port "${LITELLM_PORT:-4000}"
