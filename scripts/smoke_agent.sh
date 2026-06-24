#!/usr/bin/env bash
# End-to-end smoke test: proves Claude Code -> LiteLLM -> vLLM works and that
# per-run metrics are captured. Requires vLLM (:8000) and LiteLLM (:4000) up.
set -euo pipefail
set -a && . ./config.env && set +a
python3 -m scripts.run_agent "Create a file hello.txt containing the word PONG." smoke
ls results/*/smoke/metrics.json >/dev/null 2>&1 && echo "SMOKE OK"
