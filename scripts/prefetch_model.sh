#!/usr/bin/env bash
# Pre-download a model's weights to the persistent volume. Run this on a CHEAP pod
# (no/!small GPU) so the expensive serving node never pays for the download.
# The weights land in the config's download_dir (HF cache), where start_vllm.sh reads them.
#
# Usage: scripts/prefetch_model.sh [configs/models/<model>.yaml]
set -euo pipefail

CONFIG=${1:-configs/models/qwen3-coder-fp8.yaml}

# This script is the ONLY consumer of the HF download libs — install them here instead of
# bloating requirements.txt (which the toolchain image and debug pods install in full).
python3 -m pip install -q pyyaml huggingface_hub hf_transfer

export HF_HUB_ENABLE_HF_TRANSFER=1   # faster large downloads (needs hf_transfer)

python3 - "$CONFIG" <<'PY'
import sys
import yaml
from huggingface_hub import snapshot_download

cfg = yaml.safe_load(open(sys.argv[1]))
repo = cfg["model"]
cache_dir = cfg["download_dir"]
print(f"Downloading {repo} -> {cache_dir}", flush=True)
path = snapshot_download(repo_id=repo, cache_dir=cache_dir)
print(f"Done: {path}")
PY
