# Use the repo venv if present (it carries the benchmark runner deps: runpod, pyyaml, ...);
# otherwise fall back to python3. Override explicitly with `make PYTHON=/path/to/python ...`.
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
HARDWARE ?= l40s
BENCHMARK ?= benchmarks/dev-load/scenario.yaml
MODEL ?= qwen3-coder
GPUS ?= 1
KEEP ?=

.PHONY: help test run validate gates setup sdd reap prefetch report dashboard prune-artifacts

help:  ## List available targets
	@awk 'BEGIN {FS=":.*## "} /^[a-z-]+:.*## / {printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

test:  ## Run the unit suite (offline, free)
	$(PYTHON) -m unittest discover -s tests

# Run a benchmark on a RunPod pod (PAID). The benchmark carries the task + `devs` (N); MODEL
# and HARDWARE are the run axes. Each N is tested directly (holds SLO? + $/dev) → report.json.
# Needs RUNPOD_API_KEY + SSH_KEY_PATH in .env (pip install -r requirements-orchestrator.txt).
run:  ## Run a benchmark on a RunPod pod (PAID) — BENCHMARK= MODEL= HARDWARE= GPUS= QUANT= DEVS= KEEP=1
	$(PYTHON) -m scripts.run_benchmark --benchmark $(BENCHMARK) --model $(MODEL) --hardware $(HARDWARE) --gpus $(GPUS) $(if $(QUANT),--quant $(QUANT),) $(if $(DEVS),--devs $(DEVS),) $(if $(KEEP),--keep,)

# Cheap pre-flight (PAID, but ~1 short agent): does the stack serve + an agent complete + gates run?
validate:  ## Stack pre-flight via the smoke benchmark (PAID, ~$0.10)
	$(PYTHON) -m scripts.run_benchmark --benchmark benchmarks/smoke/scenario.yaml --model $(MODEL) --hardware $(HARDWARE) --gpus $(GPUS) $(if $(KEEP),--keep,)

# Manual/debug: prepare a pod by hand (install vLLM + deps). The live run path
# (run_benchmark → bench_sdd) SSHes scripts/setup_pod.sh directly — this target is
# only for hand-driven pods. CONFIG=configs/_composed.yaml.
setup:
	./scripts/setup_pod.sh $(CONFIG)

# Score an already-produced SDD workspace with the scenario gates (no pod needed).
# Usage: make gates WORKSPACE=results/<benchmark>/<family>-<quant>/<hw>-<gpus>gpu/<run_id>/artifacts/n<N>/agent<i>/workspace
gates:  ## Re-score a produced workspace with the benchmark's gates (free, no pod)
	$(PYTHON) -m scripts.run_gates $(BENCHMARK) $(WORKSPACE)

# Generation only (in-process; expects a reachable model). Mostly for debugging.
sdd:
	$(PYTHON) -m scripts.run_sdd_scenario $(BENCHMARK)

# Panic button: terminate paid pods (all, or older than REAP_AGE seconds).
reap:  ## Panic button: terminate ALL your RunPod pods
	$(PYTHON) -m scripts.reap_pods

# Download a model's weights to the network volume with a CHEAP single pod (no GPU used for
# the download), so a later multi-GPU run finds them cached. Needs DAIL_NETWORK_VOLUME_ID +
# DAIL_DATA_CENTER_ID in .env. Usage: make prefetch REPO=zai-org/GLM-5.2-FP8   (PAID, small)
prefetch:  ## Cache a model's weights on the network volume via a cheap pod (PAID, small)
	$(PYTHON) -m scripts.prefetch_to_volume $(REPO)

dashboard:  ## Launch the Streamlit + Plotly result viewer (auto-installs requirements-report.txt)
	$(PYTHON) -m pip install -q -r requirements-report.txt
	$(PYTHON) -m streamlit run scripts/dashboard.py --server.headless=true --browser.gatherUsageStats=false

# Drop artifacts/ (workspace + transcripts) of every run except the latest per config; keeps report.json.
prune-artifacts:  ## Drop artifacts/ of superseded runs (keeps every report.json)
	$(PYTHON) -m scripts.prune_artifacts
