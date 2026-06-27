# Use the repo venv if present (it carries the orchestrator deps: runpod, pyyaml, ...);
# otherwise fall back to python3. Override explicitly with `make PYTHON=/path/to/python ...`.
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
CONFIG ?= configs/qwen3coder.yaml
SCENARIO ?= benchmarks/todo-app/scenario.yaml

.PHONY: test orchestrate sdd-run gates setup pod concurrency sdd reap

test:
	$(PYTHON) -m unittest discover -s tests

# Run the whole benchmark on a RunPod pod from your machine
# (create -> rsync repo -> setup -> run -> fetch results -> terminate).
# Needs RUNPOD_API_KEY and: pip install -r requirements-orchestrator.txt
orchestrate:
	$(PYTHON) -m scripts.orchestrate_pod --config $(CONFIG)

# Prepare a fresh pod (install vLLM + deps + claude CLI). Run once per pod.
setup:
	./scripts/setup_pod.sh $(CONFIG)

# Full pod run: vLLM -> LiteLLM -> smoke -> concurrency sweep.
pod:
	./scripts/run_pod.sh $(CONFIG)

# Concurrency sweep with SLO (vLLM must be up).
concurrency:
	$(PYTHON) -m scripts.run_concurrency_slo

# SDD benchmark, agent-local: pod serves vLLM only; agent + gates run in a local
# toolchain container. One command (create pod -> generate -> stop pod -> gates).
# Pick a scenario with SCENARIO=... (default todo-app; e.g. benchmarks/smoke/scenario.yaml).
sdd-run:
	$(PYTHON) -m scripts.orchestrate_sdd --config $(CONFIG) --scenario $(SCENARIO)

# Score an already-produced SDD workspace with the scenario gates (no pod needed).
# Usage: make gates WORKSPACE=results/<run>/sdd/workspace
gates:
	$(PYTHON) -m scripts.run_gates $(SCENARIO) $(WORKSPACE)

# Generation only (in-process; expects a reachable model). Mostly for debugging.
sdd:
	$(PYTHON) -m scripts.run_sdd_scenario $(SCENARIO)

# Panic button: terminate paid pods (all, or older than REAP_AGE seconds).
reap:
	$(PYTHON) -m scripts.reap_pods
