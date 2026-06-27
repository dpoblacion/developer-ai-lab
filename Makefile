# Use the repo venv if present (it carries the orchestrator deps: runpod, pyyaml, ...);
# otherwise fall back to python3. Override explicitly with `make PYTHON=/path/to/python ...`.
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
MODEL ?= qwen3-coder
HARDWARE ?= l40s
SCENARIO ?= benchmarks/todo-app/scenario.yaml

.PHONY: test orchestrate sdd-run gates setup pod concurrency sdd reap

test:
	$(PYTHON) -m unittest discover -s tests

# Run the whole benchmark on a RunPod pod from your machine
# (create -> rsync repo -> setup -> run -> fetch results -> terminate).
# Needs RUNPOD_API_KEY and: pip install -r requirements-orchestrator.txt
orchestrate:
	$(PYTHON) -m scripts.orchestrate_pod --model $(MODEL) --hardware $(HARDWARE) --scenario benchmarks/concurrency/scenario.yaml

# Prepare a fresh pod (install vLLM + deps + claude CLI). Run once per pod.
# Called remotely by orchestrate_pod.py with CONFIG=configs/_composed.yaml.
setup:
	./scripts/setup_pod.sh $(CONFIG)

# Full pod run: vLLM -> LiteLLM -> smoke -> concurrency sweep.
# Called remotely by orchestrate_pod.py with CONFIG=configs/_composed.yaml.
pod:
	SCENARIO=$(SCENARIO) ./scripts/run_pod.sh $(CONFIG)

# Concurrency sweep with SLO (vLLM must be up).
concurrency:
	$(PYTHON) -m scripts.run_concurrency_slo

# SDD benchmark, agent-local: pod serves vLLM only; agent + gates run in a local
# toolchain container. One command (create pod -> generate -> stop pod -> gates).
# Pick a scenario with SCENARIO=... (default todo-app; e.g. benchmarks/smoke/scenario.yaml).
sdd-run:
	$(PYTHON) -m scripts.orchestrate_sdd --model $(MODEL) --hardware $(HARDWARE) --scenario $(SCENARIO)

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
