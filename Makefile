PYTHON ?= python3
CONFIG ?= configs/qwen3coder.yaml

.PHONY: test orchestrate setup pod concurrency sdd

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

# SDD scenario (agentic, via Claude Code).
sdd:
	$(PYTHON) -m scripts.run_sdd_scenario benchmarks/scenarios/todo-app/scenario.yaml
