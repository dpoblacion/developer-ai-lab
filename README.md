# Developer AI Lab

A reproducible lab for evaluating AI models across the **whole software development
lifecycle** — not just code generation, and not just tokens/second. The goal is real
engineering productivity and the **hardware needed to support a development team** when
self-hosting models.

## What it answers

- Which model develops better software, and follows an SDD process best?
- How many concurrent developers can one self-hosted model support on given hardware?
- What is the optimal hardware for a model and a team size?
- When does self-hosting make sense versus commercial APIs?

## How models are driven

Everything runs through the agent we actually use, kept model-independent:

```
Claude Code (headless) -> LiteLLM (Anthropic API) -> vLLM (OpenAI API) -> model
```

Swapping models is a config change (`configs/*.yaml` for serving; the LiteLLM alias
via `LITELLM_UPSTREAM_MODEL`) — never new code. Works for Qwen, GLM, MiniMax, DeepSeek, etc.

## Benchmarks

- **Concurrency & capacity** (`scripts/run_concurrency_slo.py`) — streams concurrent
  requests, measures per-stream TTFT and decode throughput, and finds the highest
  concurrency that holds an SLO (the knee) — the capacity of the hardware for the model.
  Current focus: **GLM-5.2** team sizing.
- **SDD lifecycle** (`scripts/run_sdd_scenario.py`) — drives the model through a full
  Spec-Driven-Development flow (spec → acceptance → architecture → plan → backend →
  frontend → Aspire → tests → review) building a real Todo App, scored by deterministic gates.

## Running

Experiments run on ephemeral GPU pods (create → run → copy results → destroy), driven
from your machine via the RunPod API.

One-time local setup:

```bash
cp .env.example .env        # fill RUNPOD_API_KEY and SSH_KEY_PATH
pip install -r requirements-orchestrator.txt
```

(Register your SSH public key with your RunPod account.) Then a full run is one command —
it creates the pod, rsyncs the repo up, runs the benchmark, pulls results back, and
terminates the pod so per-second billing stops:

```bash
make orchestrate CONFIG=configs/qwen3coder.yaml
```

The pod spec (GPU candidates, disk, image) lives in `infra/runpod/pod.yaml`. Inside the
pod the orchestrator runs `make setup` (install vLLM + deps + claude CLI) then `make pod`
(vLLM → LiteLLM → smoke → concurrency sweep); you can also run those by hand over SSH.

Raw output under `results/` is gitignored and pulled to your machine — the pod has no git
credentials, so nothing is pushed from it.

## Layout

- `configs/` — per-model vLLM serving configs (the only per-model change).
- `scripts/` — runners + `scripts/lib/` (tested pure logic: slo, hardware, …).
- `benchmarks/scenarios/` — SDD scenarios (per-phase prompts + gates).
- `infra/litellm/` — the Anthropic↔OpenAI translation proxy.
- `infra/runpod/` — pod spec for the orchestrator (`scripts/orchestrate_pod.py`).
- `docs/` — goals, methodology, results.

Run `make test` for the unit suite. See `docs/` for goals and methodology.
