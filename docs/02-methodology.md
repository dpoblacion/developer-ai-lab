# Methodology

## Infrastructure: ephemeral GPU pods

Experiments run on temporary GPU pods:

1. Create a pod and clone the repo.
2. `make setup CONFIG=…` — install vLLM (pinned per model), Python deps, and the
   Claude Code CLI. The base image ships none of these.
3. `make pod CONFIG=…` — bring up the stack and run the benchmark.
4. Copy `results/` to the local machine (the pod has no git credentials).
5. Destroy the pod.

Pods are never turned into permanent development environments. vLLM downloads the model
on first start (or pre-stage with `scripts/prefetch_model.sh`).

## Serving and model independence

vLLM serves the model under test via its OpenAI-compatible API. A constant LiteLLM alias
(`dev-model`) is exposed to Claude Code; the upstream served model is set per run via
`LITELLM_UPSTREAM_MODEL`. Only `configs/<model>.yaml` changes between models, so every
benchmark is identical across Qwen, GLM, MiniMax, etc.

```
Claude Code (Anthropic API) -> LiteLLM -> vLLM (OpenAI API) -> model
```

## Concurrency & capacity (team sizing)

`scripts/run_concurrency_slo.py` fires N concurrent *streaming* requests directly at
vLLM for each level in `CONCURRENCY`, measuring per-stream **TTFT** and **decode
throughput**.

- **SLO**: a developer is "served acceptably" when the median stream meets
  `SLO_MAX_TTFT` (default ≤ 2 s) and `SLO_MIN_TPS` (default ≥ 20 tok/s).
- **Knee**: the highest concurrency whose whole prefix still passes the SLO — the maximum
  concurrent streams the node sustains for this model.

The hardware (GPU/CPU/RAM) is auto-detected and written to `env.json`, so each run is
self-describing. Comparing the knee across hardware configs identifies the optimal
hardware. Translating the knee into developer counts and cost is decided later, from this
captured output — the harness itself measures, it does not hardcode prices.

## SDD benchmark (agentic)

`scripts/run_sdd_scenario.py` drives Claude Code, phase by phase, through the SDD flow
(spec → acceptance criteria → architecture → plan → backend → frontend → .NET Aspire →
tests → review), producing real artifacts in a clean workspace. Per phase it captures
tool calls, tokens, turns, and wall time.

**Scoring (two layers, kept separate):**

- **Hard score (deterministic):** pass/fail gates — does it build, do tests pass, does
  Aspire start, does the API respond, does the frontend build — plus presence/order of
  the SDD artifacts. Source of truth.
- **Soft score (qualitative, planned):** a fixed, pinned judge model scores architecture
  and code quality and SDD adherence with a rubric. Reported separately, labeled
  non-deterministic.

## Results

Raw per-run output under `results/` is gitignored and collected on the local machine. A
committable reporting format will be defined from that output once we have real runs.

## Reproducibility

The capacity measurements and the hard score are deterministic and runnable from scratch.
The soft score is acknowledged as non-deterministic and isolated.
