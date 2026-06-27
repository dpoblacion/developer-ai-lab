# Methodology

## Principle: the GPU pod runs inference only

The GPU is needed only for **model inference**. Everything else — the agent, builds,
tests, load generation, scoring — runs off the GPU. Each benchmark places work accordingly,
and both run on **ephemeral pods** driven from the local machine via the RunPod API
(create → run → pull results → terminate). Pods are never permanent dev environments.

## Serving and model independence

vLLM serves the model under test via its OpenAI-compatible API. A constant LiteLLM alias
(`dev-model`) is exposed to Claude Code; the upstream served model is set per run via
`LITELLM_UPSTREAM_MODEL`, and where vLLM is reachable via `VLLM_BASE` (localhost on the
pod, `host.docker.internal` from the SDD container). Model selection is a `--model` /
`--hardware` flag: `configs/models/<model>.yaml` + `configs/hardware/<hw>.yaml` are composed
at run time (benchmark `serving:` blocks add per-scenario overrides), so every benchmark
is identical across Qwen, GLM, MiniMax, etc.

```
Claude Code (Anthropic API) -> LiteLLM -> vLLM (OpenAI API) -> model
```

## Benchmark 1 — Concurrency & capacity (`make orchestrate`)

Pure inference load, so it runs **on the pod**. `scripts/orchestrate_pod.py` creates the
pod, installs vLLM (`make setup`), and runs `scripts/run_concurrency_slo.py`, which fires
N concurrent *streaming* requests at vLLM for each level in `CONCURRENCY`, measuring
per-stream **TTFT** and **decode throughput**.

- **SLO**: a developer is "served acceptably" when the median stream meets the scenario's
  `slo.max_ttft` (default ≤ 2 s) and `slo.min_tps` (default ≥ 20 tok/s).
- **Knee**: the highest concurrency whose whole prefix still passes the SLO — the maximum
  concurrent streams the node sustains for this model.

Hardware (GPU/CPU/RAM) is auto-detected into `env.json`, so each run is self-describing.
Comparing the knee across hardware configs identifies the optimal hardware. Translating the
knee into developer counts and cost is decided later, from this output — the harness
measures, it does not hardcode prices.

## Benchmark 2 — SDD lifecycle (`make sdd-run`), agent local

Here only the model needs the GPU, so the **agent, its build-fix loop, and the gates run
locally** in a Docker toolchain container (.NET 10 + Node/npm + Docker + Claude Code +
Python); the pod serves vLLM only, reached over an SSH tunnel with LiteLLM running locally.
`scripts/orchestrate_sdd.py` runs the whole flow as one command:

1. Create a vLLM-only pod (`INSTALL_CLAUDE=0` — no agent/toolchain on the pod).
2. Open `ssh -L 8000:localhost:8000` to vLLM.
3. **Generation** (pod up): in the container, `scripts/run_sdd_scenario.py` drives Claude
   Code phase by phase (spec → acceptance → architecture → plan → backend → frontend →
   .NET Aspire → tests → review), building the Todo App in a shared workspace with a real
   `dotnet`/`npm` build-fix loop (docker-in-docker for Aspire / PostgreSQL / Redis).
4. **Stop the pod** as soon as generation ends (GPU billing stops).
5. **Scoring** (no pod): `scripts/run_gates.py` runs the scenario's deterministic gates
   against the produced workspace.

Generation and scoring are separate modules precisely so the pod can stop between them.

**Scoring (two layers, kept separate):**

- **Hard score (deterministic):** pass/fail gates — does it build, do tests pass, does
  Aspire start, does the frontend build — plus presence/order of the SDD artifacts. Source
  of truth. (The Aspire-runtime gate may fall back to "AppHost builds + manifest generates"
  if docker-in-docker networking blocks full startup.)
- **Soft score (qualitative, planned):** a fixed, pinned judge model scores architecture
  and code quality and SDD adherence with a rubric. Reported separately, non-deterministic.

## Results

Raw per-run output under `results/` is gitignored and collected on the local machine. A
committable reporting format will be defined from that output once we have real runs.

## Reproducibility

The capacity measurements, the toolchain image, and the hard score are deterministic and
runnable from scratch. The soft score is acknowledged as non-deterministic and isolated.
