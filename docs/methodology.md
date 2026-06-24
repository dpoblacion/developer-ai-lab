# Methodology

## Purpose — the question this harness answers

This is a **hardware-sizing tool**, not a model-quality benchmark. Given a **model you have
already chosen** and **N developers** using it through a coding agent, it answers:

> What self-hosted hardware do I need to serve those N developers acceptably, and at what
> $/dev-month?

**Model quality** (does the model write good software?) is **out of scope** — assess that
separately, by other means. Here the gates are a **minimum-validity filter**: they confirm each
agent actually did real work, so the performance and token numbers can be trusted (an
all-failing run — e.g. a missing toolchain image — is a garbage measurement, not a "slow
model"). We do **not** rank models by quality.

The workload is a **normal development task** (`dev-load`): N agents concurrently build a small
module, the way real developers would hit the server.

## Principle: the GPU pod runs inference only

The GPU is needed only for **model inference**. Everything else — the agent, builds,
tests, load generation, scoring — runs off the GPU. Each benchmark places work accordingly,
and runs on **ephemeral pods** driven from the local machine via the RunPod API
(create → run → pull results → terminate). Pods are never permanent dev environments.

## Serving and model independence

vLLM serves the model under test via its OpenAI-compatible API. A constant LiteLLM alias
(`dev-model`) is exposed to Claude Code; the upstream served model is set per run via
`LITELLM_UPSTREAM_MODEL`, and where vLLM is reachable via `VLLM_BASE` (localhost on the
pod, `host.docker.internal` from the benchmark container). The model **variant** (quant) is
resolved per hardware: the hardware config declares `supported_quants`, and `compose.py`
picks the matching `configs/models/<family>-<quant>.yaml` automatically — so every benchmark
is identical across Qwen, GLM, MiniMax, etc.:

```
Claude Code (Anthropic API) -> LiteLLM -> destream proxy -> vLLM (OpenAI API) -> model
```

The destream proxy (`scripts/destream_proxy.py`) forces the upstream vLLM call to be
non-streaming — a workaround for vLLM's buggy qwen3 streaming tool-call parser — and
re-emits the reply as streaming chunks, so the agent sees a normal streaming API.

## Benchmark 1 — Capacity (`make run BENCHMARK=dev-load HARDWARE=<hw>`)

`dev-load` is the standardized capacity benchmark. The benchmark declares the task and a
`devs` list (e.g. `[4, 8]`); the model family is the `MODEL=` run parameter. Each N in `devs`
is run **directly**:
N agents concurrently build the same small TypeScript module, and the harness records whether
that N holds the SLO and the $/dev at that team size.

- **SLO**: a developer is "served acceptably" when the median stream meets the scenario's
  `slo.max_ttft` (default ≤ 2 s) and `slo.min_tps` (default ≥ 20 tok/s).
- **holds_slo**: `true` if the SLO is met at N; the dashboard shows which hardware holds at
  each team size, so you can pick the right hardware for your team.

Because the task and SLO are fixed, the `holds_slo` result is directly comparable across
hardware configs. Each run's `report.json` records its config axes (family, quant, hardware,
GPU count, price) so it is self-describing. (`smoke` is the stack pre-flight — `make validate` — a single-agent minimal
run that confirms the serving + agent + gate stack before a full capacity run.)

## Cost — $/dev-month (self-host)

Each hardware × N gives a **$/dev-month**, derived from the hardware config's list price. The
GPU is a **fixed** cost — paid whether 1 or N developers use it — amortized across the
developers it serves concurrently:

```
$/dev-month = ($/GPU-hour × 730 × GPUs) ÷ N        # list price from the hardware config
```

(What **is** measured per run is the pod wall-clock, reported as `pod_cost_usd` — the cost of
running the benchmark itself.)

so it gets **cheaper per developer as N grows** — until the SLO breaks. The dashboard shows the
$/dev-month curve vs N and where each GPU stops holding the SLO, so you can pick the **cheapest
hardware that still serves your team** at the SLO.

## Results

Per-run output is collected on the local machine under `results/`, config-keyed:
`results/<family>-<quant>/<hardware>-<gpus>gpu/<benchmark>/<run_id>/report.json`
(with per-agent artifacts under `artifacts/`), a flat `by_devs` structure (one entry per N with
`holds_slo`, `cost_per_dev_month`, `valid`, `tokens_per_dev`, etc.). The small `report.json`
files are committed (they feed the Streamlit Community Cloud deploy); the heavy `artifacts/`
are gitignored. `make dashboard` is the viewer (benchmark → N → model×hardware);
`make prune-artifacts` drops heavy artifacts of superseded runs. `docs/results.md` explains
the report layout and how to read the dashboard.

## Reproducibility

The capacity measurements, the toolchain image, and the validity gates are deterministic and
runnable from scratch.
