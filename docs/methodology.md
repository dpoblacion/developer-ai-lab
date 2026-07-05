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
  `slo.max_ttft` (default ≤ 10 s — the first-token waiter is an agent loop, which
  tolerates queueing a human chat would not) and `slo.min_tps` (default ≥ 20 tok/s).
- **holds_slo**: `true` if the SLO is met at N; the dashboard shows which hardware holds at
  each team size, so you can pick the right hardware for your team.
- **holds_slo_p90 / p90_ttft / p90_tps**: the same thresholds evaluated at the p90 tail
  (from the same server-side histograms). Medians alone let a large minority of requests
  violate the SLO unseen; tail-conditioning follows the concurrency-aware costing
  methodology of [arXiv:2606.11690](https://arxiv.org/abs/2606.11690). `holds_slo` (median)
  remains the headline gate; the p90 fields expose what it hides.

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

### Effective $/MTok and utilization (concurrency-aware costing)

Following [arXiv:2606.11690](https://arxiv.org/abs/2606.11690) ("Beyond Per-Token Pricing"),
each N also reports the **measured** per-token economics of the level — utilization is an
*output* of the measurement, never an assumed input:

- **server_tokens / tokens_per_hour**: prompt + generation tokens the level actually served,
  over the level's wall duration. Counts come from vLLM's own counters diffed per level
  (`tokens_source: "server"`); when server counters are unavailable, the agents'
  client-reported usage — what the server processed — stands in, over the slowest agent's
  wall time (`tokens_source: "client"`).
- **cost_per_mtok**: `cost_per_hour ÷ tokens_per_hour × 10⁶` — the effective $/MTok at that
  N, blended over all served tokens.
- **cost_per_mtok_output**: the same cost assigned entirely to generated tokens — the figure
  to hand-compare against per-token API pricing (which bills output 5-6× input; agentic
  coding traffic is heavily prompt-dominated, so the blended and output figures differ a lot).
- **theta_max (raw saturation)**: after the last level, while the pod is still up, a
  saturation probe (`saturation_probe:` in the scenario) drives vLLM with fresh
  random-word prompts — no shareable prefixes, so the cache cannot inflate the ceiling —
  at the workload's own prompt:output ratio, and measures Θmax from the server's counters.
  This is the engine+hardware ceiling of arXiv:2606.11690 §3.2, distinct from *goodput*:
  the harness's `holds_slo` sweep IS the goodput measurement (the largest N your SLO
  permits), and Θmax is the no-SLO ceiling it is judged against.
- **utilization / underutilization_penalty**: U = Θachieved/Θmax and its inverse — what
  you pay at low N for headroom you are not using. Against the probed Θmax when the run
  carried one (`utilization_basis: theta_max`); otherwise against the best measured level
  (`best_level`, a lower bound, requiring ≥2 levels — a single level is trivially its own
  best). This is the paper's core point transplanted to team sizing: the same GPU's
  effective $/MTok can differ by an order of magnitude between N=1 and the largest N that
  still holds SLO — which is why the `devs` grid starts at N=1, the idle edge where the
  penalty peaks (the paper's headline is 17.5–36.3× there).
- **The SLO's price** (paper Table 4): per combo, the largest measured N that holds the
  SLO, the output-$/MTok at that shippable operating point, and its premium over the
  saturation floor `Csat = cost_per_hour ÷ Θmax_output` — a floor that is unreachable
  under any real latency commitment.
- **Repeats and stability** (paper §5.8): re-measuring the same combo × N is a repeat —
  the dashboard aggregates repeats as mean ± CV (the SLO verdict is conservative: every
  repeat must hold). Run the same `make run` again to add a repeat.
- **prefix_cache_hit_rate**: agentic traffic shares real prefixes (system prompts, tool
  loops), so vLLM's prefix cache is live in these measurements; the per-level hit rate
  bounds how they compare to cache-free protocols (the paper measures real hits cutting
  saturation cost by 20–22%).
- **slo.percentile**: the SLO gate is judged at the scenario's percentile (`p50`
  default; `p90`/`p99` available — the paper's example SLA is at p99). Tail fields
  (p90/p99 TTFT and tok/s, E2E p50/p99) are always reported.
- **Quantization impact**: the dashboard compares quants only between runs of the same
  family on identical hardware and GPU count, at common team sizes — any other pairing
  confounds the quant with the GPU. (The paper finds the FP8 gain is
  architecture-dependent: ~+31% dense vs +69-74% MoE.)

Where the paper sweeps synthetic Poisson request rates (fixed 512:256 token shapes), this
harness drives **real agentic coding sessions** and adds quality gates on the produced code —
the two are complementary: their λ-sweep isolates the cost curve; our N-sweep prices the
workload you actually run.

## Results

Per-run output is collected on the local machine under `results/`, config-keyed:
`results/<benchmark>/<family>-<quant>/<hardware>-<gpus>gpu/<run_id>/report.json`
(with per-agent artifacts under `artifacts/`), a flat `by_devs` structure (one entry per N with
`holds_slo`, `cost_per_dev_month`, `valid`, `tokens_per_dev`, etc.). The small `report.json`
files are committed (they feed the Streamlit Community Cloud deploy); the heavy `artifacts/`
are gitignored. `make dashboard` is the viewer (benchmark → N → model×hardware);
`make prune-artifacts` drops heavy artifacts of superseded runs. `docs/results.md` explains
the report layout and how to read the dashboard.

## Reproducibility

The capacity measurements, the toolchain image, and the validity gates are deterministic and
runnable from scratch.
