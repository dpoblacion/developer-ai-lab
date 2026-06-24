# Results — where they live and how to read them

This page explains what a benchmark run produces, where it lands on disk, and how to
explore it in the dashboard. (For *what* is measured and why, see `methodology.md`.)

## What a run produces

Every `make run` ends by writing one **`report.json`** — the complete, self-describing
summary of the run — plus heavy per-agent **artifacts**. Results are **config-keyed**: the
path itself tells you the model, quantization, hardware, GPU count, and benchmark:

```
results/<family>-<quant>/<hardware>-<gpus>gpu/<benchmark>/<run_id>/
  report.json               # the run summary (small, kept forever)
  artifacts/n<N>/agent<i>/  # per-agent workspace, transcripts, gate output (heavy)
```

The latest run for a config+benchmark is simply the lexically-max `<run_id>`
(timestamps like `20260702-135850`).

### Anatomy of `report.json`

- **Config axes** — `family`, `quant`, `model`, `hardware`, `gpu_count`,
  `price_usd_per_gpu_hour`: everything needed to reproduce the run.
- **`by_devs`** — one entry per team size N tested, the heart of the report:
  - `holds_slo` — did the median stream meet the SLO (TTFT + tok/s) at this N?
  - `cost_per_dev_month` — the GPU's list price amortized across N developers.
  - `median_tps` / `median_ttft` — the measured per-stream latency at this N.
  - `valid` — did the agents do real work? (An all-failing run is a garbage
    measurement, not a slow model.)
  - `agents_ok` — how many agents passed the validity gates; per-agent detail (tokens,
    wall time, gate outcomes, `gate_timeout` for environmental timeouts) nests under
    `agents`.
  - `tokens_per_dev` — median tokens one developer's task consumed.
- **`timeline`** + **`pod_cost_usd`** — the billed pod time broken into steps
  (create→ready, rsync, vLLM startup, generation) and its total cost: measured
  wall-clock × the hardware's list price. Gate scoring runs with the pod already
  terminated, so it never appears here.

### What is committed vs. local

The small `report.json` files are **committed** (they feed the Streamlit Community Cloud
deploy of the dashboard); `artifacts/` stay **gitignored**. `make prune-artifacts` deletes
the artifacts of every run except the latest per config — reports are never touched.

## The dashboard

```bash
make dashboard      # Streamlit + Plotly over results/**/report.json — free, no pod
```

A hosted instance over this repo's committed reports runs at
[developer-ai-lab.streamlit.app](https://developer-ai-lab.streamlit.app/).

<!-- TODO(demo): dashboard screenshot -->

Navigation follows the question the lab answers — *benchmark → team size → which
model×hardware serves it cheapest*:

1. **Pick a benchmark** (the `smoke` pre-flight is hidden).
2. **Pick a team size N.** The dashboard ranks every model×hardware combo measured at
   that N by **$/dev-month, cheapest first** — combos that don't hold the SLO are marked,
   because a cheap GPU that misses the SLO is not an option, just cheap.
3. **The $/dev vs N curve** shows how each combo's cost falls as the fixed GPU is
   amortized across more developers — and where each one stops holding the SLO. The
   cheapest hardware *that still holds the SLO at your team size* wins.
4. **Per-run detail**: the billed-time timeline (where the pod money went), token totals,
   SLO headroom, and each agent's gate outcomes.

The same `report.json` files render a compact terminal summary at the end of every run
(`scripts/lib/report_terminal.py`), so you get the holds-SLO/$-per-dev table without
opening the dashboard.

## Re-scoring without a pod

Gates are deterministic shell commands over the agent workspaces, so you can re-score any
kept artifact later, free:

```bash
make gates WORKSPACE=results/<...>/artifacts/n<N>/agent<i>/workspace
```
