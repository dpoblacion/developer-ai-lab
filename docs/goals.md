# Goals

Developer AI Lab is a reproducible **hardware-sizing tool**. Given a **model family you have
already chosen** and **N developers** using it through a coding agent, it answers which
**self-hosted hardware** serves those N at an acceptable SLO and at what **$/dev**. It is **not** a
model-quality benchmark.

## Questions we want to answer

- For a model family + N developers, which hardware holds the SLO — and which is cheapest per developer?
- How many concurrent developers does a given hardware config serve before it stops holding the SLO?
- How does the $/dev-month change with team size, and where is each GPU's ceiling?

## What we measure

A single capacity benchmark, `dev-load`: **N agents concurrently** build a small module the
way real developers would hit the server. Each team size in the benchmark's `devs` list is
run directly (no knee search); for each N we record whether the median stream **holds the
SLO** (TTFT + tokens/s) and the resulting **$/dev-month** on that hardware.

**Model quality** (does the model write good software?) is **out of scope** — assess that
separately. Here the deterministic gates are a **minimum-validity filter**: they confirm each
agent actually did real work, so the performance and token numbers can be trusted. We do
**not** rank models by quality.

## Principles

- **GPU pod runs inference only** — the agent, builds, tests, load generation, and scoring
  run off the GPU; the pod serves the model and nothing else.
- **Reproducible and deterministic** for the objective measures.
- **Runnable from scratch** on an ephemeral pod.
- **Model-independent** — the same harness works for any model; adding one is a config
  change, never new code.
- **Easy to extend** with new models, hardware, and scenarios.

The goal is not a collection of scripts, but a professional lab for sizing and costing the
hardware that serves a development team a model of their choice.
