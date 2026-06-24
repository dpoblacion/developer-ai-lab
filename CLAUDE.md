# Developer AI Lab — guide for Claude Code

A reproducible harness for answering: **given a model family + N developers, which self-hosted
hardware serves them at SLO and at what $/dev**. See `README.md` for the full picture and the
config-field reference.

## ⚠️ Real pods cost money — never launch one without explicit consent

`make run` and the pod orchestration create **real, paid** RunPod GPU pods, billed **per
second**. Do **not** run them to "verify" a change. Verify with:

- `make test` — the unit suite (`.venv/bin/python -m unittest discover -s tests`)
- `python -c "import ast; ast.parse(open('<file>').read())"` for scripts
- `make -n run BENCHMARK=...` to check command expansion

Only run a real `make run` when the user explicitly asks. `make reap` terminates **all**
pods (panic button). A `PodGuard` watchdog + `reap_pods` guard against leaks, but treat any
real run as money spent.

## Commands

- `make test` — unit suite (keep it green).
- `make run BENCHMARK=<name|path> MODEL=<family> HARDWARE=<hw> [GPUS=N KEEP=1]` — run a benchmark **(PAID)**.
  The benchmark carries the `devs` list (N) + task; the **model** (`MODEL=`) and **hardware** are the
  run axes you vary. Each N is tested directly (holds SLO? + $/dev). Defaults `MODEL=qwen3-coder`,
  `BENCHMARK=benchmarks/dev-load/scenario.yaml`, `HARDWARE=l40s`, `GPUS=1`. Example:
  `make run BENCHMARK=dev-load MODEL=glm-5.2 HARDWARE=h200 GPUS=8`.
- `make validate HARDWARE=<hw> [MODEL=<family>]` — stack pre-flight via `smoke` **(PAID, but short)**:
  confirms the serving + agent + gate stack works before a full run.
- `make gates WORKSPACE=… [BENCHMARK=…]` — re-score a produced workspace (no pod; `BENCHMARK` defaults to `dev-load`).
- `make prune-artifacts` — drop `artifacts/` of all but the latest run per config (no pod; keeps every `report.json`).
- `make reap` — terminate all paid pods.
- `make dashboard` — benchmark→N→model×hardware result viewer (Streamlit + Plotly over `results/**/report.json`).

## Architecture

A run is composed from three reusable axes — **benchmark(task+N) × model × hardware** — that
`scripts/lib/compose.py` (pure: dicts in, dicts out) merges into the flat
`configs/_composed.yaml` rsync'd to the pod. Precedence: `constants < model < hardware <
benchmark.serving`; `tensor_parallel_size` derives from `--gpus`; the pod `image` comes from
the model config. The **model** (`MODEL=`) and **hardware** (`HARDWARE=`) are chosen at run time;
`run_benchmark` resolves the variant via `load_run_config(scenario, model_family, hardware, gpus)`.

- A **benchmark** packages the task, not the model: the `devs` list (team sizes to test) + the task
  (phases + gates). The **model family** is the `MODEL=` run parameter; its **variant** (quant, e.g.
  `fp8`) is resolved per hardware via `supported_quants` in the hardware config —
  `configs/models/<family>-<quant>.yaml` (e.g. `qwen3-coder-fp8.yaml` + `qwen3-coder-awq.yaml`, one
  per variant). Each N in `devs` is run directly; there is no concurrency sweep or knee search.
- `configs/models/<family>-<quant>.yaml`, `configs/hardware/<hw>.yaml` — the model and hardware axes
  (field tables in `README.md` → Configuration).
- `benchmarks/<name>/scenario.yaml` (+ `phases/*.md` prompts + optional `Dockerfile`) — a
  self-contained benchmark package; no `type` field — every benchmark uses the same runner.
  Built-in: `dev-load` (capacity benchmark) and `smoke` (stack pre-flight / `make validate`).
- `scripts/run_benchmark.py` — the single entry; calls `bench_sdd.run(ctx)` directly (no
  type dispatch, no handler registry). The shared prologue (create pod → ready → rsync,
  under a `PodGuard`) lives in `scripts/lib/orchestrator.py` (`provision()` + `BenchContext`).
  `bench_sdd` SSHes `setup_pod.sh` + `start_vllm.sh` on the pod, tunnels port 8000, and runs
  the N agents locally in the `dail-toolchain` Docker container.
- Agent chain (local container): `Claude Code → LiteLLM (Anthropic↔OpenAI) → destream proxy
  → ssh tunnel → vLLM on the pod`. SLO comes from the pod's `/metrics` histograms
  (`vllm_metrics.py`), not from client-side stream timing.
- `scripts/lib/` — pure, unit-tested logic (compose, pod_guard, gates, vllm_metrics,
  bench_report, timeline, command builders). SLO scoring lives in `bench_report.py`
  (`dev_record`) — there is no `slo.py`. The pod scripts are `setup_pod.sh` +
  `start_vllm.sh`.

**Adding a model, GPU, or benchmark is a new YAML file — never new code.** Don't change the
pod-script or builder (`build_vllm_args`/`build_create_kwargs`) contracts without reason, and
keep `compose()` pure.

## Gotchas / load-bearing workarounds

- **destream proxy** (`scripts/destream_proxy.py`): vLLM's qwen3 *streaming* tool-call parser
  is buggy (tool calls leak into content), so agent traffic is forced non-streaming upstream
  and re-emitted as SSE. Don't remove it from the chain; `scripts/probe_vllm_tools.py`
  (`SDD_VLLM_PROBE=1`) diagnoses it.
- **Never use `claude --bare`**: it hard-forces `[Bash, Edit, Read]` (no `Write`), which
  breaks file-creation phases. `scripts/lib/claude.py` builds the invocation deliberately.
- **`pod_cost_usd` is wall-clock × list price** from the hardware config — the run's real
  duration, but not a RunPod billing read. `cost_per_dev_month` is a pure price projection.
- **`results/**/report.json` is committed** (whitelisted in `.gitignore`; it feeds the
  Streamlit Community Cloud deploy). Everything else under `results/` stays untracked —
  don't "clean up" the reports.
- `make setup` / `make sdd` are manual-debug entry points, **not** on the `make run` path.

## Conventions

- Tests are `unittest.TestCase` under `tests/`; follow TDD and keep the suite green.
- `docs/` is **user** documentation (`goals`, `methodology`, `results`) — keep it in sync
  when behavior changes; `docs/results.md` explains the results layout + dashboard (process,
  NOT benchmark numbers — the maintainer keeps findings out of it).
- The user decides what gets committed; don't commit or push unless asked.
