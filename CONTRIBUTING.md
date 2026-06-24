# Contributing

Thanks for your interest! This project has a small, deliberate design — a few rules keep it
that way.

## Ground rules

- **Adding a model, a GPU, or a benchmark is a new YAML file — never new code.**
  Models live in `configs/models/<family>-<quant>.yaml`, hardware in
  `configs/hardware/<hw>.yaml`, benchmarks in `benchmarks/<name>/`. If you find yourself
  editing Python to add one, something is wrong — open an issue instead.
- **Never launch a real pod to verify a change.** `make run` / `make validate` create
  **paid** GPU pods billed per second. Verify with the free tools:
  - `make test` — the unit suite (offline, no Docker, no GPU)
  - `ruff check .` — lint
  - `make -n run BENCHMARK=...` — check command expansion
- **Tests are `unittest.TestCase` under `tests/`.** Please follow TDD: a failing test
  first, then the change. Keep the suite green and fast (it runs in seconds, offline).
- `scripts/lib/` is pure, unit-tested logic; the orchestration entry points stay thin.
  Keep `compose()` pure (dicts in, dicts out) and don't change the pod-script or builder
  contracts (`build_vllm_args` / `build_create_kwargs`) without a strong reason.

## Dev setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-orchestrator.txt
make test
```

## Pull requests

- One logical change per PR, with tests.
- CI must pass (`make test` + `ruff check .`).
- If your change alters behavior described in `README.md` or `docs/`, update those too —
  stale docs are treated as bugs here.
