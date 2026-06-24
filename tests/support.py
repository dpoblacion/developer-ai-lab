"""Shared test fixtures (not a test module — discovery only picks up test*.py)."""
import json
import os
import pathlib
import tempfile

from scripts.lib import pod_guard


def make_guard(killed, state_path=None, clock=lambda: 0.0, **kw):
    """A PodGuard wired for tests: terminations recorded in `killed`, state in a temp file
    unless the test needs to inspect it (pass state_path). Extra kwargs (is_alive,
    max_run, list_pods_fn, ...) pass through."""
    if state_path is None:
        state_path = os.path.join(tempfile.mkdtemp(), "active-pods.json")
    return pod_guard.PodGuard("test", terminate_fn=killed.append,
                              state_path=state_path, clock=clock, **kw)


def mk_run(root, run_id, config="qwen3-coder-fp8/h200-1gpu", bench="dev-load",
           with_artifacts=True):
    """One config-keyed run dir (<root>/<config>/<bench>/<run_id>/report.json [+ artifacts/])
    — the layout results_layout.artifacts_to_prune and prune_artifacts operate on."""
    run = pathlib.Path(root) / config / bench / run_id
    run.mkdir(parents=True)
    (run / "report.json").write_text(json.dumps({"run_id": run_id}))
    if with_artifacts:
        agent = run / "artifacts" / "n1" / "agent0"
        agent.mkdir(parents=True)
        (agent / "blob.txt").write_text("x")
    return run
