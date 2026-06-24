"""Score a produced SDD workspace with the scenario's deterministic gates.

Runs with the pod DOWN (no model needed). Usage:
  python -m scripts.run_gates benchmarks/dev-load/scenario.yaml <workspace_dir>
"""

import json
import os
import pathlib
import sys

from scripts.lib.gates import run_gates


def load_gates(scenario_path):
    import yaml  # deferred: only needed when actually scoring
    return yaml.safe_load(pathlib.Path(scenario_path).read_text()).get("task", {}).get("gates", [])


def build_hard_score(run_id, model, scenario_name, gate_summary):
    """Assemble the hard-score document (deterministic output schema)."""
    return {
        "run_id": run_id,
        "model": model,
        "scenario": scenario_name,
        "gates": gate_summary,
    }


def main(scenario_path, workspace):
    import yaml
    scenario = yaml.safe_load(pathlib.Path(scenario_path).read_text())
    gate_summary = run_gates(scenario.get("task", {}).get("gates", []), cwd=workspace)

    ws = pathlib.Path(workspace)
    run_id = ws.parent.parent.name  # results/<run_id>/sdd/workspace
    hard = build_hard_score(run_id, os.getenv("MODEL", "unknown"),
                            scenario.get("name", "?"), gate_summary)
    out = ws.parent / "hard-score.json"
    out.write_text(json.dumps(hard, indent=2))
    print(f"{gate_summary['passed']}/{gate_summary['total']} gates passed -> {out}")
    sys.exit(0 if gate_summary["passed"] == gate_summary["total"] else 1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
