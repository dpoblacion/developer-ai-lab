"""Generation phase of an SDD scenario: drive the model through each phase with Claude
Code, producing the app in a clean workspace and capturing per-phase metrics.

Scoring is separate (scripts/run_gates.py) so the GPU pod can stop before gates run.
Phases run in a single workspace (later phases read earlier artifacts); output goes to
``results/<timestamp>/sdd/``.
"""

import json
import os
import pathlib
import sys
import time

from scripts.lib.claude import invoke_claude
from scripts.lib.transcript import parse_metrics

MODEL = os.getenv("ANTHROPIC_MODEL", "dev-model")


def load_scenario(path):
    import yaml  # deferred: only needed when actually running a scenario (on a pod)
    return yaml.safe_load(pathlib.Path(path).read_text())


def run(scenario_path):
    scenario = load_scenario(scenario_path)
    scenario_dir = pathlib.Path(scenario_path).parent

    run_id = time.strftime("%Y%m%d-%H%M%S")
    # SDD_OUT_DIR lets the orchestrator point output at a mounted dir inside the container.
    out_dir = pathlib.Path(os.getenv("SDD_OUT_DIR") or pathlib.Path("results") / run_id / "sdd")
    workspace = out_dir / "workspace"
    phases_dir = out_dir / "phases"
    workspace.mkdir(parents=True, exist_ok=True)
    phases_dir.mkdir(parents=True, exist_ok=True)

    phase_metrics = []
    for phase in scenario["task"]["phases"]:
        prompt = (scenario_dir / phase["prompt"]).read_text()
        allowed_tools = ",".join(phase.get("allowed_tools", []))
        max_turns = phase.get("max_turns", 30)

        print(f"Phase {phase['id']}: {phase['title']}")
        transcript, wall_time = invoke_claude(
            prompt, allowed_tools, max_turns, cwd=str(workspace))

        (phases_dir / f"{phase['id']}.jsonl").write_text(transcript)
        metrics = parse_metrics(transcript.splitlines())
        metrics.update({"phase": phase["id"], "wall_time": wall_time})
        phase_metrics.append(metrics)

    result = {
        "run_id": run_id,
        "model": MODEL,
        "scenario": scenario["name"],
        "workspace": str(workspace),
        "phases": phase_metrics,
    }
    (out_dir / "generation.json").write_text(json.dumps(result, indent=2))
    print(f"Generation done. Workspace: {workspace}")
    return result


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/dev-load/scenario.yaml"
    run(path)
