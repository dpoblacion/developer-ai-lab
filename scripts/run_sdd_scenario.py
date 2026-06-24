"""Run an SDD scenario: drive the model through each phase with Claude Code, then
score the produced workspace with the deterministic hard-score gates.

Phases run in a single clean workspace (later phases read earlier artifacts). Per-phase
metrics and the gate results are written under ``results/<timestamp>/sdd/``.

Requires Claude Code, the LiteLLM proxy + vLLM, and the build toolchain (dotnet, npm) for
the gates — i.e. a pod, not local.
"""

import json
import os
import pathlib
import sys
import time

from scripts.lib.claude import invoke_claude
from scripts.lib.gates import run_gates
from scripts.lib.transcript import parse_metrics

MODEL = os.getenv("ANTHROPIC_MODEL", "dev-model")


def load_scenario(path):
    import yaml  # deferred: only needed when actually running a scenario (on a pod)
    return yaml.safe_load(pathlib.Path(path).read_text())


def build_hard_score(run_id, model, scenario_name, gate_summary, phases):
    """Assemble the hard-score document (deterministic output schema)."""
    return {
        "run_id": run_id,
        "model": model,
        "scenario": scenario_name,
        "gates": gate_summary,
        "phases": phases,
    }


def run(scenario_path):
    scenario = load_scenario(scenario_path)
    scenario_dir = pathlib.Path(scenario_path).parent

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = pathlib.Path("results") / run_id / "sdd"
    workspace = out_dir / "workspace"
    phases_dir = out_dir / "phases"
    workspace.mkdir(parents=True, exist_ok=True)
    phases_dir.mkdir(parents=True, exist_ok=True)

    phase_metrics = []
    for phase in scenario["phases"]:
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

    gate_summary = run_gates(scenario.get("gates", []), cwd=str(workspace))
    hard_score = build_hard_score(
        run_id, MODEL, scenario["name"], gate_summary, phase_metrics)
    (out_dir / "hard-score.json").write_text(json.dumps(hard_score, indent=2))

    print(json.dumps({
        "scenario": scenario["name"],
        "gates_passed": f"{gate_summary['passed']}/{gate_summary['total']}",
        "results": str(out_dir),
    }, indent=2))
    return hard_score


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/scenarios/todo-app/scenario.yaml"
    run(path)
