"""Run Claude Code headless against the model under test and capture metrics.

Drives ``claude -p --output-format stream-json`` (which reaches the model via the
LiteLLM proxy configured in config.env), streams the transcript to disk, and writes
parsed metrics next to it under ``results/<timestamp>/<label>/``.
"""

import json
import os
import pathlib
import sys
import time

from scripts.lib.claude import invoke_claude
from scripts.lib.transcript import parse_metrics

MODEL = os.getenv("ANTHROPIC_MODEL", "dev-model")


def build_metrics(lines, wall_time, label, model):
    """Combine parsed transcript metrics with run metadata."""
    metrics = parse_metrics(lines)
    metrics.update({"wall_time": wall_time, "label": label, "model": model})
    return metrics


def run(prompt, label, allowed_tools="Bash,Read,Edit,Write", max_turns=30):
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = pathlib.Path("results") / run_id / label
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript, wall_time = invoke_claude(prompt, allowed_tools, max_turns)

    (out_dir / "transcript.jsonl").write_text(transcript)

    metrics = build_metrics(transcript.splitlines(), wall_time, label, MODEL)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run(prompt=sys.argv[1], label=sys.argv[2] if len(sys.argv) > 2 else "run")
