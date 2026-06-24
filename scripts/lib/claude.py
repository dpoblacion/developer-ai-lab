"""Shared helper to invoke Claude Code headless.

``build_claude_cmd`` is pure (testable offline); ``invoke_claude`` runs it and returns
the transcript and wall time. Used by both the single-run wrapper and the SDD
orchestrator so the invocation lives in one place.
"""

import subprocess
import time


def build_claude_cmd(prompt, allowed_tools, max_turns, permission_mode="acceptEdits"):
    """Build the ``claude -p`` stream-json command line."""
    return [
        "claude", "-p", prompt,
        "--output-format", "stream-json", "--verbose",
        "--allowedTools", allowed_tools,
        "--permission-mode", permission_mode,
        "--max-turns", str(max_turns),
    ]


def invoke_claude(prompt, allowed_tools, max_turns, cwd=None):
    """Run Claude Code headless. Returns (stdout_transcript, wall_time_seconds)."""
    cmd = build_claude_cmd(prompt, allowed_tools, max_turns)
    start = time.time()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.stdout, time.time() - start
