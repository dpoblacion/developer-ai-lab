"""Shared helper to invoke Claude Code headless.

``build_claude_cmd`` is pure (testable offline); ``invoke_claude`` runs it and returns
the transcript and wall time. Used by both the single-run wrapper and the SDD
orchestrator so the invocation lives in one place.
"""

import subprocess
import time


def build_claude_cmd(prompt, allowed_tools, max_turns, permission_mode="acceptEdits"):
    """Build the ``claude -p`` stream-json command line.

    ``--tools`` sets the AVAILABLE built-in toolset to exactly the phase's tools, so the
    model actually gets ``Write`` (not just Bash/Edit/Read) and can't reach tools the phase
    didn't grant. ``--allowedTools`` then auto-approves them so headless ``-p`` never blocks
    on a permission prompt.

    We deliberately do NOT use ``--bare``: it hard-forces the toolset to ``[Bash, Edit,
    Read]`` (no ``Write``), which made file-creation phases fail. Its other benefits are
    preserved another way — the small system prompt comes from restricting ``--tools``, and
    isolation comes from the clean toolchain container (no ~/.claude, CLAUDE.md, or plugins).
    """
    return [
        "claude", "-p", prompt,
        "--tools", allowed_tools.replace(",", " "),
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
