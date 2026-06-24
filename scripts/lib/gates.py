"""Run the deterministic hard-score gates of a scenario.

A gate is a shell command run at a working directory; it passes when its exit code
equals ``expect_exit`` (default 0). This is the objective, reproducible core of the
SDD score.
"""

import subprocess


def run_gates(gates, cwd="."):
    """Run each gate and return a summary dict.

    Returns: {"total": int, "passed": int, "results": [
        {"id": str, "passed": bool, "exit_code": int}, ...]}
    """
    results = []
    for gate in gates:
        proc = subprocess.run(
            gate["cmd"], shell=True, cwd=cwd,
            capture_output=True, text=True,
        )
        passed = proc.returncode == gate.get("expect_exit", 0)
        results.append({
            "id": gate["id"],
            "passed": passed,
            "exit_code": proc.returncode,
        })
    return {
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "results": results,
    }
