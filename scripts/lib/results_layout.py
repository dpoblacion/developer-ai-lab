"""On-disk layout of results/ — pure path logic (dirs in, Paths out), unit-tested offline.

A run lives at results/<family>-<quant>/<hardware>-<gpus>gpu/<benchmark>/<run_id>/, so the model,
hardware, quant and benchmark are in the PATH (navigable) and "latest per config" is just the
lexically-max <run_id> under a <benchmark>/ dir.
"""
import pathlib


def run_out_dir(family, quant, hardware, gpus, benchmark, run_id, root="results"):
    """The on-disk dir for one run:
    <root>/<family>-<quant>/<hardware>-<gpus>gpu/<benchmark>/<run_id>."""
    return (pathlib.Path(root) / f"{family}-{quant}" / f"{hardware}-{gpus}gpu"
            / benchmark / run_id)


def artifacts_to_prune(results_dir="results"):
    """Every run's `artifacts/` dir EXCEPT the latest run per <config>/<benchmark>.
    Latest = lexically-max run_id (timestamps). Never returns a report.json; a missing or
    single-run tree returns []."""
    root = pathlib.Path(results_dir)
    by_bench = {}
    for report in root.glob("**/report.json"):
        run_dir = report.parent
        by_bench.setdefault(run_dir.parent, []).append(run_dir)
    prune = []
    for _bench_dir, runs in by_bench.items():
        latest = max(runs, key=lambda p: p.name)
        for run in runs:
            artifacts = run / "artifacts"
            if run != latest and artifacts.is_dir():
                prune.append(artifacts)
    return prune
