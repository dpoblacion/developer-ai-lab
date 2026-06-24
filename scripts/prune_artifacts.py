"""Delete non-latest run artifacts/ dirs (keeps every report.json). Run: make prune-artifacts."""
import shutil
from scripts.lib.results_layout import artifacts_to_prune


def main(results_dir="results"):
    targets = artifacts_to_prune(results_dir)
    for t in targets:
        print(f"removing {t}")
        shutil.rmtree(t)
    print(f"pruned {len(targets)} artifacts dir(s)")


if __name__ == "__main__":
    main()
