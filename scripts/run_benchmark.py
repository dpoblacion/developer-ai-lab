"""Single entry for a benchmark run. Provisions the pod and runs the single benchmark runner
(bench_sdd); the shared prologue (provision) and pod lifecycle live in scripts.lib.orchestrator.

Usage: python -m scripts.run_benchmark --benchmark smoke   # or a full path
       [--hardware l40s] [--gpus N] [--key PATH] [--keep]
A bare --benchmark name resolves to benchmarks/<name>/scenario.yaml; a path is used as-is.
The devs list + task are read from the benchmark; the model family (--model) and hardware
(--hardware, +--gpus) are the run axes.

Requires (.env): RUNPOD_API_KEY, SSH_KEY_PATH. SDD scenarios also need local Docker + the
dail-toolchain image (docker build -t dail-toolchain -f infra/toolchain/Dockerfile .).
"""

import argparse
import os
import pathlib
import time

from scripts.lib import bench_sdd
from scripts.lib.compose import attach_weight_cache, load_run_config, model_meta
from scripts.lib.dotenv import load_dotenv
from scripts.lib.orchestrator import provision
from scripts.lib.results_layout import run_out_dir


def resolve_benchmark(value):
    """A bare benchmark name (no '/' and not a .yaml path) resolves to
    benchmarks/<name>/scenario.yaml; anything that looks like a path is used as-is."""
    if "/" in value or value.endswith(".yaml"):
        return value
    return f"benchmarks/{value}/scenario.yaml"


def validate_benchmark(benchmark):
    """Fail fast (before any paid pod) if the benchmark is malformed."""
    if not benchmark.get("task", {}).get("phases"):
        raise SystemExit("benchmark has no task.phases")
    if not benchmark.get("devs"):
        raise SystemExit("benchmark has no devs")


def main():
    load_dotenv()  # local secrets from .env (RUNPOD_API_KEY, SSH_KEY_PATH); env wins
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    help="benchmark name (e.g. dev-load) or path to its scenario.yaml")
    ap.add_argument("--model", required=True, help="model family, e.g. qwen3-coder")
    ap.add_argument("--hardware", default="l40s")
    ap.add_argument("--gpus", type=int, default=1, help="number of GPUs (tensor-parallel size)")
    ap.add_argument("--quant", default=None,
                    help="force the model variant's quant (e.g. awq) instead of the "
                         "hardware's preference order — for same-hardware quant pairs")
    ap.add_argument("--devs", default=None,
                    help="replace the scenario's team-size grid (comma-separated, e.g. "
                         "16,32) — top-up levels run as separate runs")
    ap.add_argument("--key", default=os.environ.get("SSH_KEY_PATH"))
    ap.add_argument("--keep", action="store_true", help="do not terminate the pod at the end")
    args = ap.parse_args()

    bench_path = resolve_benchmark(args.benchmark)  # allow `BENCHMARK=smoke` shorthand
    if not pathlib.Path(bench_path).exists():
        avail = sorted(p.parent.name for p in pathlib.Path("benchmarks").glob("*/scenario.yaml"))
        raise SystemExit(f"benchmark not found: {bench_path}\navailable: {', '.join(avail)}")

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key or not args.key:
        raise SystemExit("Set RUNPOD_API_KEY and SSH_KEY_PATH in .env")
    args.key = os.path.expanduser(args.key)  # ~ is not expanded when passed as an argv item

    import yaml
    import runpod
    runpod.api_key = api_key

    devs_override = None
    if args.devs:
        try:
            devs_override = sorted({int(x) for x in args.devs.split(",") if x.strip()})
        except ValueError:
            raise SystemExit(
                f"--devs must be comma-separated integers, got: {args.devs!r}") from None
        if not devs_override or min(devs_override) < 1:
            raise SystemExit(f"--devs must be positive team sizes, got: {args.devs!r}")
    vllm_cfg, pod_spec, variant, devs, slo = load_run_config(
        bench_path, args.model, args.hardware, gpu_count=args.gpus, quant=args.quant,
        devs=devs_override)
    # Persistent weight cache (see README → GLM): mounted only for models that declare
    # weights_cached — the volume pins the pod to one data center, so it must never
    # constrain runs that don't need it.
    pod_spec = attach_weight_cache(pod_spec, variant, os.environ)
    composed_path = "configs/_composed.yaml"
    pathlib.Path(composed_path).write_text(yaml.safe_dump(vllm_cfg))

    pub = pathlib.Path(args.key + ".pub").read_text().strip()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    bench_name = pathlib.Path(bench_path).parent.name  # bench_path is the resolved path
    out_dir = run_out_dir(variant["family"], variant["quant"], args.hardware,
                          args.gpus, bench_name, run_id).resolve()
    # No upfront mkdir: writers create dirs lazily, so a failure before any work leaves no orphan dir.

    benchmark = yaml.safe_load(pathlib.Path(bench_path).read_text())
    validate_benchmark(benchmark)
    bench_sdd.require_toolchain_image(bench_path)  # fail fast (no pod) if the image isn't built
    hw = yaml.safe_load(pathlib.Path(f"configs/hardware/{args.hardware}.yaml").read_text())
    price = hw["price_usd_per_gpu_hour"]

    with provision(runpod, pod_spec=pod_spec, vllm_cfg=vllm_cfg, key=args.key, pub=pub,
                   scenario_path=bench_path, composed_path=composed_path, out_dir=out_dir,
                   keep=args.keep, hardware=args.hardware, price_usd_per_gpu_hour=price,
                   label=f"bench-{vllm_cfg['served_model_name']}",
                   family=variant["family"], quant=variant["quant"], run_id=run_id,
                   model_meta=model_meta(variant), devs=devs_override) as ctx:
        bench_sdd.run(ctx)


if __name__ == "__main__":
    main()
