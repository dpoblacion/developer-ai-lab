"""Capture the hardware environment of a benchmark run.

Pure parsers (``detect_gpus``, ``parse_mem_total_gb``, ``collect_environment``) are
unit-testable offline. ``gather`` does the actual side-effecting detection (nvidia-smi,
/proc, env) and is meant to run on the pod. No pricing/cost here — the harness measures
hardware; cost reporting is decided later from the captured output.
"""

import json
import os
import pathlib
import subprocess


def detect_gpus(smi_csv):
    """Parse ``nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits``."""
    gpus = []
    for line in smi_csv.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        name, mem = [p.strip() for p in line.split(",")]
        gpus.append({"name": name, "memory_mib": int(mem)})
    return gpus


def parse_mem_total_gb(meminfo):
    """Parse total RAM in GB from the contents of /proc/meminfo."""
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            kb = int(line.split()[1])
            return round(kb / 1024 / 1024, 1)
    return None


def collect_environment(gpus, ram_gb, cpu_count, env):
    """Assemble the run's hardware document (metadata only, no cost)."""
    return {
        "provider": env.get("HW_PROVIDER"),
        "instance": env.get("HW_INSTANCE"),
        "gpus": gpus,
        "gpu_count": len(gpus),
        "cpu_count": cpu_count,
        "ram_gb": ram_gb,
    }


def gather():
    """Detect the real environment (runs nvidia-smi, reads /proc, env)."""
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
        gpus = detect_gpus(smi)
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpus = []

    meminfo = pathlib.Path("/proc/meminfo")
    ram_gb = parse_mem_total_gb(meminfo.read_text()) if meminfo.exists() else None

    return collect_environment(gpus, ram_gb, os.cpu_count(), os.environ)


def write_env(out_dir, environment):
    """Write env.json into the run's result directory."""
    path = pathlib.Path(out_dir) / "env.json"
    path.write_text(json.dumps(environment, indent=2))
    return path
