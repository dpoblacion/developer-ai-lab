"""Compose a run config from three reusable axes: model x hardware x benchmark.

compose() is pure (dicts in, dicts out). load_run_config() resolves the model variant for a
hardware (via supported_quants) from files under configs/models, configs/hardware, and a
benchmark scenario.yaml, then composes. Output feeds the existing builders unchanged:
vllm_cfg -> build_vllm_args + the pod scripts; pod_spec -> build_create_kwargs.
"""

import pathlib

CONSTANTS = {"host": "0.0.0.0", "port": 8000, "download_dir": "/workspace/huggingface"}
# Only SSH is exposed. vLLM is reached over the SSH tunnel, so it is never published on
# RunPod's public proxy — exposing 8000/http would serve the model unauthenticated to
# anyone with the pod URL (and let them skew the benchmark's latency numbers via contention).
POD_CONSTANTS = {"ports": "22/tcp", "volume_gb": 0, "env": {}}

# Fallback when a benchmark declares no slo: — the single source of truth (bench_sdd
# imports it; keep them from drifting apart).
DEFAULT_SLO = {"max_ttft": 10.0, "min_tps": 20.0}

# Model serving-recipe keys copied verbatim into vllm_cfg when present in the model config.
_MODEL_VLLM_KEYS = ["model", "served_model_name", "vllm_version", "tool_call_parser",
                    "reasoning_parser", "enable_auto_tool_choice", "kv_cache_dtype",
                    "extra_args", "pip_constraints", "pip_extra_index_url", "env"]


def compose(model, hardware, benchmark, gpu_count=1):
    """Merge the three axes into (vllm_cfg, pod_spec).

    Precedence (later wins): constants < model < hardware < benchmark.serving.
    Derivations: tensor_parallel_size = gpu_count (the run's --gpus); pod image from the model.
    """
    if gpu_count < 1:
        raise ValueError(f"gpu_count must be >= 1, got {gpu_count}")
    vllm_cfg = dict(CONSTANTS)
    for key in _MODEL_VLLM_KEYS:
        if key in model:
            vllm_cfg[key] = model[key]
    vllm_cfg["gpu_memory_utilization"] = hardware["gpu_memory_utilization"]
    vllm_cfg["tensor_parallel_size"] = gpu_count
    serving = benchmark.get("serving") or {}
    vllm_cfg.update(serving)
    # vLLM must admit all N agents at once; a smaller cap would queue them and inflate TTFT
    # before the real memory limit. Derive from the benchmark's devs list.
    devs = benchmark.get("devs") or [1]
    vllm_cfg["max_num_seqs"] = max(max(devs), serving.get("max_num_seqs", 1))
    # The benchmark REQUESTS a context length; the card AND the model variant CONSTRAIN
    # it (minimum wins). Small-VRAM GPUs cannot hold the full request next to a big
    # model's weights (vLLM refuses to boot if one max-length sequence doesn't fit in the
    # KV cache — live 2026-07-05 on 24GB), and a fat variant (e.g. GPTQ gs32's extra
    # quant metadata) tightens the fit further than its siblings.
    caps = [c for c in (hardware.get("max_model_len_cap"),
                        model.get("max_model_len_cap")) if c]
    if caps and vllm_cfg.get("max_model_len", min(caps)) > min(caps):
        vllm_cfg["max_model_len"] = min(caps)

    pod_spec = dict(POD_CONSTANTS)
    pod_spec["env"] = {k: v for k, v in (
        ("HW_PROVIDER", hardware.get("provider")),
        ("HW_INSTANCE", hardware.get("instance")),
    ) if v}
    pod_spec["name"] = f"dail-{model['served_model_name']}"
    if "image" not in model:
        raise ValueError(
            f"model '{model.get('served_model_name', '?')}' has no 'image' — set it in "
            "its configs/models/<model>.yaml before provisioning a pod.")
    pod_spec["image"] = model["image"]
    pod_spec["container_disk_gb"] = model["container_disk_gb"]
    pod_spec["gpu_type_ids"] = hardware["gpu_type_ids"]
    pod_spec["gpu_count"] = gpu_count
    return vllm_cfg, pod_spec


def resolve_variant(family, hardware, variants, quant=None):
    """Pick the model variant for a hardware: the first of the hardware's preference-ordered
    supported_quants that some variant of `family` provides. An explicit `quant` (QUANT=)
    overrides the preference order — it must still be supported by the hardware and exist
    for the family, so a same-hardware quantization pair can be measured deliberately but
    an unservable combination never launches a pod. SystemExit if none match."""
    by_quant = {v["quant"]: v for v in variants if v.get("family") == family}
    if not by_quant:
        raise SystemExit(f"no model variant for family '{family}' (have families: "
                         f"{sorted({v.get('family') for v in variants})})")
    supported = hardware.get("supported_quants", [])
    if quant is not None:
        if quant not in supported:
            raise SystemExit(f"quant '{quant}' is not supported by this hardware "
                             f"(supports {supported})")
        if quant not in by_quant:
            raise SystemExit(f"no '{family}' variant for quant '{quant}' "
                             f"(family has {sorted(by_quant)})")
        return by_quant[quant]
    for q in supported:
        if q in by_quant:
            return by_quant[q]
    raise SystemExit(f"no '{family}' variant for this hardware (supports "
                     f"{supported}; family has {sorted(by_quant)})")


def attach_weight_cache(pod_spec, variant, env):
    """Mount the persistent weight-cache volume ONLY for models that declare their
    weights live on it (`weights_cached: true` in the model config) and only when the
    env provides one. The volume pins the pod to its data center, so attaching it
    globally starved unrelated runs of GPUs (live 2026-07-04). Pure: returns a new dict."""
    spec = dict(pod_spec)
    if variant.get("weights_cached") and env.get("DAIL_NETWORK_VOLUME_ID"):
        spec["network_volume_id"] = env["DAIL_NETWORK_VOLUME_ID"]
        spec["data_center_id"] = env.get("DAIL_DATA_CENTER_ID")
    return spec


def model_meta(variant):
    """Architecture metadata a model config may declare (dense/MoE, total/active params)
    — flows into report.json to enable active-parameters analyses (arXiv:2606.11690:
    active parameter count, not total size, appears to drive saturation economics)."""
    return {k: variant[k] for k in ("architecture", "total_params_b", "active_params_b")
            if k in variant}


def load_variants(models_dir="configs/models"):
    """Every model-variant config under models_dir."""
    return [_load_yaml(p) for p in sorted(pathlib.Path(models_dir).glob("*.yaml"))]


def _load_yaml(path):
    import yaml
    return yaml.safe_load(pathlib.Path(path).read_text())


def load_run_config(scenario_path, model_family, hardware_name, gpu_count=1,
                    models_dir="configs/models", hardware_dir="configs/hardware",
                    quant=None, devs=None):
    """Resolve the model variant of `model_family` for the hardware and compose the benchmark
    (devs + slo + serving). `model_family` is the run's model choice (a --model flag, no longer
    read from the scenario); `quant` optionally forces the variant (--quant / QUANT=);
    `devs` optionally replaces the scenario's team-size grid (--devs / DEVS=) — how top-up
    levels (e.g. 16/32) run as separate runs the dashboard merges per combo×N.
    Returns (vllm_cfg, pod_spec, variant, devs, slo)."""
    benchmark = _load_yaml(scenario_path)
    if devs:
        benchmark["devs"] = sorted(devs)   # before compose: max_num_seqs derives from it
    hardware = _load_yaml(pathlib.Path(hardware_dir) / f"{hardware_name}.yaml")
    variant = resolve_variant(model_family, hardware, load_variants(models_dir), quant=quant)
    vllm_cfg, pod_spec = compose(variant, hardware, benchmark, gpu_count=gpu_count)
    devs = benchmark.get("devs") or [1]
    slo = benchmark.get("slo") or DEFAULT_SLO
    return vllm_cfg, pod_spec, variant, devs, slo
