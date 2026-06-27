"""Compose a run config from three reusable axes: model x hardware x benchmark.

compose() is pure (dicts in, dicts out). load_config() resolves names to files under
configs/models, configs/hardware, and a benchmark scenario.yaml, then composes. Output feeds
the existing builders unchanged: vllm_cfg -> build_vllm_args + the pod scripts;
pod_spec -> build_create_kwargs.
"""

import pathlib

CONSTANTS = {"host": "0.0.0.0", "port": 8000, "download_dir": "/workspace/huggingface"}
POD_CONSTANTS = {"ports": "8000/http,22/tcp", "volume_gb": 0, "env": {}}

# Model serving-recipe keys copied verbatim into vllm_cfg when present in the model config.
_MODEL_VLLM_KEYS = ["model", "served_model_name", "vllm_version", "tool_call_parser",
                    "reasoning_parser", "enable_auto_tool_choice", "kv_cache_dtype",
                    "extra_args", "pip_constraints", "env"]


def compose(model, hardware, benchmark):
    """Merge the three axes into (vllm_cfg, pod_spec).

    Precedence (later wins): constants < model < hardware < benchmark.serving.
    Derivations: tensor_parallel_size = hardware.gpu_count; pod image from the model.
    """
    vllm_cfg = dict(CONSTANTS)
    for key in _MODEL_VLLM_KEYS:
        if key in model:
            vllm_cfg[key] = model[key]
    vllm_cfg["gpu_memory_utilization"] = hardware["gpu_memory_utilization"]
    vllm_cfg["tensor_parallel_size"] = hardware["gpu_count"]
    vllm_cfg.update(benchmark.get("serving") or {})

    pod_spec = dict(POD_CONSTANTS)
    pod_spec["name"] = f"dail-{model['served_model_name']}"
    pod_spec["image"] = model["image"]
    pod_spec["container_disk_gb"] = model["container_disk_gb"]
    pod_spec["gpu_type_ids"] = hardware["gpu_type_ids"]
    pod_spec["gpu_count"] = hardware["gpu_count"]
    return vllm_cfg, pod_spec


def _load_yaml(path):
    import yaml
    return yaml.safe_load(pathlib.Path(path).read_text())


def load_config(model_name, hardware_name, scenario_path,
                models_dir="configs/models", hardware_dir="configs/hardware"):
    """Resolve names to config files and compose them. Returns (vllm_cfg, pod_spec)."""
    model = _load_yaml(pathlib.Path(models_dir) / f"{model_name}.yaml")
    hardware = _load_yaml(pathlib.Path(hardware_dir) / f"{hardware_name}.yaml")
    benchmark = _load_yaml(scenario_path)
    return compose(model, hardware, benchmark)
