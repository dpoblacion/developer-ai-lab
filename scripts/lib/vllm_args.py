"""Build the ``vllm serve`` argument list from a model config dict.

Config-driven and model-independent: common fields map to flags; per-model special
flags go in ``extra_args`` (appended verbatim). Pure ``build_vllm_args`` is unit-tested;
the ``__main__`` entry loads the YAML and prints args one per line for start_vllm.sh.
"""

import sys


def build_vllm_args(cfg):
    """Return the positional model + flags for ``vllm serve``."""
    args = [str(cfg["model"])]
    args += ["--host", str(cfg["host"])]
    args += ["--port", str(cfg["port"])]
    args += ["--served-model-name", str(cfg["served_model_name"])]
    args += ["--download-dir", str(cfg["download_dir"])]
    args += ["--gpu-memory-utilization", str(cfg["gpu_memory_utilization"])]
    args += ["--max-model-len", str(cfg["max_model_len"])]
    args += ["--max-num-seqs", str(cfg["max_num_seqs"])]

    if cfg.get("tensor_parallel_size"):
        args += ["--tensor-parallel-size", str(cfg["tensor_parallel_size"])]
    if cfg.get("kv_cache_dtype"):
        args += ["--kv-cache-dtype", str(cfg["kv_cache_dtype"])]
    if cfg.get("tool_call_parser"):
        args += ["--tool-call-parser", str(cfg["tool_call_parser"])]
    if cfg.get("reasoning_parser"):
        args += ["--reasoning-parser", str(cfg["reasoning_parser"])]
    if cfg.get("enable_auto_tool_choice"):
        args += ["--enable-auto-tool-choice"]

    for extra in cfg.get("extra_args", []):
        args.append(str(extra))
    return args


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open(sys.argv[1]))
    for arg in build_vllm_args(cfg):
        print(arg)
