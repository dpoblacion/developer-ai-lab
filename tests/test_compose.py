import unittest

from scripts.lib.compose import compose, resolve_variant, load_variants

MODEL = {
    "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    "served_model_name": "qwen3-coder-30b-fp8",
    "vllm_version": "0.11.0",
    "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    "container_disk_gb": 100,
    "pip_constraints": ["transformers<5"],
    "tool_call_parser": "qwen3_xml",
    "enable_auto_tool_choice": True,
    "env": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
}
HW_1GPU = {"gpu_type_ids": ["NVIDIA L40S"],
           "price_usd_per_gpu_hour": 0.79, "gpu_memory_utilization": 0.92}
HW_8GPU = {"gpu_type_ids": ["NVIDIA H200"],
           "price_usd_per_gpu_hour": 3.59, "gpu_memory_utilization": 0.95}
BENCH_SDD = {"concurrency": [1], "slo": {"max_ttft": 2.0, "min_tps": 20.0},
             "serving": {"max_model_len": 65536},
             "task": {"phases": [], "gates": []}}


class ComposeTest(unittest.TestCase):
    def test_vllm_cfg_merges_axes(self):
        vllm_cfg, _ = compose(MODEL, HW_1GPU, BENCH_SDD)
        # constants
        self.assertEqual(vllm_cfg["host"], "0.0.0.0")
        self.assertEqual(vllm_cfg["port"], 8000)
        self.assertEqual(vllm_cfg["download_dir"], "/workspace/huggingface")
        # model recipe
        self.assertEqual(vllm_cfg["model"], MODEL["model"])
        self.assertEqual(vllm_cfg["tool_call_parser"], "qwen3_xml")
        self.assertEqual(vllm_cfg["vllm_version"], "0.11.0")
        self.assertEqual(vllm_cfg["pip_constraints"], ["transformers<5"])
        # hardware
        self.assertEqual(vllm_cfg["gpu_memory_utilization"], 0.92)
        # benchmark workload
        self.assertEqual(vllm_cfg["max_model_len"], 65536)
        self.assertEqual(vllm_cfg["max_num_seqs"], 1)

    def test_tensor_parallel_from_gpu_count_argument(self):
        vllm_cfg, _ = compose(MODEL, HW_8GPU, BENCH_SDD, gpu_count=8)
        self.assertEqual(vllm_cfg["tensor_parallel_size"], 8)

    def test_default_gpu_count_is_one(self):
        vllm_cfg, pod_spec = compose(MODEL, HW_1GPU, BENCH_SDD)
        self.assertEqual(vllm_cfg["tensor_parallel_size"], 1)
        self.assertEqual(pod_spec["gpu_count"], 1)

    def test_gpu_count_flows_to_pod_spec(self):
        vllm_cfg, pod_spec = compose(MODEL, HW_1GPU, BENCH_SDD, gpu_count=4)
        self.assertEqual(vllm_cfg["tensor_parallel_size"], 4)
        self.assertEqual(pod_spec["gpu_count"], 4)

    def test_pod_spec_shape(self):
        _, pod_spec = compose(MODEL, HW_1GPU, BENCH_SDD)
        self.assertEqual(pod_spec["image"], MODEL["image"])
        self.assertEqual(pod_spec["container_disk_gb"], 100)
        self.assertEqual(pod_spec["gpu_type_ids"], HW_1GPU["gpu_type_ids"])
        self.assertEqual(pod_spec["gpu_count"], 1)
        # Only SSH is exposed: vLLM is reached over the SSH tunnel, never RunPod's public
        # proxy. Exposing 8000/http would serve the model unauthenticated to anyone with
        # the pod URL (and let them skew the benchmark's own latency numbers via contention).
        self.assertEqual(pod_spec["ports"], "22/tcp")
        self.assertEqual(pod_spec["name"], "dail-qwen3-coder-30b-fp8")

    def test_benchmark_serving_overrides_hardware(self):
        bench = {"serving": {"max_model_len": 32768, "max_num_seqs": 8,
                             "gpu_memory_utilization": 0.85}}
        vllm_cfg, _ = compose(MODEL, HW_1GPU, bench)
        self.assertEqual(vllm_cfg["gpu_memory_utilization"], 0.85)  # benchmark wins

    def test_max_num_seqs_derived_from_devs(self):
        bench = {"devs": [4, 8], "serving": {"max_model_len": 65536},
                 "slo": {"max_ttft": 2.0, "min_tps": 20.0}}
        vllm_cfg, _ = compose(MODEL, HW_1GPU, bench, gpu_count=1)
        self.assertEqual(vllm_cfg["max_num_seqs"], 8)

    def test_model_without_image_raises_clear_error(self):
        model = {k: v for k, v in MODEL.items() if k != "image"}  # e.g. unprovisioned GLM
        with self.assertRaises(ValueError) as ctx:
            compose(model, HW_1GPU, BENCH_SDD)
        msg = str(ctx.exception)
        self.assertIn("qwen3-coder-30b-fp8", msg)   # names the offending model
        self.assertIn("image", msg)

    def test_gpu_count_below_one_raises(self):
        # --gpus 0 would otherwise emit tensor_parallel_size: 0 (silently dropped by
        # build_vllm_args' truthiness guard) and a gpu_count: 0 pod create. Fail upfront.
        with self.assertRaises(ValueError) as ctx:
            compose(MODEL, HW_1GPU, BENCH_SDD, gpu_count=0)
        self.assertIn("gpu_count", str(ctx.exception))


def _variants():
    return [{"family": "qwen3-coder", "quant": "fp8", "served_model_name": "q-fp8"},
            {"family": "qwen3-coder", "quant": "awq", "served_model_name": "q-awq"},
            {"family": "glm", "quant": "fp8", "served_model_name": "glm-fp8"}]


class ResolveVariantTest(unittest.TestCase):
    def test_picks_first_supported_quant_present_in_family(self):
        hw = {"supported_quants": ["fp8", "awq", "bf16"]}      # Ada/Hopper
        self.assertEqual(resolve_variant("qwen3-coder", hw, _variants())["quant"], "fp8")

    def test_falls_to_next_supported_when_top_absent(self):
        hw = {"supported_quants": ["awq", "gptq", "bf16"]}     # Ampere: no fp8
        self.assertEqual(resolve_variant("qwen3-coder", hw, _variants())["quant"], "awq")

    def test_no_variant_for_supported_quants_raises(self):
        hw = {"supported_quants": ["gptq", "bf16"]}            # family has only fp8/awq
        with self.assertRaises(SystemExit) as cm:
            resolve_variant("qwen3-coder", hw, _variants())
        self.assertIn("qwen3-coder", str(cm.exception))

    def test_unknown_family_raises(self):
        with self.assertRaises(SystemExit) as cm:
            resolve_variant("nope", {"supported_quants": ["fp8"]}, _variants())
        self.assertIn("nope", str(cm.exception))


class LoadVariantsTest(unittest.TestCase):
    def test_loads_the_model_configs_with_family_and_quant(self):
        variants = load_variants("configs/models")
        self.assertGreaterEqual(len(variants), 2)
        for v in variants:
            self.assertIn("family", v)
            self.assertIn("quant", v)


from scripts.lib.compose import load_run_config
import scripts.lib.compose as compose_module

HW_WITH_PROVIDER = {"gpu_type_ids": ["NVIDIA L40S"],
                    "price_usd_per_gpu_hour": 0.79, "gpu_memory_utilization": 0.92,
                    "provider": "RunPod", "instance": "NVIDIA L40S"}


class PodSpecEnvTest(unittest.TestCase):
    def test_hw_provider_instance_in_pod_spec_env(self):
        _, pod_spec = compose(MODEL, HW_WITH_PROVIDER, BENCH_SDD)
        self.assertEqual(pod_spec["env"]["HW_PROVIDER"], "RunPod")
        self.assertEqual(pod_spec["env"]["HW_INSTANCE"], "NVIDIA L40S")

    def test_hw_without_provider_yields_empty_env(self):
        # HW_1GPU has no provider/instance — env should not contain those keys
        _, pod_spec = compose(MODEL, HW_1GPU, BENCH_SDD)
        self.assertNotIn("HW_PROVIDER", pod_spec["env"])
        self.assertNotIn("HW_INSTANCE", pod_spec["env"])

    def test_pod_constants_not_mutated(self):
        # shallow copy of POD_CONSTANTS means pod_spec["env"] could alias the module dict
        compose(MODEL, HW_WITH_PROVIDER, BENCH_SDD)
        self.assertEqual(compose_module.POD_CONSTANTS["env"], {})


class LoadRunConfigTest(unittest.TestCase):
    def test_qwen_l40s_smoke(self):
        vllm_cfg, _, variant, devs, _ = load_run_config(
            "benchmarks/smoke/scenario.yaml", "qwen3-coder", "l40s")
        self.assertEqual(vllm_cfg["max_num_seqs"], 1)        # smoke has devs: [1]
        self.assertEqual(vllm_cfg["max_model_len"], 65536)
        self.assertEqual(variant["quant"], "fp8")            # l40s prefers fp8
        self.assertEqual(devs, [1])

    def test_gpu_count_override(self):
        vllm_cfg, pod_spec, _, _, _ = load_run_config(
            "benchmarks/smoke/scenario.yaml", "qwen3-coder", "l40s", gpu_count=2)
        self.assertEqual(vllm_cfg["tensor_parallel_size"], 2)
        self.assertEqual(pod_spec["gpu_count"], 2)

    def test_h200_gpu_count(self):
        vllm_cfg, pod_spec, _, _, _ = load_run_config(
            "benchmarks/smoke/scenario.yaml", "qwen3-coder", "h200", gpu_count=8)
        self.assertEqual(vllm_cfg["tensor_parallel_size"], 8)
        self.assertEqual(pod_spec["gpu_count"], 8)
        self.assertEqual(pod_spec["gpu_type_ids"][0], "NVIDIA H200")
        self.assertEqual(pod_spec["env"]["HW_INSTANCE"], "NVIDIA H200")

    def test_qwen_l40s_dev_load_env_and_serving(self):
        vllm_cfg, pod_spec, _, _, _ = load_run_config(
            "benchmarks/dev-load/scenario.yaml", "qwen3-coder", "l40s")
        self.assertEqual(vllm_cfg["max_model_len"], 65536)          # dev-load serving override
        self.assertEqual(pod_spec["env"]["HW_PROVIDER"], "RunPod")
        self.assertEqual(pod_spec["env"]["HW_INSTANCE"], "NVIDIA L40S")

    def test_no_hardware_config_carries_count_or_old_price_key(self):
        # Regression guard for the per-GPU-type migration: count is the --gpus run param,
        # and the price key is per-GPU. A reintroduced gpu_count / old price key would
        # silently revert the schema.
        import pathlib
        import yaml
        configs = list(pathlib.Path("configs/hardware").glob("*.yaml"))
        self.assertTrue(configs)  # don't let an empty glob pass vacuously
        for path in configs:
            data = yaml.safe_load(path.read_text())
            self.assertNotIn("gpu_count", data, f"{path.name} reintroduced gpu_count")
            self.assertNotIn("price_usd_per_hour", data, f"{path.name} uses the old price key")
            self.assertIn("price_usd_per_gpu_hour", data, f"{path.name} missing per-GPU price")


class GlmComposesTest(unittest.TestCase):
    """GLM-5.2 needs an explicit image (vLLM 0.23 / newer CUDA) and disk that fits its
    ~744GB FP8 weights. compose() requires `image`, so a blank one fails the run loudly."""

    def test_glm_h200_8gpu_composes_with_image_and_enough_disk(self):
        vllm_cfg, pod_spec, variant, _, _ = load_run_config(
            "benchmarks/dev-load/scenario.yaml", "glm-5.2", "h200", gpu_count=8)
        self.assertTrue(pod_spec["image"], "GLM image must be set (was intentionally blank)")
        # Must be an SSH-capable base image, NOT vllm/vllm-openai (no sshd — the harness
        # needs SSH for rsync + scripts; that combo failed live on 2026-07-03).
        self.assertNotIn("vllm/vllm-openai", pod_spec["image"])
        self.assertGreaterEqual(pod_spec["container_disk_gb"], 800)  # 744GB weights + overhead
        self.assertEqual(variant["quant"], "fp8")
        self.assertEqual(pod_spec["gpu_count"], 8)
        # vLLM 0.23's wheel is CUDA 13; the base image must match (cu130), not cu129.
        self.assertIn("cu130", pod_spec["image"])


class DefaultSloTest(unittest.TestCase):
    def test_default_slo_is_a_single_shared_constant(self):
        # The default SLO used to be duplicated in compose.py and bench_sdd.py — a drift
        # there would silently score runs against two different SLOs.
        from scripts.lib import bench_sdd, compose
        self.assertEqual(compose.DEFAULT_SLO, {"max_ttft": 2.0, "min_tps": 20.0})
        self.assertIs(bench_sdd.DEFAULT_SLO, compose.DEFAULT_SLO)


class SmokeScenarioTest(unittest.TestCase):
    def test_smoke_migrated_to_model_and_devs(self):
        import yaml
        import pathlib
        d = yaml.safe_load(pathlib.Path("benchmarks/smoke/scenario.yaml").read_text())
        self.assertNotIn("model", d)
        self.assertEqual(d["devs"], [1])
        self.assertNotIn("concurrency", d)
