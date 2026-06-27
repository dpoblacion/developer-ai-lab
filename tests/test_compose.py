import unittest

from scripts.lib.compose import compose

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
HW_1GPU = {"gpu_type_ids": ["NVIDIA L40S", "NVIDIA A100 80GB PCIe"], "gpu_count": 1,
           "price_usd_per_hour": 0.79, "gpu_memory_utilization": 0.92}
HW_8GPU = {"gpu_type_ids": ["NVIDIA H200"], "gpu_count": 8,
           "price_usd_per_hour": 28.7, "gpu_memory_utilization": 0.95}
BENCH_SDD = {"type": "sdd", "serving": {"max_model_len": 65536, "max_num_seqs": 2}}


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
        self.assertEqual(vllm_cfg["max_num_seqs"], 2)

    def test_tensor_parallel_derived_from_gpu_count(self):
        vllm_cfg, _ = compose(MODEL, HW_8GPU, BENCH_SDD)
        self.assertEqual(vllm_cfg["tensor_parallel_size"], 8)

    def test_pod_spec_shape(self):
        _, pod_spec = compose(MODEL, HW_1GPU, BENCH_SDD)
        self.assertEqual(pod_spec["image"], MODEL["image"])
        self.assertEqual(pod_spec["container_disk_gb"], 100)
        self.assertEqual(pod_spec["gpu_type_ids"], HW_1GPU["gpu_type_ids"])
        self.assertEqual(pod_spec["gpu_count"], 1)
        self.assertEqual(pod_spec["ports"], "8000/http,22/tcp")
        self.assertEqual(pod_spec["name"], "dail-qwen3-coder-30b-fp8")

    def test_benchmark_serving_overrides_hardware(self):
        bench = {"serving": {"max_model_len": 32768, "max_num_seqs": 8,
                             "gpu_memory_utilization": 0.85}}
        vllm_cfg, _ = compose(MODEL, HW_1GPU, bench)
        self.assertEqual(vllm_cfg["gpu_memory_utilization"], 0.85)  # benchmark wins

    def test_model_without_image_raises_clear_error(self):
        model = {k: v for k, v in MODEL.items() if k != "image"}  # e.g. unprovisioned GLM
        with self.assertRaises(ValueError) as ctx:
            compose(model, HW_1GPU, BENCH_SDD)
        msg = str(ctx.exception)
        self.assertIn("qwen3-coder-30b-fp8", msg)   # names the offending model
        self.assertIn("image", msg)


from scripts.lib.compose import load_config
import scripts.lib.compose as compose_module

HW_WITH_PROVIDER = {"gpu_type_ids": ["NVIDIA L40S"], "gpu_count": 1,
                    "price_usd_per_hour": 0.79, "gpu_memory_utilization": 0.92,
                    "provider": "RunPod", "instance": "1x NVIDIA L40S"}


class PodSpecEnvTest(unittest.TestCase):
    def test_hw_provider_instance_in_pod_spec_env(self):
        _, pod_spec = compose(MODEL, HW_WITH_PROVIDER, BENCH_SDD)
        self.assertEqual(pod_spec["env"]["HW_PROVIDER"], "RunPod")
        self.assertEqual(pod_spec["env"]["HW_INSTANCE"], "1x NVIDIA L40S")

    def test_hw_without_provider_yields_empty_env(self):
        # HW_1GPU has no provider/instance — env should not contain those keys
        _, pod_spec = compose(MODEL, HW_1GPU, BENCH_SDD)
        self.assertNotIn("HW_PROVIDER", pod_spec["env"])
        self.assertNotIn("HW_INSTANCE", pod_spec["env"])

    def test_pod_constants_not_mutated(self):
        # shallow copy of POD_CONSTANTS means pod_spec["env"] could alias the module dict
        compose(MODEL, HW_WITH_PROVIDER, BENCH_SDD)
        self.assertEqual(compose_module.POD_CONSTANTS["env"], {})


class LoadConfigTest(unittest.TestCase):
    def test_qwen_l40s_todo_app(self):
        vllm_cfg, pod_spec = load_config("qwen3-coder", "l40s",
                                         "benchmarks/todo-app/scenario.yaml")
        self.assertEqual(vllm_cfg["model"], "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
        self.assertEqual(vllm_cfg["tool_call_parser"], "qwen3_xml")
        self.assertEqual(vllm_cfg["max_model_len"], 65536)   # from the todo-app serving block
        self.assertEqual(vllm_cfg["max_num_seqs"], 2)
        self.assertEqual(vllm_cfg["gpu_memory_utilization"], 0.92)
        self.assertEqual(vllm_cfg["tensor_parallel_size"], 1)
        self.assertEqual(pod_spec["gpu_type_ids"][0], "NVIDIA L40S")
        self.assertEqual(pod_spec["image"], "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404")
        # provider/instance from l40s.yaml hardware config
        self.assertEqual(pod_spec["env"]["HW_PROVIDER"], "RunPod")
        self.assertEqual(pod_spec["env"]["HW_INSTANCE"], "1x NVIDIA L40S")

    def test_qwen_l40s_concurrency(self):
        vllm_cfg, _ = load_config("qwen3-coder", "l40s",
                                  "benchmarks/concurrency/scenario.yaml")
        self.assertEqual(vllm_cfg["max_num_seqs"], 8)        # concurrency workload
        self.assertEqual(vllm_cfg["max_model_len"], 32768)
