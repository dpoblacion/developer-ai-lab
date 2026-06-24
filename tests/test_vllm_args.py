import unittest

from scripts.lib.vllm_args import build_vllm_args

QWEN = {
    "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    "served_model_name": "qwen3-coder-30b-fp8",
    "host": "0.0.0.0",
    "port": 8000,
    "download_dir": "/workspace/huggingface",
    "gpu_memory_utilization": 0.92,
    "max_model_len": 32768,
    "max_num_seqs": 8,
}

GLM = {
    **QWEN,
    "model": "zai-org/GLM-5.2-FP8",
    "served_model_name": "glm-5.2-fp8",
    "tensor_parallel_size": 8,
    "kv_cache_dtype": "fp8",
    "tool_call_parser": "glm47",
    "reasoning_parser": "glm45",
    "enable_auto_tool_choice": True,
    "extra_args": ["--speculative-config.method", "mtp",
                   "--speculative-config.num_speculative_tokens", "5"],
}


class BuildVllmArgsTest(unittest.TestCase):
    def test_minimal_config_backward_compatible(self):
        args = build_vllm_args(QWEN)
        self.assertEqual(args[0], "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
        self.assertIn("--served-model-name", args)
        self.assertEqual(args[args.index("--max-model-len") + 1], "32768")
        # no optional flags when absent
        self.assertNotIn("--tensor-parallel-size", args)
        self.assertNotIn("--enable-auto-tool-choice", args)

    def test_glm_config_includes_all_flags(self):
        args = build_vllm_args(GLM)
        self.assertEqual(args[0], "zai-org/GLM-5.2-FP8")
        self.assertEqual(args[args.index("--tensor-parallel-size") + 1], "8")
        self.assertEqual(args[args.index("--kv-cache-dtype") + 1], "fp8")
        self.assertEqual(args[args.index("--tool-call-parser") + 1], "glm47")
        self.assertEqual(args[args.index("--reasoning-parser") + 1], "glm45")
        self.assertIn("--enable-auto-tool-choice", args)
        # extra_args appended verbatim, in order, at the end
        self.assertEqual(args[-4:], ["--speculative-config.method", "mtp",
                                     "--speculative-config.num_speculative_tokens", "5"])


if __name__ == "__main__":
    unittest.main()
