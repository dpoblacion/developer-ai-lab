import pathlib
import unittest

import yaml

from scripts.lib.compose import load_run_config
from scripts.run_gates import load_gates

BENCH = "benchmarks/dev-load"
SCENARIO = f"{BENCH}/scenario.yaml"


class DevLoadBenchmarkTest(unittest.TestCase):
    def test_schema_shape(self):
        d = yaml.safe_load(pathlib.Path(SCENARIO).read_text())
        self.assertEqual(d["name"], "dev-load")
        self.assertNotIn("type", d)                       # new-model benchmark
        self.assertNotIn("model", d)
        # devs is a tunable experiment knob — pin the SHAPE (sorted positive ints), not
        # the values, so editing the team sizes doesn't break the suite.
        self.assertTrue(d["devs"])
        self.assertTrue(all(isinstance(n, int) and n >= 1 for n in d["devs"]))
        self.assertEqual(d["devs"], sorted(d["devs"]))
        self.assertNotIn("concurrency", d)
        self.assertEqual(len(d["task"]["phases"]), 4)
        self.assertEqual(len(d["task"]["gates"]), 2)

    def test_load_gates_finds_two(self):
        self.assertEqual(len(load_gates(SCENARIO)), 2)

    def test_run_config_resolves_variant_and_max_seqs(self):
        vllm_cfg, pod_spec, variant, devs, slo = load_run_config(
            SCENARIO, "qwen3-coder", "l40s", gpu_count=1)
        self.assertEqual(variant["family"], "qwen3-coder")
        self.assertEqual(variant["quant"], "fp8")               # l40s supports fp8
        self.assertEqual(vllm_cfg["max_num_seqs"], max(devs))   # admits the largest N
        self.assertTrue(devs)                                   # values are a tunable knob

    def test_phase_prompt_files_exist(self):
        d = yaml.safe_load(pathlib.Path(SCENARIO).read_text())
        for ph in d["task"]["phases"]:
            self.assertTrue((pathlib.Path(BENCH) / ph["prompt"]).exists(), ph["prompt"])

    def test_acceptance_test_shipped(self):
        self.assertTrue(pathlib.Path(f"{BENCH}/acceptance/acceptance.test.ts").exists())
