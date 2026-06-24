import unittest

from scripts.lib.hardware import detect_gpus, parse_mem_total_gb, collect_environment

SMI = "NVIDIA H200, 143771\nNVIDIA H200, 143771\n"
MEMINFO = "MemTotal:       263852132 kB\nMemFree:  1000 kB\n"


class HardwareTest(unittest.TestCase):
    def test_detect_gpus(self):
        gpus = detect_gpus(SMI)
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0], {"name": "NVIDIA H200", "memory_mib": 143771})

    def test_parse_mem_total_gb(self):
        self.assertEqual(parse_mem_total_gb(MEMINFO), 251.6)

    def test_collect_environment(self):
        env = collect_environment(
            gpus=detect_gpus(SMI), ram_gb=251.6, cpu_count=64,
            env={"HW_PROVIDER": "RunPod", "HW_INSTANCE": "8x H200"})
        self.assertEqual(env["provider"], "RunPod")
        self.assertEqual(env["instance"], "8x H200")
        self.assertEqual(env["gpu_count"], 2)
        self.assertEqual(env["cpu_count"], 64)
        self.assertEqual(env["ram_gb"], 251.6)
        self.assertNotIn("usd_per_hour", env)


if __name__ == "__main__":
    unittest.main()
