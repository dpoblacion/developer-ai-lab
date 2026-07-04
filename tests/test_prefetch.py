import unittest

from scripts.prefetch_to_volume import hf_download_cmd, prefetch_spec


class HfDownloadCmdTest(unittest.TestCase):
    def test_downloads_to_the_volume_cache_with_hf_transfer(self):
        cmd = hf_download_cmd("zai-org/GLM-5.2-FP8", "/workspace/huggingface", "hf_TOK")
        self.assertIn("HF_HUB_ENABLE_HF_TRANSFER=1", cmd)
        self.assertIn("HF_TOKEN=hf_TOK", cmd)
        self.assertIn("zai-org/GLM-5.2-FP8", cmd)
        self.assertIn("/workspace/huggingface", cmd)
        self.assertIn("snapshot_download", cmd)
        self.assertIn("--break-system-packages", cmd)   # base image python is PEP-668 managed

    def test_omits_token_when_absent(self):
        cmd = hf_download_cmd("m/x", "/workspace/huggingface", "")
        self.assertNotIn("HF_TOKEN=", cmd)


class PrefetchSpecTest(unittest.TestCase):
    def test_spec_pins_the_volume_and_dc_no_big_disk(self):
        spec = prefetch_spec("NVIDIA H100 NVL", "vol-1", "US-GA-2")
        self.assertEqual(spec["gpu_type_ids"], ["NVIDIA H100 NVL"])
        self.assertEqual(spec["gpu_count"], 1)            # download needs no GPU parallelism
        self.assertEqual(spec["network_volume_id"], "vol-1")
        self.assertEqual(spec["data_center_id"], "US-GA-2")
        self.assertLessEqual(spec["container_disk_gb"], 40)  # weights land on the volume


if __name__ == "__main__":
    unittest.main()
