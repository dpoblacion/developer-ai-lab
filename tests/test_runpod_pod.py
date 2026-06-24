import unittest

from scripts.lib.runpod_pod import (
    build_create_kwargs, is_ready, ssh_endpoint,
    ssh_run_cmd, rsync_up_cmd, rsync_down_cmd)

SPEC = {
    "name": "dail-qwen",
    "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    "gpu_type_ids": ["NVIDIA L40S", "NVIDIA A100 80GB PCIe"],
    "gpu_count": 1,
    "container_disk_gb": 100,
    "volume_gb": 0,
    "ports": "8000/http,22/tcp",
    "env": {"FOO": "bar"},
}


class BuildCreateKwargsTest(unittest.TestCase):
    def test_maps_spec_to_sdk_kwargs(self):
        kw = build_create_kwargs(SPEC)
        self.assertEqual(kw["name"], "dail-qwen")
        self.assertEqual(kw["image_name"], SPEC["image"])
        self.assertEqual(kw["gpu_type_id"], "NVIDIA L40S")   # first candidate by default
        self.assertEqual(kw["gpu_count"], 1)
        self.assertEqual(kw["container_disk_in_gb"], 100)
        self.assertEqual(kw["ports"], "8000/http,22/tcp")
        self.assertTrue(kw["support_public_ip"])
        self.assertTrue(kw["start_ssh"])
        self.assertEqual(kw["env"], {"FOO": "bar"})

    def test_gpu_override(self):
        kw = build_create_kwargs(SPEC, gpu_type_id="NVIDIA A100 80GB PCIe")
        self.assertEqual(kw["gpu_type_id"], "NVIDIA A100 80GB PCIe")


class ReadinessTest(unittest.TestCase):
    def test_is_ready(self):
        self.assertFalse(is_ready({"runtime": None}))
        self.assertFalse(is_ready({"runtime": {"ports": []}}))
        self.assertTrue(is_ready({"runtime": {"ports": [{"privatePort": 22}]}}))

    def test_ssh_endpoint(self):
        pod = {"runtime": {"ports": [
            {"privatePort": 8000, "isIpPublic": True, "ip": "1.2.3.4", "publicPort": 1},
            {"privatePort": 22, "isIpPublic": True, "ip": "1.2.3.4", "publicPort": 40022},
        ]}}
        self.assertEqual(ssh_endpoint(pod), ("1.2.3.4", 40022))

    def test_ssh_endpoint_none_when_no_public_ssh(self):
        pod = {"runtime": {"ports": [{"privatePort": 22, "isIpPublic": False}]}}
        self.assertIsNone(ssh_endpoint(pod))


class CommandBuilderTest(unittest.TestCase):
    def test_ssh_run_cmd(self):
        cmd = ssh_run_cmd("1.2.3.4", 40022, "/k", "echo hi")
        self.assertEqual(cmd[0], "ssh")
        self.assertEqual(cmd[cmd.index("-p") + 1], "40022")
        self.assertEqual(cmd[cmd.index("-i") + 1], "/k")
        self.assertEqual(cmd[-2:], ["root@1.2.3.4", "echo hi"])

    def test_rsync_up_cmd_has_excludes_and_endpoint(self):
        cmd = rsync_up_cmd("1.2.3.4", 40022, "/k", "./", "/workspace/repo",
                           excludes=[".git", "results"])
        self.assertEqual(cmd[0], "rsync")
        self.assertIn("--delete", cmd)
        self.assertEqual(cmd.count("--exclude"), 2)
        self.assertEqual(cmd[-1], "root@1.2.3.4:/workspace/repo")
        e = cmd[cmd.index("-e") + 1]
        self.assertTrue(e.startswith("ssh ") and "-p 40022" in e)

    def test_rsync_down_cmd(self):
        cmd = rsync_down_cmd("1.2.3.4", 40022, "/k", "/workspace/repo/results/", "results/")
        self.assertEqual(cmd[0], "rsync")
        self.assertEqual(cmd[-2:], ["root@1.2.3.4:/workspace/repo/results/", "results/"])


if __name__ == "__main__":
    unittest.main()
