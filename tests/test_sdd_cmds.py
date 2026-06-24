import unittest

from scripts.lib.sdd_cmds import ssh_tunnel_cmd, docker_run_cmd


class SddCmdsTest(unittest.TestCase):
    def test_ssh_tunnel_cmd(self):
        cmd = ssh_tunnel_cmd("1.2.3.4", 40022, "/k", local=8000, remote=8000)
        self.assertEqual(cmd[0], "ssh")
        self.assertIn("-N", cmd)
        self.assertEqual(cmd[cmd.index("-L") + 1], "8000:localhost:8000")
        self.assertEqual(cmd[cmd.index("-p") + 1], "40022")
        self.assertEqual(cmd[cmd.index("-i") + 1], "/k")
        self.assertEqual(cmd[-1], "root@1.2.3.4")

    def test_docker_run_cmd(self):
        cmd = docker_run_cmd(
            "dail-toolchain", "/host/ws", {"ANTHROPIC_BASE_URL": "http://x:4000"},
            ["bash", "-lc", "echo hi"], name="dail-sdd")
        self.assertEqual(cmd[:2], ["docker", "run"])
        self.assertIn("--rm", cmd)
        self.assertIn("/host/ws:/workspace", cmd)
        self.assertIn("/var/run/docker.sock:/var/run/docker.sock", cmd)
        self.assertIn("host.docker.internal:host-gateway", cmd)
        self.assertEqual(cmd[cmd.index("-e") + 1], "ANTHROPIC_BASE_URL=http://x:4000")
        self.assertEqual(cmd[-4:], ["dail-toolchain", "bash", "-lc", "echo hi"])


if __name__ == "__main__":
    unittest.main()
