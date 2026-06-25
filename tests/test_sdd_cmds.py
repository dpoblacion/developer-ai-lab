import unittest

from scripts.lib.sdd_cmds import ssh_tunnel_cmd, docker_run_cmd
from scripts.orchestrate_sdd import toolchain_image


class ToolchainImageTest(unittest.TestCase):
    def test_scenario_without_dockerfile_uses_base(self):
        self.assertEqual(
            toolchain_image("benchmarks/scenarios/smoke/scenario.yaml"),
            "dail-toolchain")

    def test_scenario_with_dockerfile_uses_overlay(self):
        self.assertEqual(
            toolchain_image("benchmarks/scenarios/todo-app/scenario.yaml"),
            "dail-toolchain-todo-app")


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
            "dail-toolchain",
            ["bash", "-lc", "echo hi"],
            mounts=[("/host/repo", "/repo"), ("/host/out", "/out")],
            env={"ANTHROPIC_BASE_URL": "http://x:4000"},
            workdir="/repo")
        self.assertEqual(cmd[:2], ["docker", "run"])
        self.assertIn("--rm", cmd)
        self.assertIn("/host/repo:/repo", cmd)
        self.assertIn("/host/out:/out", cmd)
        self.assertIn("host.docker.internal:host-gateway", cmd)
        self.assertEqual(cmd[cmd.index("-w") + 1], "/repo")
        self.assertEqual(cmd[cmd.index("-e") + 1], "ANTHROPIC_BASE_URL=http://x:4000")
        self.assertEqual(cmd[-4:], ["dail-toolchain", "bash", "-lc", "echo hi"])


if __name__ == "__main__":
    unittest.main()
