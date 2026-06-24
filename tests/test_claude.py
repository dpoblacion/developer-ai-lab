import unittest

from scripts.lib.claude import build_claude_cmd


class BuildClaudeCmdTest(unittest.TestCase):
    def test_builds_headless_command(self):
        cmd = build_claude_cmd("do it", "Read,Edit", 5)
        self.assertEqual(cmd[:3], ["claude", "-p", "do it"])
        self.assertIn("--bare", cmd)
        self.assertEqual(cmd[cmd.index("--allowedTools") + 1], "Read,Edit")
        self.assertEqual(cmd[cmd.index("--max-turns") + 1], "5")
        self.assertEqual(cmd[cmd.index("--output-format") + 1], "stream-json")
        self.assertEqual(cmd[cmd.index("--permission-mode") + 1], "acceptEdits")


if __name__ == "__main__":
    unittest.main()
