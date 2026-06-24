import unittest

from scripts.lib.dotenv import parse_dotenv, apply_dotenv

TEXT = """
# local secrets
RUNPOD_API_KEY=abc123
SSH_KEY_PATH="/home/u/.ssh/id_ed25519"
EMPTY=
FOO = bar
"""


class ParseDotenvTest(unittest.TestCase):
    def test_parse(self):
        env = parse_dotenv(TEXT)
        self.assertEqual(env["RUNPOD_API_KEY"], "abc123")
        self.assertEqual(env["SSH_KEY_PATH"], "/home/u/.ssh/id_ed25519")  # quotes stripped
        self.assertEqual(env["EMPTY"], "")
        self.assertEqual(env["FOO"], "bar")  # surrounding spaces stripped
        self.assertNotIn("# local secrets", env)

    def test_apply_does_not_override_existing(self):
        target = {"RUNPOD_API_KEY": "already-set"}
        apply_dotenv(TEXT, target)
        self.assertEqual(target["RUNPOD_API_KEY"], "already-set")  # explicit env wins
        self.assertEqual(target["FOO"], "bar")                      # new key added


if __name__ == "__main__":
    unittest.main()
