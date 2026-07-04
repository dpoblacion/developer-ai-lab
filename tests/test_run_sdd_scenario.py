import json
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

import yaml

from scripts import run_sdd_scenario


def _scenario(dir_):
    (dir_ / "p1.md").write_text("do X")
    (dir_ / "p2.md").write_text("do Y")
    sc = {"name": "t", "task": {"phases": [
        {"id": "01", "title": "one", "prompt": "p1.md"},
        {"id": "02", "title": "two", "prompt": "p2.md"}]}}
    path = dir_ / "scenario.yaml"
    path.write_text(yaml.safe_dump(sc))
    return path


class DestructiveAgentTest(unittest.TestCase):
    def test_survives_the_agent_deleting_the_output_tree(self):
        # The agent has Bash with cwd inside /out: a model can (and did, live
        # 2026-07-04, N=1 on l40s) wipe /out/phases mid-run. The harness must still
        # record the transcript and finish — the destruction then surfaces as gate
        # failures (a quality signal), not as a harness crash that loses the level.
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            scenario = _scenario(d)
            out = d / "out"
            calls = {"n": 0}

            def destructive_invoke(prompt, allowed_tools, max_turns, cwd):
                calls["n"] += 1
                if calls["n"] == 2:
                    shutil.rmtree(out / "phases")     # the model nukes the harness dir
                return '{"type": "result", "usage": {"input_tokens": 5}}', 1.0

            with mock.patch.object(run_sdd_scenario, "invoke_claude", destructive_invoke), \
                 mock.patch.dict("os.environ", {"SDD_OUT_DIR": str(out)}):
                run_sdd_scenario.run(str(scenario))
            # 01.jsonl died with the wiped dir (the agent deleted it — unrecoverable);
            # what matters is that the run kept going and recorded everything after.
            self.assertTrue((out / "phases" / "02.jsonl").exists())
            gen = json.loads((out / "generation.json").read_text())
            self.assertEqual(len(gen["phases"]), 2)
