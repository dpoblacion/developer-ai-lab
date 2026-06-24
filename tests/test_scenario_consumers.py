import os
import tempfile
import unittest

import yaml

from scripts.run_gates import load_gates, main


class SchemaContractTest(unittest.TestCase):
    def test_load_gates_reads_nested_task_gates(self):
        # the shipped smoke benchmark has gates under task: — load_gates must find them
        gates = load_gates("benchmarks/smoke/scenario.yaml")
        self.assertTrue(gates, "load_gates returned no gates (task.gates not read?)")

    def test_phases_live_under_task(self):
        d = yaml.safe_load(open("benchmarks/smoke/scenario.yaml"))
        self.assertIn("phases", d["task"])
        self.assertTrue(d["task"]["phases"])


class GatesExitCodeTest(unittest.TestCase):
    def _scenario(self, cmd, expect_exit):
        d = {"name": "t", "task": {"gates": [{"id": "g", "description": "d",
             "cmd": cmd, "expect_exit": expect_exit}]}}
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump(d, f)
        f.close()
        return f.name

    def test_failing_gate_exits_nonzero(self):
        scen = self._scenario("false", 0)   # `false` exits 1, expect 0 -> gate fails
        ws = tempfile.mkdtemp()
        os.makedirs(os.path.join(ws, "wsdir"))
        with self.assertRaises(SystemExit) as cm:
            main(scen, os.path.join(ws, "wsdir"))
        self.assertEqual(cm.exception.code, 1)

    def test_passing_gate_exits_zero(self):
        scen = self._scenario("true", 0)    # `true` exits 0, expect 0 -> gate passes
        ws = tempfile.mkdtemp()
        os.makedirs(os.path.join(ws, "wsdir"))
        with self.assertRaises(SystemExit) as cm:
            main(scen, os.path.join(ws, "wsdir"))
        self.assertEqual(cm.exception.code, 0)
