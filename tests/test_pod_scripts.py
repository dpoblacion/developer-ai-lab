"""Behavioral tests for the pod bash scripts, driven with stubs (no real pod)."""
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
PYBIN = str(pathlib.Path(sys.executable).parent)   # a python3 that has pyyaml


class SetupPodImageVllmTest(unittest.TestCase):
    """Camino A: images like vllm/vllm-openai ship vLLM already. setup_pod must REUSE it —
    creating a venv and reinstalling would duplicate the ~4min install and risk a version
    clash. When no vllm is on PATH (our pytorch base images), the normal venv path runs."""

    def _run(self, extra_path=""):
        with tempfile.TemporaryDirectory() as d:
            cfg = pathlib.Path(d) / "cfg.yaml"
            cfg.write_text("download_dir: /tmp/hf-test\nvllm_version: '0.23.0'\n")
            env = {**os.environ, "INSTALL_CLAUDE": "0", "HOME": d}
            env["PATH"] = f"{extra_path}:{PYBIN}:{env['PATH']}" if extra_path \
                else f"{PYBIN}:{env['PATH']}"
            return subprocess.run(["bash", str(REPO / "scripts/setup_pod.sh"), str(cfg)],
                                  capture_output=True, text=True, timeout=60, env=env)

    def test_reuses_image_vllm_and_skips_venv(self):
        with tempfile.TemporaryDirectory() as bindir:
            stub = pathlib.Path(bindir) / "vllm"
            stub.write_text("#!/bin/bash\necho 'vLLM 0.23.0'\n")
            stub.chmod(0o755)
            r = self._run(extra_path=bindir)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("reusing", (r.stdout + r.stderr).lower())
            self.assertNotIn("python3 -m venv", r.stdout)   # did not build a venv
