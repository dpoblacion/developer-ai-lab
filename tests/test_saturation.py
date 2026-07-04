import threading
import unittest

from scripts.lib.saturation import probe_shape, run_probe, summarize_probe

BEFORE = """
vllm:prompt_tokens_total{engine="0"} 1000
vllm:generation_tokens_total{engine="0"} 500
"""
AFTER = """
vllm:prompt_tokens_total{engine="0"} 3601000
vllm:generation_tokens_total{engine="0"} 1200500
"""


class ProbeShapeTest(unittest.TestCase):
    def test_matches_the_workload_io_ratio(self):
        # The workload served 20:1 prompt:generation -> the probe keeps that ratio so
        # Θmax is the ceiling for THIS traffic shape (arXiv:2606.11690 §5.7: the I/O mix
        # moves the absolute cost level, so Θmax must be shape-matched to be comparable).
        shape = probe_shape(prompt_tokens=2_000_000, generation_tokens=100_000,
                            output_tokens=256)
        self.assertEqual(shape["output_tokens"], 256)
        self.assertEqual(shape["prompt_tokens"], 5120)     # 20 x 256

    def test_defaults_to_paper_shape_without_workload_data(self):
        # No measured traffic to mirror -> the paper's 512:256 reference shape.
        shape = probe_shape(prompt_tokens=0, generation_tokens=0, output_tokens=256)
        self.assertEqual(shape, {"prompt_tokens": 512, "output_tokens": 256})

    def test_prompt_clamped_to_sane_bounds(self):
        shape = probe_shape(prompt_tokens=10_000_000, generation_tokens=100,
                            output_tokens=256)
        self.assertLessEqual(shape["prompt_tokens"], 8192)
        shape = probe_shape(prompt_tokens=1, generation_tokens=10_000, output_tokens=256)
        self.assertGreaterEqual(shape["prompt_tokens"], 128)


class SummarizeProbeTest(unittest.TestCase):
    def test_theta_max_from_counter_diffs(self):
        out = summarize_probe(BEFORE, AFTER, duration_s=1800.0,
                              shape={"prompt_tokens": 768, "output_tokens": 256},
                              concurrency=64)
        # 3.6M prompt + 1.2M generation tokens in 0.5h
        self.assertAlmostEqual(out["tokens_per_hour"], 9_600_000.0)
        self.assertAlmostEqual(out["output_tokens_per_hour"], 2_400_000.0)
        self.assertEqual(out["shape"], {"prompt_tokens": 768, "output_tokens": 256})
        self.assertEqual(out["concurrency"], 64)
        self.assertEqual(out["duration_s"], 1800.0)


class RunProbeTest(unittest.TestCase):
    def test_drives_requests_until_the_window_closes(self):
        clock = {"t": 0.0}
        calls = []
        lock = threading.Lock()

        def request_fn():
            with lock:
                calls.append(1)
                clock["t"] += 10.0          # each request advances fake time 10s

        snapshots = iter([BEFORE, AFTER])
        out = run_probe(request_fn, lambda: next(snapshots), lambda: clock["t"],
                        duration_s=60.0, concurrency=2,
                        shape={"prompt_tokens": 512, "output_tokens": 256})
        self.assertGreaterEqual(len(calls), 6)             # kept firing for the window
        self.assertGreater(out["tokens_per_hour"], 0)
        self.assertEqual(out["shape"]["prompt_tokens"], 512)
