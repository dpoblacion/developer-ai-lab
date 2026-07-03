import json
import os
import pathlib
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from scripts.lib import bench_sdd, pod_guard
from scripts.lib.bench_sdd import _load_json
from tests.support import make_guard


class LoadJsonTest(unittest.TestCase):
    def test_reads_json_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            p.write_text(json.dumps({"a": 1}))
            self.assertEqual(_load_json(p), {"a": 1})

    def test_missing_file_returns_empty(self):
        self.assertEqual(_load_json(Path("/no/such/file.json")), {})

    def test_invalid_json_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            p.write_text("{not json")
            self.assertEqual(_load_json(p), {})


class VllmFailScanTest(unittest.TestCase):
    """The startup poll greps vllm.log for fatal patterns to fail fast instead of burning
    MAX_STARTUP of GPU. Run the real shell pipeline against sample logs: non-empty output
    means 'fatal detected'."""

    def scan(self, log_text):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write(log_text)
            path = f.name
        out = subprocess.run(["bash", "-c", bench_sdd.vllm_fail_scan_cmd(path)],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip()

    def test_healthy_loading_log_not_flagged(self):
        self.assertEqual(self.scan(
            "INFO 07-01 loading model weights ...\n"
            "INFO 07-01 still loading shard 3/8 ...\n"), "")

    def test_benign_fp8_warning_not_flagged(self):
        self.assertEqual(self.scan(
            "WARNING 07-01 Config file not found at /x/y.json, using defaults\n"
            "INFO 07-01 loading ...\n"), "")

    def test_traceback_flagged(self):
        self.assertNotEqual(self.scan(
            "INFO 07-01 loading ...\n"
            "Traceback (most recent call last):\n"
            '  File "serve.py", line 1\n'), "")

    def test_cuda_oom_flagged(self):
        self.assertNotEqual(self.scan(
            "INFO 07-01 loading ...\n"
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.2 GiB\n"), "")

    def test_oom_killer_flagged(self):
        self.assertNotEqual(self.scan(
            "INFO 07-01 loading ...\n"
            "./scripts/start_vllm.sh: line 44:  1234 Killed  vllm serve ...\n"), "")

    def test_value_error_flagged(self):
        self.assertNotEqual(self.scan(
            "ValueError: model architecture not supported by this vLLM build\n"), "")


class VllmExitSentinelTest(unittest.TestCase):
    """A grep blocklist can't know every fatal message. The launch command writes the
    chain's exit code to a sentinel file when it dies; the scan flags the sentinel, so ANY
    startup death fails fast regardless of what it logged."""

    def test_scan_flags_exit_sentinel_even_with_clean_log(self):
        with tempfile.TemporaryDirectory() as d:
            log_p = pathlib.Path(d) / "vllm.log"
            log_p.write_text("INFO all good, still loading\n")
            exit_p = pathlib.Path(d) / "vllm.exit"
            exit_p.write_text("3\n")
            out = subprocess.run(["bash", "-c",
                                  bench_sdd.vllm_fail_scan_cmd(str(log_p), str(exit_p))],
                                 capture_output=True, text=True, timeout=10)
            self.assertNotEqual(out.stdout.strip(), "")
            self.assertIn("exit", out.stdout.lower())

    def test_scan_quiet_without_sentinel_and_clean_log(self):
        with tempfile.TemporaryDirectory() as d:
            log_p = pathlib.Path(d) / "vllm.log"
            log_p.write_text("INFO loading shard 3/8\n")
            out = subprocess.run(["bash", "-c",
                                  bench_sdd.vllm_fail_scan_cmd(str(log_p), str(d) + "/vllm.exit")],
                                 capture_output=True, text=True, timeout=10)
            self.assertEqual(out.stdout.strip(), "")

    def test_launch_cmd_writes_sentinel_when_the_chain_dies(self):
        # Run the REAL launch command with stub setup/start scripts: setup ok, vllm dies
        # with 3 → the sentinel must hold 3 and the log must have both scripts' output.
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "scripts").mkdir()
            (root / "scripts" / "setup_pod.sh").write_text("#!/bin/bash\necho setup-ok\n")
            (root / "scripts" / "start_vllm.sh").write_text("#!/bin/bash\necho boom >&2\nexit 3\n")
            for f in (root / "scripts").iterdir():
                f.chmod(0o755)
            log_p, exit_p = root / "vllm.log", root / "vllm.exit"
            cmd = bench_sdd.vllm_launch_cmd(str(root), "configs/_composed.yaml",
                                            log_path=str(log_p), exit_path=str(exit_p))
            subprocess.run(["bash", "-c", cmd], timeout=10)
            for _ in range(50):                       # the chain is backgrounded
                if exit_p.exists():
                    break
                time.sleep(0.1)
            self.assertEqual(exit_p.read_text().strip(), "3")
            logged = log_p.read_text()
            self.assertIn("setup-ok", logged)
            self.assertIn("boom", logged)

    def test_launch_cmd_carries_hf_token_to_start_vllm(self):
        # .env holds RUNPOD_API_KEY (account-takeover-grade) and is NOT rsync'd to the pod;
        # HF_TOKEN is exported inline instead. Run the real chain with a stub start_vllm
        # that records $HF_TOKEN — it must actually reach the script, not just appear.
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "scripts").mkdir()
            (root / "scripts" / "setup_pod.sh").write_text("#!/bin/bash\n")
            seen = root / "seen.txt"
            (root / "scripts" / "start_vllm.sh").write_text(
                f'#!/bin/bash\necho "$HF_TOKEN" > {seen}\n')
            for f in (root / "scripts").iterdir():
                f.chmod(0o755)
            cmd = bench_sdd.vllm_launch_cmd(str(root), "cfg.yaml", hf_token="hf_ABC123",
                                            log_path=str(root / "l"), exit_path=str(root / "e"))
            subprocess.run(["bash", "-c", cmd], timeout=10)
            for _ in range(50):
                if seen.exists() and seen.read_text().strip():
                    break
                time.sleep(0.1)
            self.assertEqual(seen.read_text().strip(), "hf_ABC123")

    def test_launch_cmd_omits_hf_token_when_absent(self):
        cmd = bench_sdd.vllm_launch_cmd("/repo", "cfg.yaml")
        self.assertNotIn("HF_TOKEN", cmd)


def _test_guard(tmpdir):
    return make_guard([], state_path=os.path.join(tmpdir, "pods.json"))


class StartupPollIterationsTest(unittest.TestCase):
    """The health-poll backstop must outlast MAX_STARTUP so the guard's max_phase governs
    (not a fixed 30-min loop). A large model like GLM downloads 744GB and legitimately
    needs a raised MAX_STARTUP; the loop must honor it."""

    def test_iterations_cover_max_startup_with_margin(self):
        # default MAX_STARTUP 720s / 5s poll = 144, + margin
        self.assertGreaterEqual(bench_sdd.startup_poll_iterations(720, poll=5), 144 + 6)

    def test_scales_with_a_raised_max_startup(self):
        # a 60-min ceiling must give ~720 polls, not stay at the old fixed 360
        self.assertGreaterEqual(bench_sdd.startup_poll_iterations(3600, poll=5), 720)


class SpawnPrefixedTest(unittest.TestCase):
    """N agents share the console during generation; every line must carry its agent tag
    or the interleaving reads as turn-taking (a real user doubt)."""

    def test_lines_carry_the_agent_prefix(self):
        lines = []
        p = bench_sdd._spawn_prefixed(["bash", "-c", "echo hola; echo dos"], "agent7",
                                      sink=lines.append)
        p.wait()
        p.pump_thread.join(timeout=5)
        self.assertEqual(lines, ["[agent7] hola\n", "[agent7] dos\n"])

    def test_stderr_is_prefixed_too(self):
        lines = []
        p = bench_sdd._spawn_prefixed(["bash", "-c", "echo err >&2"], "agent0",
                                      sink=lines.append)
        p.wait()
        p.pump_thread.join(timeout=5)
        self.assertEqual(lines, ["[agent0] err\n"])

    def test_wait_procs_still_works_on_prefixed_procs(self):
        with tempfile.TemporaryDirectory() as d:
            procs = [bench_sdd._spawn_prefixed(["true"], f"agent{i}", sink=lambda s: None)
                     for i in range(2)]
            bench_sdd._wait_procs(procs, _test_guard(d), poll=0.05)
            for p in procs:
                self.assertEqual(p.poll(), 0)


class WaitProcsTest(unittest.TestCase):
    """Agent containers are waited on with a guard poll: if the watchdog aborts (the pod is
    already dead), the survivors are killed and the run stops locally too — instead of
    waiting forever on agents retrying a dead endpoint."""

    def test_returns_when_all_procs_finish(self):
        with tempfile.TemporaryDirectory() as d:
            procs = [subprocess.Popen(["true"]) for _ in range(2)]
            bench_sdd._wait_procs(procs, _test_guard(d), poll=0.05)
            for p in procs:
                self.assertEqual(p.poll(), 0)

    def test_guard_abort_kills_survivors_and_raises(self):
        with tempfile.TemporaryDirectory() as d:
            guard = _test_guard(d)
            p = subprocess.Popen(["sleep", "30"])
            guard.aborted = "stall"
            t0 = time.time()
            with self.assertRaises(pod_guard.PodGuardAborted):
                bench_sdd._wait_procs([p], guard, poll=0.05)
            self.assertLess(time.time() - t0, 10)   # did not wait out the 30s sleep
            self.assertIsNotNone(p.poll())          # the survivor was killed

    def test_abort_raises_even_when_all_procs_already_exited(self):
        # Regression: if the watchdog kills the pod and every agent dies instantly, no
        # wait() ever times out — the abort must still surface, or the truncated last
        # level gets scored and reported as a success.
        with tempfile.TemporaryDirectory() as d:
            guard = _test_guard(d)
            p = subprocess.Popen(["true"])
            p.wait()                                 # already finished before entering
            guard.aborted = "stall"
            with self.assertRaises(pod_guard.PodGuardAborted):
                bench_sdd._wait_procs([p], guard, poll=0.05)

    def test_abort_force_removes_named_containers(self):
        # SIGKILL to the docker CLI does NOT stop the container; only `docker rm -f`
        # does. On abort the named survivors must be force-removed.
        with tempfile.TemporaryDirectory() as d:
            guard = _test_guard(d)
            p = subprocess.Popen(["sleep", "30"])
            guard.aborted = "stall"
            removed = []
            with self.assertRaises(pod_guard.PodGuardAborted):
                bench_sdd._wait_procs([p], guard, poll=0.05,
                                      names=["dail-agent-x-n1-0"], rm=removed.extend)
            self.assertEqual(removed, ["dail-agent-x-n1-0"])


class UrlOkTest(unittest.TestCase):
    """Health/metrics polls must never hang the loop on a half-open tunnel."""

    def test_true_for_reachable_url(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("ok")
        self.assertTrue(bench_sdd._url_ok(f"file://{f.name}"))

    def test_false_for_closed_port(self):
        self.assertFalse(bench_sdd._url_ok("http://127.0.0.1:1/health", timeout=2))

    def test_false_and_prompt_on_silent_server(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            t0 = time.time()
            self.assertFalse(bench_sdd._url_ok(f"http://127.0.0.1:{port}/", timeout=1))
            # margin above the op's own 6s ceiling (timeout+5) so CI runners don't flake
            self.assertLess(time.time() - t0, 10)
        finally:
            srv.close()


class MetricsSnapshotTest(unittest.TestCase):
    def test_empty_on_unreachable(self):
        self.assertEqual(bench_sdd._metrics_snapshot(url="http://127.0.0.1:1/metrics",
                                                     timeout=2), "")

    def test_empty_and_prompt_on_silent_server(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            t0 = time.time()
            self.assertEqual(bench_sdd._metrics_snapshot(
                url=f"http://127.0.0.1:{port}/metrics", timeout=1), "")
            # margin above the op's own 6s ceiling (timeout+5) so CI runners don't flake
            self.assertLess(time.time() - t0, 10)
        finally:
            srv.close()


class RunWithTimeoutTest(unittest.TestCase):
    """Gate containers get a hard timeout: a hung gate must not stall the run while the
    pod bills. On expiry the caller treats missing output as a failed gate."""

    def test_true_on_completion(self):
        calls = []
        ok = bench_sdd._run_with_timeout(["cmd"], 5, "gates x",
                                         runner=lambda cmd, timeout: calls.append((cmd, timeout)))
        self.assertTrue(ok)
        self.assertEqual(calls, [(["cmd"], 5)])

    def test_false_on_timeout(self):
        def runner(cmd, timeout):
            raise subprocess.TimeoutExpired(cmd, timeout)
        self.assertFalse(bench_sdd._run_with_timeout(["cmd"], 5, "gates x", runner=runner))

    def test_cleanup_runs_on_timeout_only(self):
        # subprocess timeout SIGKILLs only the docker CLI; the container needs an
        # explicit `docker rm -f` — the cleanup hook is where that happens.
        cleaned = []
        def runner(cmd, timeout):
            raise subprocess.TimeoutExpired(cmd, timeout)
        bench_sdd._run_with_timeout(["cmd"], 5, "gates x", runner=runner,
                                    cleanup=lambda: cleaned.append(1))
        self.assertEqual(cleaned, [1])
        bench_sdd._run_with_timeout(["cmd"], 5, "gates x",
                                    runner=lambda cmd, timeout: None,
                                    cleanup=lambda: cleaned.append(2))
        self.assertEqual(cleaned, [1])               # no cleanup on the happy path


class ContainerNameTest(unittest.TestCase):
    def test_names_embed_the_run_id(self):
        # Container names were run-independent (dail-agent-n4-0): an orphan left by a
        # timeout/kill collided with the next run's identical name → environmental failure.
        a = bench_sdd._container_name("agent", "20260701-130150", 4, 0)
        b = bench_sdd._container_name("agent", "20260702-090000", 4, 0)
        self.assertNotEqual(a, b)
        self.assertIn("20260701-130150", a)
        self.assertIn("agent", a)
        self.assertIn("n4", a)

    def test_empty_run_id_still_yields_valid_name(self):
        name = bench_sdd._container_name("gates", "", 8, "agent3")
        self.assertNotIn("--", name)                 # docker names: no stray double dashes


class ScanPodLogTest(unittest.TestCase):
    """The ssh log-scan must never take the whole run down: a stalled ssh exec during a
    healthy model load raises TimeoutExpired, which must read as 'nothing found this poll'."""

    def test_returns_stdout_on_success(self):
        class Out:
            stdout = "Traceback ..."
        got = bench_sdd._scan_pod_log("1.2.3.4", 22, "/k",
                                      runner=lambda cmd, **kw: Out())
        self.assertEqual(got, "Traceback ...")

    def test_empty_on_ssh_timeout(self):
        def runner(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
        self.assertEqual(bench_sdd._scan_pod_log("1.2.3.4", 22, "/k", runner=runner), "")


class RequireToolchainImageTest(unittest.TestCase):
    def test_returns_image_when_present(self):
        img = bench_sdd.require_toolchain_image("benchmarks/smoke/scenario.yaml",
                                                exists=lambda _: True)
        self.assertEqual(img, "dail-toolchain")

    def test_raises_with_build_hint_when_missing(self):
        with self.assertRaises(SystemExit) as cm:
            bench_sdd.require_toolchain_image("benchmarks/smoke/scenario.yaml",
                                              exists=lambda _: False)
        msg = str(cm.exception)
        self.assertIn("dail-toolchain", msg)
        self.assertIn("docker build", msg)
        self.assertIn("infra/toolchain/Dockerfile", msg)

    def test_overlay_hint_points_at_scenario_dockerfile(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "myscenario"
            sd.mkdir()
            (sd / "Dockerfile").write_text("FROM x")
            with self.assertRaises(SystemExit) as cm:
                bench_sdd.require_toolchain_image(str(sd / "scenario.yaml"), exists=lambda _: False)
            msg = str(cm.exception)
            self.assertIn("dail-toolchain-myscenario", msg)
            self.assertIn("Dockerfile", msg)

