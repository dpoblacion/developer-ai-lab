# Security

## Reporting

Please report security issues **privately** via
[GitHub Security Advisories](https://github.com/dpoblacion/developer-ai-lab/security/advisories/new)
rather than opening a public issue.

## Secrets model

- All credentials (`RUNPOD_API_KEY`, `SSH_KEY_PATH`, optional `HF_TOKEN`) live in a local
  `.env` file that is **gitignored** — see `.env.example`. Never commit it.
- **`.env` is never copied to the pod.** The rsync excludes it, so `RUNPOD_API_KEY` (which
  can create/destroy pods on your account) never lands on the third-party ephemeral machine.
  Only `HF_TOKEN` is passed to the pod, inline in the launch env, so weight downloads still
  authenticate.
- **vLLM is not published publicly.** The pod exposes only SSH (`22/tcp`); the benchmark
  reaches vLLM over an SSH tunnel. The model is never served on RunPod's public proxy, so it
  is not usable — or able to skew your latency measurements — by anyone who learns the pod URL.
- The `master_key: sk-dev-lab` in `infra/litellm/config.yaml` is **not a secret**: it only
  gates a loopback proxy inside an ephemeral benchmark container that is never exposed
  publicly, which is why it is committed.
- Pods are ephemeral and SSH host-key checking is deliberately disabled for them
  (`scripts/lib/runpod_pod.py`) — do not copy that pattern into contexts where MITM
  protection matters. With `RUNPOD_API_KEY` no longer on the pod, the residual exposure of
  a MITM is limited to the benchmark repo/transcript content, not your account.

## Running untrusted branches

Benchmark gate commands are shell snippets read from `benchmarks/*/scenario.yaml` and the
agent runs with a `Bash` tool. **Do not run `make run` / `make gates` on a branch you
haven't reviewed** — a malicious scenario could run code on your machine. CI never executes
gates or benchmarks (it runs only the offline unit suite + lint), so opening a PR is safe;
the caution applies only to running a branch locally.

## Supply chain

The pod-setup and toolchain-image builds fetch installers over the network
(`deb.nodesource.com`, `claude.ai/install.sh`) and install some unpinned pip packages. This
is standard build-time exposure; pin to digests in a hardened environment if that matters
to you.

## Money safety

This tool creates real, per-second-billed GPU pods. The `PodGuard` watchdog, the orphan
reapers, and `make reap` exist to prevent runaway spend — if you touch pod lifecycle code,
please read the guard tests (`tests/test_pod_guard.py`) first.
