"""Concurrency sweep with SLO.

Fires N concurrent *streaming* requests directly at vLLM, measures per-stream TTFT and
decode throughput, and finds the highest concurrency that holds the SLO (the knee) — the
capacity of this hardware for the model. Developer counts and cost are derived later,
outside this harness, from the captured output.

Run on the pod (needs vLLM serving the model). Pure logic lives in scripts/lib/* and is
unit-tested offline.
"""

import asyncio
import json
import os
import pathlib
import time

import aiohttp

from scripts.lib.stream import compute_stream_metrics
from scripts.lib.slo import summarize_level, evaluate_slo, find_knee
from scripts.lib.hardware import gather, write_env

MODEL = os.getenv("MODEL", "glm-5.2-fp8")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")  # vLLM directly (no proxy)
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
CONCURRENCY = [int(v) for v in os.getenv("CONCURRENCY", "1,2,4,8,16,32").split(",") if v.strip()]
SLO = {
    "max_ttft": float(os.getenv("SLO_MAX_TTFT", "2.0")),
    "min_tps": float(os.getenv("SLO_MIN_TPS", "20.0")),
}
PROMPT = os.getenv(
    "PROMPT",
    "Write a complete C# ASP.NET Core Minimal API for a Todo application with EF Core.")


async def one_stream(session, idx):
    start = time.time()
    first_token = None
    last_token = start
    content_chunks = 0
    usage_completion = None

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with session.post(f"{BASE_URL}/v1/chat/completions", json=payload) as resp:
        async for raw in resp.content:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices", [])
            if choices and choices[0].get("delta", {}).get("content"):
                now = time.time()
                if first_token is None:
                    first_token = now
                last_token = now
                content_chunks += 1
            if chunk.get("usage"):
                usage_completion = chunk["usage"].get("completion_tokens", usage_completion)

    if first_token is None:
        first_token = last_token = time.time()
    # Prefer server-reported usage; fall back to counting content chunks so a backend that
    # omits the usage chunk doesn't silently report 0 tokens (which would fake a knee of 0).
    usage_present = usage_completion is not None
    tokens = usage_completion if usage_present else content_chunks
    metrics = compute_stream_metrics(start, first_token, last_token, tokens)
    metrics["idx"] = idx
    metrics["usage_present"] = usage_present
    return metrics


async def run_level(concurrency):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        samples = await asyncio.gather(
            *(one_stream(session, i) for i in range(concurrency)))
    summary = summarize_level(concurrency, samples)
    summary["slo_pass"] = evaluate_slo(summary, SLO)
    summary["samples"] = samples
    return summary


async def main():
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = pathlib.Path("results") / run_id / "concurrency-slo"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_doc = gather()
    write_env(out_dir, env_doc)

    summaries = []
    for concurrency in CONCURRENCY:
        print(f"=== concurrency {concurrency} ===")
        summary = await run_level(concurrency)
        (out_dir / f"concurrency-{concurrency}.json").write_text(json.dumps(summary, indent=2))
        print(f"  ttft_median={summary['ttft_median']:.2f}s "
              f"tps_median={summary['tps_median']:.1f} pass={summary['slo_pass']}")
        summaries.append(summary)

    knee = find_knee(summaries, SLO)

    result = {
        "run_id": run_id,
        "model": MODEL,
        "slo": SLO,
        "env": env_doc,
        "levels": [{k: v for k, v in s.items() if k != "samples"} for s in summaries],
        "knee_streams": knee,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(f"Knee (max concurrent streams at SLO): {knee}")


if __name__ == "__main__":
    asyncio.run(main())
