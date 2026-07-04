"""Raw-saturation probe: measure Θmax(H, M, Q) — the engine+hardware token ceiling —
right after the benchmark levels, while the pod is still up.

arXiv:2606.11690 §3.2 defines utilization as U = Θachieved/Θmax; without a measured Θmax,
U can only be bounded by the best observed level. The probe saturates vLLM (closed-loop,
high concurrency) with random-word prompts (no shareable prefixes, so the prefix cache
cannot inflate the ceiling) at the workload's own prompt:output ratio — the I/O mix moves
the absolute ceiling (§5.7), so Θmax must be shape-matched to be comparable with the
levels. Θmax comes from the server's own token counters, like every other token count.
"""
import concurrent.futures
import random
import time

from scripts.lib.vllm_metrics import level_tokens

PAPER_PROMPT, PAPER_OUTPUT = 512, 256   # the reference chat shape (arXiv:2606.11690 §4.3)
MIN_PROMPT, MAX_PROMPT = 128, 8192


def probe_shape(prompt_tokens, generation_tokens, output_tokens=PAPER_OUTPUT):
    """Probe I/O shape mirroring the measured workload ratio; the paper's 512:256 when
    there is no traffic to mirror. Prompt length is clamped to sane serving bounds."""
    if prompt_tokens <= 0 or generation_tokens <= 0:
        return {"prompt_tokens": PAPER_PROMPT, "output_tokens": output_tokens}
    ratio = prompt_tokens / generation_tokens
    prompt = int(min(MAX_PROMPT, max(MIN_PROMPT, round(ratio * output_tokens))))
    return {"prompt_tokens": prompt, "output_tokens": output_tokens}


def summarize_probe(before_text, after_text, duration_s, shape, concurrency):
    """Θmax from the /metrics counter diffs across the probe window."""
    tokens = level_tokens(before_text, after_text)
    total = tokens["prompt"] + tokens["generation"]
    return {
        "tokens_per_hour": total / duration_s * 3600,
        "output_tokens_per_hour": tokens["generation"] / duration_s * 3600,
        "server_tokens": tokens,
        "duration_s": duration_s,
        "shape": shape,
        "concurrency": concurrency,
    }


def run_probe(request_fn, snapshot_fn, clock, duration_s, concurrency, shape):
    """Closed-loop saturation: `concurrency` workers re-issue request_fn until the window
    closes; Θmax from snapshot diffs. request_fn/snapshot_fn/clock injected for tests."""
    before = snapshot_fn()
    start = clock()

    def worker():
        while clock() - start < duration_s:
            try:
                request_fn()
            except Exception:
                # A failed request must not kill the probe; the counters only advance on
                # served tokens, so failures simply don't contribute to Θmax.
                time.sleep(0.5)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for _ in range(concurrency):
            pool.submit(worker)
    after = snapshot_fn()
    return summarize_probe(before, after, clock() - start, shape, concurrency)


# ~1.3 tokens/word for English-like random text; used to size the random prompt.
_WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
          "november oscar papa quebec romeo sierra tango uniform victor whiskey xray "
          "yankee zulu").split()


def random_prompt(prompt_tokens, rng=None):
    """Random-word prompt sized to ~prompt_tokens; no two prompts share a prefix, so the
    prefix cache cannot inflate the measured ceiling (paper §4.3's rationale)."""
    rng = rng or random.Random()
    n_words = max(1, int(prompt_tokens / 1.3))
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


def completion_payload(model, shape, rng=None):
    """A /v1/completions body that forces exactly output_tokens (ignore_eos) on a fresh
    random prompt."""
    return {"model": model, "prompt": random_prompt(shape["prompt_tokens"], rng),
            "max_tokens": shape["output_tokens"], "ignore_eos": True, "temperature": 0.0}
