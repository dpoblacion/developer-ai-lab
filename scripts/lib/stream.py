"""Per-stream metric computation for the concurrency sweep.

Pure: given the timestamps a streaming request produced and its completion token
count, compute TTFT, decode throughput, and end-to-end latency. The aiohttp loop that
captures the timestamps lives in the runner; this stays testable offline.
"""


def compute_stream_metrics(start, first_token, last_token, completion_tokens):
    """Metrics for one streamed response.

    - ttft: time to first token (s)
    - decode_tps: tokens generated *after* the first, over the decode window. The first
      token belongs to TTFT, not decode, so the numerator is completion_tokens - 1 to
      match the (last - first) interval. 0 if the window is empty or only one token.
    - latency: end-to-end (s)
    """
    decode_window = last_token - first_token
    decode_tps = 0.0
    if decode_window > 0 and completion_tokens > 1:
        decode_tps = (completion_tokens - 1) / decode_window
    return {
        "ttft": first_token - start,
        "decode_tps": decode_tps,
        "latency": last_token - start,
        "completion_tokens": completion_tokens,
    }
