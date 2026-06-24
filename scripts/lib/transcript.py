"""Parse a Claude Code ``--output-format stream-json`` transcript into metrics.

Pure function over transcript lines: no network, no Claude Code invocation, so it
is unit-testable offline without a GPU or a pod.
"""

import json


def parse_metrics(lines):
    """Return per-run metrics from stream-json transcript lines.

    Keys: prompt_tokens, completion_tokens, total_tokens, tool_calls (dict
    name->count), turns (assistant turns), result_text.
    """
    prompt = completion = turns = 0
    tool_calls = {}
    result_text = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        etype = event.get("type")
        if etype == "assistant":
            turns += 1
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    name = block.get("name", "unknown")
                    tool_calls[name] = tool_calls.get(name, 0) + 1
        elif etype == "result":
            usage = event.get("usage", {})
            prompt = usage.get("input_tokens", 0)
            completion = usage.get("output_tokens", 0)
            result_text = event.get("result", "")

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "tool_calls": tool_calls,
        "turns": turns,
        "result_text": result_text,
    }
