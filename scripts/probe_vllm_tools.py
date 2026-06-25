"""Diagnostic: probe vLLM's tool-call parsing directly, bypassing LiteLLM + Claude Code.

Sends the SAME chat/completions request twice — non-streaming and streaming — with one tool
and a prompt that should trigger it. Records, for each, whether vLLM populated structured
tool_calls or left the call in message content. This isolates the vLLM tool parser (and any
streaming-vs-non-streaming difference) from the rest of the pipeline.

Usage: python -m scripts.probe_vllm_tools <base_url> <served_model_name> [out.json]
       base_url like http://localhost:8000/v1
"""
import json
import sys
import urllib.request

TOOL = {
    "type": "function",
    "function": {
        "name": "create_file",
        "description": "Create a file with the given content.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
}
MESSAGES = [
    {"role": "user", "content": "Create a file hello.txt containing PONG. Use the create_file tool."},
]


def _post(base_url, model, stream):
    body = {
        "model": model,
        "messages": MESSAGES,
        "tools": [TOOL],
        "tool_choice": "auto",
        "max_tokens": 256,
        "temperature": 0,
        "stream": stream,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=120)


def probe_non_streaming(base_url, model):
    with _post(base_url, model, stream=False) as r:
        data = json.loads(r.read())
    msg = data["choices"][0]["message"]
    return {
        "finish_reason": data["choices"][0].get("finish_reason"),
        "tool_calls": msg.get("tool_calls"),
        "tool_calls_count": len(msg.get("tool_calls") or []),
        "content": msg.get("content"),
    }


def probe_streaming(base_url, model):
    content, tool_call_deltas = "", 0
    tool_args, tool_names = "", []
    finish_reason = None
    with _post(base_url, model, stream=True) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            choices = chunk.get("choices") or []  # final usage-only chunk has no choices
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            if delta.get("content"):
                content += delta["content"]
            for tc in delta.get("tool_calls") or []:
                tool_call_deltas += 1
                fn = (tc.get("function") or {})
                tool_args += fn.get("arguments") or ""
                if fn.get("name"):
                    tool_names.append(fn["name"])
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    return {
        "finish_reason": finish_reason,
        "tool_call_deltas": tool_call_deltas,
        "tool_names": tool_names,
        "tool_args": tool_args,
        "content": content,
    }


def run(base_url, model):
    result = {"base_url": base_url, "model": model}
    for label, fn in (("non_streaming", probe_non_streaming), ("streaming", probe_streaming)):
        try:
            result[label] = fn(base_url, model)
        except Exception as e:  # keep going so we always capture both legs
            result[label] = {"error": f"{type(e).__name__}: {e}"}
    return result


if __name__ == "__main__":
    out = run(sys.argv[1], sys.argv[2])
    text = json.dumps(out, indent=2)
    print(text)
    if len(sys.argv) > 3:
        open(sys.argv[3], "w").write(text)
