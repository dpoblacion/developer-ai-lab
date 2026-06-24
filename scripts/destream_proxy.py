"""De-streaming proxy between LiteLLM and vLLM.

vLLM's qwen3 streaming tool-call parser is buggy: in streaming mode it leaks the tool call
into message content (tool_calls stays empty), while non-streaming parses correctly. Claude
Code always streams, so it always hits the broken path. This proxy forces the UPSTREAM vLLM
call to be non-streaming and re-emits the result as OpenAI SSE chunks, so LiteLLM (and thus
Claude Code) still get a streamed response — with tool calls intact.

Stdlib-only (runs in the toolchain container). Point LiteLLM's api_base at this proxy and
set DESTREAM_UPSTREAM to the real vLLM base.

Env: DESTREAM_UPSTREAM  real vLLM origin, no /v1 suffix (e.g. http://host.docker.internal:8000)
     DESTREAM_PORT      listen port (default 8011)
"""
import json
import os
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Read lazily-tolerant: only *serving* needs the env (checked in __main__); importers
# (tests, tooling) must not crash on a missing variable.
UPSTREAM = os.environ.get("DESTREAM_UPSTREAM", "").rstrip("/")
PORT = int(os.getenv("DESTREAM_PORT", "8011"))


def _upstream(path, body_bytes, headers, method="POST"):
    req = urllib.request.Request(UPSTREAM + path, data=body_bytes, headers=headers, method=method)
    return urllib.request.urlopen(req, timeout=300)


def _chunks_from_completion(resp):
    """Turn a non-streaming chat.completion into OpenAI chat.completion.chunk deltas."""
    meta = {"id": resp.get("id", "chatcmpl-destream"), "object": "chat.completion.chunk",
            "created": resp.get("created", 0), "model": resp.get("model", "")}
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    out = [{**meta, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}]
    if msg.get("content"):
        out.append({**meta, "choices": [{"index": 0, "delta": {"content": msg["content"]},
                                         "finish_reason": None}]})
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        out.append({**meta, "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": i, "id": tc.get("id"), "type": tc.get("type", "function"),
             "function": {"name": fn.get("name"), "arguments": fn.get("arguments", "")}}]},
            "finish_reason": None}]})
    out.append({**meta, "choices": [{"index": 0, "delta": {},
                "finish_reason": choice.get("finish_reason", "stop")}]})
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _fwd_headers(self):
        h = {"Content-Type": "application/json"}
        if self.headers.get("Authorization"):
            h["Authorization"] = self.headers["Authorization"]
        return h

    def do_GET(self):  # health/models passthrough
        try:
            with _upstream(self.path, None, self._fwd_headers(), method="GET") as r:
                data = r.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:  # any upstream failure -> 502; the proxy must never crash
            self.send_error(502, str(e))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        # Only chat/completions needs de-streaming; anything else is a transparent proxy.
        if not self.path.endswith("/chat/completions"):
            try:
                with _upstream(self.path, raw, self._fwd_headers()) as r:
                    data, status = r.read(), r.status
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:  # any upstream failure -> 502; the proxy must never crash
                self.send_error(502, str(e))
            return

        body = json.loads(raw or b"{}")
        want_stream = bool(body.get("stream"))
        body["stream"] = False
        body.pop("stream_options", None)
        try:
            with _upstream(self.path, json.dumps(body).encode(), self._fwd_headers()) as r:
                completion = json.loads(r.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(e.read())
            return

        if not want_stream:
            data = json.dumps(completion).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in _chunks_from_completion(completion):
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


if __name__ == "__main__":
    if not UPSTREAM:
        raise SystemExit("DESTREAM_UPSTREAM not set (real vLLM origin, no /v1 suffix)")
    print(f"destream proxy on :{PORT} -> {UPSTREAM}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
