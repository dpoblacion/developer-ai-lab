"""The destream proxy is the load-bearing workaround for vLLM's buggy qwen3 streaming
tool-call parser — a regression here breaks every agent silently (tool calls leak into
content). Tests cover the pure chunk re-emission and the proxy end-to-end against a fake
upstream, stdlib-only like the proxy itself."""
import importlib
import json
import os
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The proxy must be importable without DESTREAM_UPSTREAM set (only *serving* needs it):
# an import-time os.environ[...] would make it untestable and crash any importer.
os.environ.pop("DESTREAM_UPSTREAM", None)
import scripts.destream_proxy as destream_proxy
importlib.reload(destream_proxy)

COMPLETION = {
    "id": "chatcmpl-1", "created": 7, "model": "m", "object": "chat.completion",
    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "let me write that file",
        "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "Write", "arguments": "{\"path\": \"hello.txt\"}"}}]}}],
}


class ChunksFromCompletionTest(unittest.TestCase):
    def test_content_and_tool_calls_become_deltas_in_order(self):
        chunks = destream_proxy._chunks_from_completion(COMPLETION)
        self.assertTrue(all(c["object"] == "chat.completion.chunk" for c in chunks))
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": "let me write that file"})
        tc = chunks[2]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual((tc["index"], tc["id"], tc["function"]["name"]),
                         (0, "call_1", "Write"))
        self.assertEqual(tc["function"]["arguments"], "{\"path\": \"hello.txt\"}")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_content_only_completion(self):
        resp = {"id": "x", "choices": [{"message": {"role": "assistant", "content": "hi"},
                                        "finish_reason": "stop"}]}
        chunks = destream_proxy._chunks_from_completion(resp)
        deltas = [c["choices"][0]["delta"] for c in chunks]
        self.assertEqual(deltas, [{"role": "assistant"}, {"content": "hi"}, {}])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_empty_completion_still_yields_role_and_finish(self):
        chunks = destream_proxy._chunks_from_completion({})
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")


class _FakeUpstream(BaseHTTPRequestHandler):
    """Canned vLLM: records the body it got, always answers COMPLETION."""
    received = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _FakeUpstream.received.append(json.loads(raw))
        data = json.dumps(COMPLETION).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)


class ProxyEndToEndTest(unittest.TestCase):
    """POST a streaming chat/completions through the real Handler: the upstream call must be
    forced non-streaming, and the reply must come back as SSE chunks with tool calls intact."""

    @classmethod
    def setUpClass(cls):
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
        threading.Thread(target=cls.upstream.serve_forever, daemon=True).start()
        destream_proxy.UPSTREAM = f"http://127.0.0.1:{cls.upstream.server_address[1]}"
        cls.proxy = ThreadingHTTPServer(("127.0.0.1", 0), destream_proxy.Handler)
        threading.Thread(target=cls.proxy.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.proxy.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.proxy.shutdown()
        cls.upstream.shutdown()

    def _post(self, body):
        req = urllib.request.Request(f"{self.base}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=10)

    def test_streaming_request_destreams_upstream_and_reemits_sse(self):
        _FakeUpstream.received.clear()
        with self._post({"stream": True, "stream_options": {"include_usage": True},
                         "messages": [{"role": "user", "content": "go"}]}) as r:
            self.assertEqual(r.headers["Content-Type"], "text/event-stream")
            payload = r.read().decode()
        sent = _FakeUpstream.received[0]
        self.assertIs(sent["stream"], False)              # upstream forced non-streaming
        self.assertNotIn("stream_options", sent)
        self.assertTrue(payload.rstrip().endswith("data: [DONE]"))
        chunks = [json.loads(line[6:]) for line in payload.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        tool_deltas = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
        self.assertEqual(len(tool_deltas), 1)
        self.assertEqual(tool_deltas[0]["choices"][0]["delta"]["tool_calls"][0]
                         ["function"]["name"], "Write")

    def test_non_streaming_request_passes_through_as_json(self):
        with self._post({"stream": False, "messages": []}) as r:
            body = json.loads(r.read())
        self.assertEqual(body["choices"][0]["message"]["tool_calls"][0]
                         ["function"]["name"], "Write")
