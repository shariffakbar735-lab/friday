from __future__ import annotations

import json
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import app.ai.ollama_client as ollama_module
from app.ai.ollama_client import OllamaClient, OllamaError


class OllamaReasoningTests(unittest.TestCase):
    def run_events(self, events: list[dict], *, model: str = "test:latest"):
        body = b"".join(json.dumps(event).encode("utf-8") + b"\n" for event in events)
        return self.run_body(body, model=model)[0]

    def run_body(self, body: bytes, *, model: str = "test:latest", thinking=None):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.server.request_bodies.append(request_body)
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        server.request_bodies = []
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = OllamaClient(f"http://127.0.0.1:{server.server_port}", timeout=1)
            result = "".join(client.stream_chat(
                model, [{"role": "user", "content": "private prompt sentinel"}],
                thinking_callback=thinking,
            ))
            return result, json.loads(server.request_bodies[0])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_structured_thinking_and_nonfinal_channels_never_yield(self):
        events = [
            {"message": {"role": "assistant", "thinking": "First thought: private."}, "done": False},
            {"message": {"role": "assistant", "content": "The user just asked this", "channel": "think"}, "done": False},
            {"message": {"role": "assistant", "content": "Final answer", "channel": "final"}, "done": False},
            {"message": {"role": "assistant", "content": ".", "channel": "final"}, "done": True},
        ]
        body = b"".join(json.dumps(event).encode("utf-8") + b"\n" for event in events)
        result, request = self.run_body(body, model="qwen3:4b")
        self.assertFalse(request["think"])
        self.assertEqual(result, "Final answer.")
        self.assertNotIn("First thought", result)
        self.assertNotIn("The user just asked", result)

    def test_split_inline_think_tags_are_suppressed_but_final_text_streams(self):
        events = [
            {"message": {"content": "<thi"}, "done": False},
            {"message": {"content": "nk>First thought: private."}, "done": False},
            {"message": {"content": "</thi"}, "done": False},
            {"message": {"content": "nk>Answer."}, "done": True},
        ]
        self.assertEqual(self.run_events(events), "Answer.")

    def test_structured_thinking_is_delivered_separately(self):
        thinking = []
        events = [
            {"message": {"thinking": "private thought 1"}, "done": False},
            {"message": {"thinking": " and thought 2"}, "done": False},
            {"message": {"content": "Final answer."}, "done": True},
        ]
        result, _request = self.run_body(
            b"".join(json.dumps(event).encode("utf-8") + b"\n" for event in events),
            model="qwen3:4b", thinking=thinking.append,
        )
        self.assertEqual(result, "Final answer.")
        self.assertEqual("".join(thinking), "private thought 1 and thought 2")
        self.assertNotIn("private thought", result)

    def test_reasoning_chunks_before_late_close_never_reach_client_output(self):
        events = [
            {"message": {"content": "First thought: The user just asked"}, "done": False},
            {"message": {"content": " how are you?"}, "done": False},
            {"message": {"content": "</think>"}, "done": False},
            {"message": {"content": "I am well."}, "done": True},
        ]
        result = self.run_events(events)
        self.assertEqual(result, "I am well.")
        self.assertNotIn("First thought", result)
        self.assertNotIn("The user just asked", result)
        self.assertNotIn("</think>", result)

    def test_unclosed_inline_reasoning_is_discarded_on_successful_completion(self):
        events = [
            {"message": {"content": "<think>First thought: private and unclosed"}, "done": False},
            {"message": {"content": "More private thoughts"}, "done": True},
        ]
        self.assertEqual(self.run_events(events), "")

    def test_unmarked_meta_analysis_before_late_close_never_yields(self):
        events = [
            {"message": {"content": "Okay, the user just said hi twice in a row."}, "done": False},
            {"message": {"content": "Interesting pattern here."}, "done": False},
            {"message": {"content": "</think>"}, "done": False},
            {"message": {"content": "Hi!"}, "done": True},
        ]
        self.assertEqual(self.run_events(events), "Hi!")

    def test_ordinary_answer_with_meta_like_opening_is_preserved(self):
        events = [
            {"message": {"content": "Okay, here's the answer."}, "done": True},
        ]
        self.assertEqual(self.run_events(events), "Okay, here's the answer.")

    def test_legitimate_thought_wording_without_meta_subject_is_preserved(self):
        events = [
            {"message": {"content": "First thought: the answer is 42."}, "done": True},
        ]
        self.assertEqual(self.run_events(events), "First thought: the answer is 42.")

    def test_terminal_metadata_event_without_message_is_valid(self):
        events = [
            {"message": {"content": "Answer"}, "done": False},
            {"done": True, "done_reason": "stop"},
        ]
        self.assertEqual(self.run_events(events), "Answer")

    def test_plain_final_content_without_markers_is_preserved(self):
        events = [
            {"message": {"role": "assistant", "content": "Normal final answer."}, "done": True},
        ]
        result, request = self.run_body(
            b"".join(json.dumps(event).encode("utf-8") + b"\n" for event in events),
            model="llama3:8b",
        )
        self.assertEqual(result, "Normal final answer.")
        self.assertNotIn("think", request)

    def test_stream_timing_diagnostics_do_not_include_response_content(self):
        events = [
            {"message": {"content": "timing-private-answer"}, "done": True},
        ]
        body = b"".join(json.dumps(event).encode("utf-8") + b"\n" for event in events)
        with self.assertLogs(ollama_module.__name__, level="INFO") as logs:
            self.assertEqual(self.run_body(body)[0], "timing-private-answer")
        diagnostic = " ".join(logs.output)
        self.assertIn("category=first_visible", diagnostic)
        self.assertIn("category=completed", diagnostic)
        self.assertIn("elapsed_ms=", diagnostic)
        self.assertNotIn("timing-private-answer", diagnostic)

    def test_malformed_event_diagnostic_does_not_echo_event_or_prompt(self):
        body = b'{"message":[],"private":"reasoning sentinel","done":true}\n'
        with self.assertLogs(ollama_module.__name__, level="WARNING") as logs:
            with self.assertRaisesRegex(OllamaError, "invalid streaming response") as raised:
                self.run_body(body)
        diagnostic = " ".join(logs.output)
        self.assertIn("unexpected_event_shape", diagnostic)
        self.assertNotIn("reasoning sentinel", diagnostic)
        self.assertNotIn("private prompt sentinel", diagnostic)
        self.assertNotIn("reasoning sentinel", str(raised.exception))

    def test_server_error_event_does_not_echo_private_content(self):
        body = b'{"error":"private prompt and reasoning sentinel"}\n'
        with self.assertLogs(ollama_module.__name__, level="WARNING") as logs:
            with self.assertRaises(OllamaError) as raised:
                self.run_body(body)
        self.assertNotIn("sentinel", " ".join(logs.output))
        self.assertNotIn("sentinel", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
