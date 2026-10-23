from __future__ import annotations

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from unittest.mock import patch

import app.ai.ollama_client as ollama_module
from app.ai.ollama_client import OllamaClient, OllamaError


class ReleaseFailureTests(unittest.TestCase):
    def serve(self, handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def stop_server(self, server, thread):
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    def test_request_unavailable_after_startup_is_explained(self):
        with self.assertRaises(OllamaError):
            list(OllamaClient("http://127.0.0.1:1", timeout=0.2).stream_chat(
                "missing:latest", [{"role": "user", "content": "hello"}]))

    def test_malformed_model_catalog_is_a_readable_startup_error(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = b'[]'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *_args):
                pass

        server, thread = self.serve(Handler)
        try:
            with self.assertRaisesRegex(OllamaError, "invalid model catalog"):
                OllamaClient(f"http://127.0.0.1:{server.server_port}").list_models()
        finally:
            self.stop_server(server, thread)

    def test_model_missing_error_event_is_reported(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = b'{"error":"model not found"}\n'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *_args):
                pass

        server, thread = self.serve(Handler)
        try:
            client = OllamaClient(f"http://127.0.0.1:{server.server_port}")
            with self.assertRaisesRegex(OllamaError, "model not found"):
                list(client.stream_chat("missing:latest", [{"role": "user", "content": "hello"}]))
        finally:
            self.stop_server(server, thread)

    def test_malformed_stream_is_reported_and_response_is_closed(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = b'{not-json}\n'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *_args):
                pass

        server, thread = self.serve(Handler)
        try:
            client = OllamaClient(f"http://127.0.0.1:{server.server_port}")
            with self.assertRaisesRegex(OllamaError, "invalid streaming response"):
                list(client.stream_chat("test:latest", [{"role": "user", "content": "hello"}]))
        finally:
            self.stop_server(server, thread)

    def test_read_timeout_is_bounded(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.flush()
                time.sleep(0.35)
                try:
                    self.wfile.write(b'{"message":{"content":"late"},"done":true}\n')
                    self.wfile.flush()
                except OSError:
                    pass
            def log_message(self, *_args):
                pass

        server, thread = self.serve(Handler)
        try:
            client = OllamaClient(f"http://127.0.0.1:{server.server_port}", timeout=0.1)
            started = time.monotonic()
            with self.assertRaisesRegex(OllamaError, "timed out"):
                list(client.stream_chat("test:latest", [{"role": "user", "content": "hello"}]))
            self.assertLess(time.monotonic() - started, 1.0)
        finally:
            self.stop_server(server, thread)

    def test_stop_closes_a_stalled_stream_promptly(self):
        first_chunk_sent = Event()
        first_chunk_received = Event()
        release_server = Event()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.server.request_count = getattr(self.server, "request_count", 0) + 1
                if self.server.request_count > 1:
                    body = b'{"message":{"content":"new request works"},"done":true}\n'
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    first = b'{"message":{"content":"first"},"done":false}\n'
                    self.wfile.write(f"{len(first):X}\r\n".encode() + first + b"\r\n")
                    self.wfile.flush()
                    first_chunk_sent.set()
                    release_server.wait(3)
                    second = b'{"message":{"content":"second"},"done":true}\n'
                    self.wfile.write(f"{len(second):X}\r\n".encode() + second + b"\r\n0\r\n\r\n")
                    self.wfile.flush()
                except OSError:
                    pass
            def log_message(self, *_args):
                pass

        server, thread = self.serve(Handler)
        stop = Event()
        received = []
        errors = []
        opened_responses = []
        real_urlopen = ollama_module.urllib.request.urlopen

        def capture_response(*args, **kwargs):
            response = real_urlopen(*args, **kwargs)
            opened_responses.append(response)
            return response

        def consume():
            try:
                for piece in OllamaClient(f"http://127.0.0.1:{server.server_port}", timeout=5).stream_chat(
                    "test:latest", [{"role": "user", "content": "hello"}], stop=stop):
                    received.append(piece)
                    first_chunk_received.set()
            except Exception as exc:
                errors.append(exc)

        consumer = Thread(target=consume, daemon=True)
        client = OllamaClient(f"http://127.0.0.1:{server.server_port}", timeout=5)
        with patch.object(ollama_module.urllib.request, "urlopen", side_effect=capture_response):
            consumer.start()
            try:
                self.assertTrue(first_chunk_sent.wait(1), "server did not send its first chunk")
                self.assertTrue(first_chunk_received.wait(1), "client did not yield its first chunk")
                stop.set()
                stop.set()  # Stop is idempotent; a repeated click must not block or fail.
                consumer.join(timeout=1)
                self.assertFalse(consumer.is_alive(), "cancellation waited for another network line")
                self.assertEqual(received, ["first"])
                self.assertEqual(errors, [])
                self.assertEqual(len(opened_responses), 1)
                self.assertTrue(opened_responses[0].closed, "cancelled HTTP response was not closed")
                release_server.set()
                self.assertEqual("".join(client.stream_chat(
                    "test:latest", [{"role": "user", "content": "second request"}])),
                    "new request works")
            finally:
                stop.set()
                release_server.set()
                consumer.join(timeout=1)
                self.stop_server(server, thread)


if __name__ == "__main__":
    unittest.main()
