from __future__ import annotations

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import patch

from app.ai.cloud_client import CloudConfigurationError, OpenAICompatibleClient


class CloudClientTests(unittest.TestCase):
    def test_unconfigured_cloud_refuses_to_send(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            client = OpenAICompatibleClient()
            self.assertFalse(client.configured)
            with self.assertRaises(CloudConfigurationError):
                list(client.stream_chat([], threading.Event(), 12))

    def test_configured_client_streams_from_compatible_mock(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.server.path_seen = self.path
                self.server.authorization = self.headers.get("Authorization")
                self.server.request_body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                body = (b'data: {"choices":[{"delta":{"content":"cloud"}}]}\n\n'
                        b'data: [DONE]\n\n')
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {
                "FRIDAY_CLOUD_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                "FRIDAY_CLOUD_MODEL": "test-model",
                "FRIDAY_CLOUD_API_KEY": "test-only-secret",
            }, clear=True):
                client = OpenAICompatibleClient()
                text = "".join(client.stream_chat([{"role": "user", "content": "hello"}],
                                                  threading.Event(), 24))
            self.assertEqual(text, "cloud")
            self.assertEqual(server.path_seen, "/v1/chat/completions")
            self.assertEqual(server.authorization, "Bearer test-only-secret")
            self.assertEqual(server.request_body["max_tokens"], 24)
            self.assertTrue(server.request_body["stream"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_external_http_endpoint_is_rejected_before_sending(self) -> None:
        with patch.dict(os.environ, {
            "FRIDAY_CLOUD_BASE_URL": "http://api.example.invalid/v1",
            "FRIDAY_CLOUD_MODEL": "test-model",
            "FRIDAY_CLOUD_API_KEY": "test-only-secret",
        }, clear=True):
            client = OpenAICompatibleClient()
            with self.assertRaisesRegex(CloudConfigurationError, "HTTPS"):
                list(client.stream_chat([], threading.Event(), 12))

    def test_provider_error_body_cannot_echo_api_key_into_exception(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = b'{"error":"rejected test-only-secret"}'
                self.send_response(401)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {
                "FRIDAY_CLOUD_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                "FRIDAY_CLOUD_MODEL": "test-model",
                "FRIDAY_CLOUD_API_KEY": "test-only-secret",
            }, clear=True):
                client = OpenAICompatibleClient()
                with self.assertRaises(RuntimeError) as caught:
                    list(client.stream_chat([], threading.Event(), 12))
            self.assertIn("401", str(caught.exception))
            self.assertNotIn("test-only-secret", str(caught.exception))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
