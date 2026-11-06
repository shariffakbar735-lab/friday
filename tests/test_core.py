from __future__ import annotations

import json
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from app.ai.ollama_client import OllamaClient, OllamaError
from app.memory.storage import ConversationStore


class CoreTests(unittest.TestCase):
    def test_conversation_persistence_context_limit_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "friday.sqlite3"
            store = ConversationStore(db_path)
            cid = store.create_conversation()
            for index in range(6):
                store.add_message(cid, "user", f"question {index}")
                store.add_message(cid, "assistant", f"answer {index}")
            self.assertEqual(len(store.get_messages(cid, limit=4)), 4)
            self.assertEqual(store.get_messages(cid, limit=4)[-1]["content"], "answer 5")
            self.assertIn("question 0", store.list_conversations()[0]["title"])
            reopened = ConversationStore(db_path)
            self.assertEqual(len(reopened.get_messages(cid)), 12)
            reopened.delete_conversation(cid)
            self.assertEqual(reopened.list_conversations(), [])

    def test_invalid_role_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = ConversationStore(Path(temp) / "friday.sqlite3")
            cid = store.create_conversation()
            with self.assertRaises(ValueError):
                store.add_message(cid, "system", "not allowed")

    def test_ollama_discovers_models_and_streams(self) -> None:
        chunks = [
            {"message": {"content": "Hello"}, "done": False},
            {"message": {"content": " FRIDAY"}, "done": True},
        ]
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"models": [{"name": "qwen3:4b", "size": 2048}]}).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)
            def do_POST(self):
                request_body = self.rfile.read(int(self.headers["Content-Length"]))
                if not hasattr(self.server, "request_bodies"):
                    self.server.request_bodies = []
                self.server.request_bodies.append(request_body)
                body = b"".join(json.dumps(item).encode() + b"\n" for item in chunks)
                self.send_response(200); self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)
            def log_message(self, *_args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            client = OllamaClient(f"http://127.0.0.1:{server.server_port}")
            models = client.list_models()
            self.assertEqual([m.name for m in models], ["qwen3:4b"])
            self.assertEqual(models[0].size_bytes, 2048)
            self.assertEqual("".join(client.stream_chat(
                "qwen3:4b", [{"role":"user","content":"Hi"}], options={"num_gpu": 0})), "Hello FRIDAY")
            self.assertEqual("".join(client.stream_chat(
                "qwen3:4b", [{"role":"user","content":"Hi again"}])), "Hello FRIDAY")
            self.assertEqual("".join(client.stream_chat(
                "llama3:8b", [{"role":"user","content":"Hi without reasoning"}])), "Hello FRIDAY")
            sent = json.loads(server.request_bodies[0])
            self.assertEqual(sent["model"], "qwen3:4b")
            self.assertTrue(sent["stream"])
            self.assertEqual(sent["options"]["num_gpu"], 0)
            self.assertFalse(sent["think"])
            ordinary = json.loads(server.request_bodies[1])
            self.assertNotIn("num_gpu", ordinary.get("options", {}))
            self.assertFalse(ordinary["think"])
            self.assertNotIn("think", json.loads(server.request_bodies[2]))
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)

    def test_unavailable_ollama_is_explained(self) -> None:
        with self.assertRaises(OllamaError):
            OllamaClient("http://127.0.0.1:1", timeout=0.2).list_models()

if __name__ == "__main__":
    unittest.main()
