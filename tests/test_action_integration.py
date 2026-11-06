from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app import main as friday_main
from app.system.actions import ActionService


class MemorySettings:
    def __init__(self, *_args):
        self.values = {"provider": "cloud"}

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value


class ActionUIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_cloud_window(self, localappdata):
        env = {
            "LOCALAPPDATA": str(localappdata),
            "FRIDAY_CLOUD_BASE_URL": "http://127.0.0.1:12345/v1",
            "FRIDAY_CLOUD_MODEL": "test-model",
            "FRIDAY_CLOUD_API_KEY": "test-only-secret",
        }
        with patch.dict(os.environ, env), patch.object(friday_main, "QSettings", MemorySettings), \
             patch.object(friday_main.FridayWindow, "_start_discovery", lambda _self: None):
            return friday_main.FridayWindow()

    def test_project_listing_is_local_transient_and_never_enters_cloud_context(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            project = base / "project"
            project.mkdir()
            (project / "private-file-name.txt").write_text("private file contents", encoding="utf-8")
            window = self.make_cloud_window(base / "localappdata")
            try:
                window.action_service = ActionService(
                    base / "localappdata" / "FRIDAY",
                    root_overrides={"friday_project": project},
                )
                window.input.setText("List files in my FRIDAY project directory")
                with patch.object(window, "_confirm_cloud_use") as confirm, \
                     patch.object(friday_main, "OpenAICompatibleClient") as cloud_client:
                    window._send()
                    worker = window.action_worker
                    self.assertIsNotNone(worker)
                    self.assertTrue(worker.wait(3000))
                self.app.processEvents()
                confirm.assert_not_called()
                cloud_client.return_value.stream_chat.assert_not_called()
                self.assertEqual(window.backend.currentData(), "cloud")
                self.assertEqual(window.store.get_messages(window.conversation_id), [])
                transcript = window.transcript.toPlainText()
                self.assertIn("private-file-name.txt", transcript)
                self.assertNotIn("private file contents", transcript)

                window.input.setText("Hello FRIDAY")
                with patch.dict(os.environ, {
                    "FRIDAY_CLOUD_BASE_URL": "http://127.0.0.1:12345/v1",
                    "FRIDAY_CLOUD_MODEL": "test-model",
                    "FRIDAY_CLOUD_API_KEY": "test-only-secret",
                }), patch.object(window, "_confirm_cloud_use", return_value=True), \
                     patch.object(window, "_start_generation") as start:
                    window._send()
                sent_messages = start.call_args.args[0]
                serialized = " ".join(message["content"] for message in sent_messages)
                self.assertNotIn("private-file-name.txt", serialized)
                self.assertNotIn("List files in my FRIDAY project directory", serialized)
                self.assertNotIn("private file contents", serialized)
                self.assertIn("Hello FRIDAY", serialized)
            finally:
                if window.action_worker is not None:
                    window.action_worker.wait(1000)
                window.close()

    def test_unrecognized_action_intent_is_refused_without_cloud_confirmation_or_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_cloud_window(Path(directory) / "localappdata")
            try:
                window.input.setText(r"Open C:\private\folder")
                with patch.object(window, "_confirm_cloud_use") as confirm, \
                     patch.object(window.action_service, "record_rejected_request") as audit:
                    window._send()
                confirm.assert_not_called()
                audit.assert_called_once()
                self.assertEqual(window.store.get_messages(window.conversation_id), [])
                self.assertIn("No action was run", window.transcript.toPlainText())
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
