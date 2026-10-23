from __future__ import annotations

import os
import tempfile
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from app import main as friday_main
from app.ai.hardware import HardwareSnapshot
from app.ai.ollama_client import OllamaError
from app.ai.profiles import PROFILES


class MemorySettings:
    def __init__(self, *args):
        self.values = {"provider": "cloud"}
    def value(self, key, default=None):
        return self.values.get(key, default)
    def setValue(self, key, value):
        self.values[key] = value


class UISafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_window(self, directory):
        env = {
            "LOCALAPPDATA": directory,
            "FRIDAY_CLOUD_BASE_URL": "http://127.0.0.1:12345/v1",
            "FRIDAY_CLOUD_MODEL": "test-model",
            "FRIDAY_CLOUD_API_KEY": "test-only-secret",
        }
        with patch.dict(os.environ, env), patch.object(friday_main, "QSettings", MemorySettings), \
             patch.object(friday_main.FridayWindow, "_start_discovery", lambda _self: None):
            window = friday_main.FridayWindow()
        return window

    def test_context_budget_keeps_newest_contiguous_suffix(self):
        messages = [{"role": "user", "content": "a" * 5},
                    {"role": "assistant", "content": "b" * 5},
                    {"role": "user", "content": "c" * 5}]
        self.assertEqual(friday_main.bounded_context(messages, 11), messages[1:])
        self.assertEqual(friday_main.bounded_context(messages, 4), [])
        self.assertLessEqual(friday_main.prompt_char_budget(PROFILES["low-resource"].num_ctx),
                             PROFILES["low-resource"].num_ctx)

    def test_gpu_error_classification_is_targeted(self):
        self.assertTrue(friday_main.is_gpu_initialization_error("CUDA driver failed to initialize"))
        self.assertTrue(friday_main.is_gpu_initialization_error("GPU offload failed"))
        self.assertFalse(friday_main.is_gpu_initialization_error("Connection timed out"))

    def test_simulated_gpu_initialization_failure_offers_explicit_local_cpu_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(directory)
            try:
                window.backend.blockSignals(True)
                window.backend.setCurrentIndex(window.backend.findData("local"))
                window.backend.blockSignals(False)
                window.active_response_conversation_id = window.conversation_id
                window.active_messages = [{"role": "user", "content": "last request"}]
                with patch.object(QMessageBox, "warning"):
                    window._show_error("CUDA driver failed to initialize")
                self.assertFalse(window.cpu_retry_button.isHidden())
                self.assertEqual(window.last_failed_conversation_id, window.conversation_id)
            finally:
                window.close()

    def test_partial_answer_is_preserved_in_local_conversation(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(directory)
            try:
                window.store.add_message(window.conversation_id, "user", "question")
                window.active_response_conversation_id = window.conversation_id
                window.streamed_text = "partial answer"
                with patch.object(QMessageBox, "warning"):
                    window._show_error("Ollama response timed out", "partial answer")
                saved = window.store.get_messages(window.conversation_id)
                self.assertEqual([m["role"] for m in saved], ["user", "assistant"])
                self.assertEqual(saved[-1]["content"], "partial answer")
            finally:
                window.close()

    def test_cloud_persisted_from_prior_run_requires_session_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(directory)
            try:
                self.assertEqual(window.backend.currentData(), "cloud")
                self.assertFalse(window._cloud_confirmed)
                window.input.setText("message that must not be sent without consent")
                class ConfiguredClient:
                    configured = True
                with patch.object(friday_main, "OpenAICompatibleClient", ConfiguredClient), \
                     patch.object(QMessageBox, "question", return_value=QMessageBox.No) as confirm:
                    window._send()
                confirm.assert_called_once()
                self.assertIsNone(window.worker)
                self.assertEqual(window.store.get_messages(window.conversation_id)[0]["content"],
                                 "message that must not be sent without consent")
                self.assertFalse(window._cloud_confirmed)
            finally:
                window.close()

    def test_cpu_retry_cannot_switch_to_a_different_conversation(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(directory)
            try:
                window.active_messages = [{"role": "user", "content": "failed prompt"}]
                window.last_failed_conversation_id = "original-conversation"
                window.conversation_id = "other-conversation"
                with patch.object(QMessageBox, "information"), \
                     patch.object(window, "_start_generation") as start:
                    window._retry_cpu()
                start.assert_not_called()
            finally:
                window.close()

    def test_send_guard_prevents_chat_during_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(directory)
            try:
                window.benchmark_worker = object()
                window.input.setText("must not overlap")
                window._send()
                self.assertEqual(window.store.get_messages(window.conversation_id), [])
            finally:
                window.close()

    def test_startup_with_ollama_unavailable_keeps_history_available(self):
        class OfflineClient:
            def __init__(self):
                pass
            def list_models(self):
                raise OllamaError("not running")

        with tempfile.TemporaryDirectory() as directory:
            snapshot = HardwareSnapshot(8 * 1024**3, 4 * 1024**3, "AMD64", 4,
                                        None, None, None, False, False,
                                        ("simulated: no GPU",))
            with patch.dict(os.environ, {"LOCALAPPDATA": directory,
                                         "FRIDAY_CLOUD_BASE_URL": "",
                                         "FRIDAY_CLOUD_MODEL": "",
                                         "FRIDAY_CLOUD_API_KEY": ""}), \
                 patch.object(friday_main, "QSettings", MemorySettings), \
                 patch.object(friday_main, "OllamaClient", OfflineClient), \
                 patch.object(friday_main, "detect_hardware", return_value=snapshot):
                window = friday_main.FridayWindow()
            try:
                worker = window.startup_worker
                self.assertIsNotNone(worker)
                loop = QEventLoop()
                worker.initialized.connect(loop.quit)
                QTimer.singleShot(5000, loop.quit)
                loop.exec()
                self.assertTrue(worker.wait(1000))
                self.app.processEvents()
                self.assertFalse(window.snapshot.ollama_running)
                self.assertFalse(window.send_button.isEnabled())
                self.assertTrue(window.history.isEnabled())
                self.assertTrue(window.new_button.isEnabled())
                window.store.add_message(window.conversation_id, "user", "offline note")
                self.assertEqual(window.store.get_messages(window.conversation_id)[0]["content"], "offline note")
            finally:
                window.close()

    def test_benchmark_guard_blocks_start_while_chat_worker_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(directory)
            try:
                window.worker = object()
                window._run_benchmark()
                self.assertIsNone(window.benchmark_worker)
            finally:
                window.close()

    def test_chat_worker_emits_partial_text_with_failure(self):
        class PartialFailureClient:
            def stream_chat(self, *_args, **_kwargs):
                yield "partial"
                raise RuntimeError("simulated stream reset")

        worker = friday_main.ChatWorker(PartialFailureClient(), "test", [],
                                        {"num_ctx": PROFILES["balanced"].num_ctx,
                                         "num_predict": 16})
        result = []
        loop = QEventLoop()
        worker.failed.connect(lambda message, partial: (result.append((message, partial)), loop.quit()))
        worker.start()
        QTimer.singleShot(3000, loop.quit)
        loop.exec()
        self.assertTrue(worker.wait(1000))
        self.assertEqual(result, [("simulated stream reset", "partial")])

    def test_repeated_stop_releases_worker_and_allows_a_new_request(self):
        first_started = threading.Event()
        calls = []

        class SequencedClient:
            def stream_chat(self, _model, _messages, stop=None, options=None):
                calls.append(1)
                if len(calls) == 1:
                    first_started.set()
                    yield "partial"
                    stop.wait(3)
                    return
                yield "new answer"

        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(directory)
            try:
                window.backend.blockSignals(True)
                window.backend.setCurrentIndex(window.backend.findData("local"))
                window.backend.blockSignals(False)
                window.models.addItem("test:latest")
                window.client = SequencedClient()
                window.store.add_message(window.conversation_id, "user", "first prompt")
                messages = [{"role": "user", "content": "first prompt"}]
                window._start_generation(messages, "local", cpu_only=False)
                first_worker = window.worker
                self.assertTrue(first_started.wait(1))
                loop_ready = QEventLoop()
                def await_first_render():
                    if window.streamed_text:
                        loop_ready.quit()
                    else:
                        QTimer.singleShot(10, await_first_render)
                QTimer.singleShot(10, await_first_render)
                QTimer.singleShot(1000, loop_ready.quit)
                loop_ready.exec()
                self.assertEqual(window.streamed_text, "partial")
                window._stop()
                window._stop()  # Mirrors a repeated Stop click; the signal is idempotent.
                loop = QEventLoop()
                first_worker.finished.connect(loop.quit)
                QTimer.singleShot(2000, loop.quit)
                loop.exec()
                self.assertTrue(first_worker.wait(500))
                self.app.processEvents()
                self.assertIsNone(window.worker)
                self.assertFalse(window.stop_button.isEnabled())
                self.assertTrue(window.input.isEnabled())
                history = window.store.get_messages(window.conversation_id)
                self.assertEqual(history[-1]["content"], "partial")

                window.store.add_message(window.conversation_id, "user", "second prompt")
                window._start_generation([{"role": "user", "content": "second prompt"}],
                                         "local", cpu_only=False)
                second_worker = window.worker
                loop2 = QEventLoop()
                second_worker.finished.connect(loop2.quit)
                QTimer.singleShot(2000, loop2.quit)
                loop2.exec()
                self.assertTrue(second_worker.wait(500))
                self.app.processEvents()
                self.assertIsNone(window.worker)
                self.assertEqual(calls, [1, 1])
                self.assertEqual(window.store.get_messages(window.conversation_id)[-1]["content"],
                                 "new answer")
            finally:
                if window.worker is not None:
                    window.worker.stop()
                    window.worker.wait(1000)
                window.close()


if __name__ == "__main__":
    unittest.main()
