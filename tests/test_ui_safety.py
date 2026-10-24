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
from app.ai.ollama_client import Model, OllamaError
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

    def make_scrolling_window(self, directory):
        window = self.make_window(directory)
        window.resize(760, 420)
        window.show()
        for index in range(45):
            window.store.add_message(
                window.conversation_id,
                "user",
                f"Earlier message {index}: " + ("historical text " * 24),
            )
        window._render_messages()
        self.app.processEvents()
        self.assertGreater(window.transcript.verticalScrollBar().maximum(), 0)
        window.active_response_conversation_id = window.conversation_id
        window._streaming_visible = True
        window._render_messages()
        self.app.processEvents()
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

    def test_valid_empty_local_completion_is_not_a_worker_failure(self):
        class EmptyCompletedClient:
            def stream_chat(self, *_args, **_kwargs):
                return iter(())

        worker = friday_main.ChatWorker(
            EmptyCompletedClient(), "test:latest",
            [{"role": "user", "content": "Complete with no visible tokens"}],
            {"num_ctx": 2048, "num_predict": 16},
        )
        completed, failed = [], []
        worker.finished_text.connect(completed.append)
        worker.failed.connect(lambda message, partial: failed.append((message, partial)))
        worker.run()
        self.assertEqual(completed, [""])
        self.assertEqual(failed, [])

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

    def test_process_inspection_request_routes_to_local_even_when_cloud_is_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(directory)
            try:
                window.snapshot = HardwareSnapshot(16 * 1024**3, 8 * 1024**3, "AMD64", 8,
                                                   None, None, None, True, True, ())
                window.models.addItem("test:local")
                window.installed_models = [Model("test:local", 1024**3)]
                window.input.setText("List my running applications")
                with patch.object(QMessageBox, "question") as cloud_confirm, \
                     patch.object(window, "_start_generation") as start:
                    window._send()
                cloud_confirm.assert_not_called()
                start.assert_called_once()
                self.assertEqual(start.call_args.args[1], "local")
                self.assertTrue(start.call_args.kwargs["inspect_processes"])
                self.assertEqual(start.call_args.kwargs["model_override"], "test:local")
                self.assertEqual(window.backend.currentData(), "cloud")
                self.assertEqual(window.store.get_messages(window.conversation_id)[0]["content"],
                                 "List my running applications")
            finally:
                window.close()

    def test_system_status_routes_local_and_only_user_facing_response_is_persisted(self):
        from app.system.metrics import MetricObservation, SystemMetricsSnapshot

        class RecordingClient:
            def __init__(self):
                self.calls = []
            def stream_chat(self, model, messages, stop=None, options=None):
                self.calls.append((model, messages, options))
                yield "Local system status explanation."

        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(directory)
            try:
                window.snapshot = HardwareSnapshot(16 * 1024**3, 8 * 1024**3, "AMD64", 8,
                                                   None, None, None, True, True, ())
                window.models.addItem("test:local")
                window.installed_models = [Model("test:local", 1024**3)]
                client = RecordingClient()
                window.client = client
                window.input.setText("Show my computer's system status")
                tool_snapshot = SystemMetricsSnapshot(
                    "2026-10-24T01:00:00+00:00",
                    ("cpu", "ram", "disk", "battery", "processes"),
                    (MetricObservation("cpu_utilization", 37.25, "percent", "ok", "t1"),), False,
                )
                with patch.object(friday_main, "collect_system_metrics", return_value=tool_snapshot) as collect, \
                     patch.object(QMessageBox, "question") as cloud_confirm:
                    window._send()
                    worker = window.worker
                    self.assertTrue(worker.wait(3000))
                    self.app.processEvents()
                collect.assert_called_once()
                self.assertEqual(collect.call_args.args[0], ("cpu", "ram", "disk", "battery", "processes"))
                cloud_confirm.assert_not_called()
                self.assertEqual(window.backend.currentData(), "cloud")
                self.assertEqual(window.active_generation_backend, "local")
                self.assertEqual(len(client.calls), 1)
                self.assertIn("37.25", " ".join(message["content"] for message in client.calls[0][1]))
                saved = window.store.get_messages(window.conversation_id)
                self.assertEqual([item["role"] for item in saved], ["user", "assistant"])
                self.assertEqual(saved[-1]["content"], "Local system status explanation.")
                self.assertNotIn("37.25", " ".join(item["content"] for item in saved))
            finally:
                if window.worker is not None:
                    window.worker.stop()
                    window.worker.wait(1000)
                window.close()

    def test_system_query_without_local_model_never_collects_or_routes_to_cloud(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(directory)
            try:
                window.snapshot = HardwareSnapshot(16 * 1024**3, 8 * 1024**3, "AMD64", 8,
                                                   None, None, None, True, True, ())
                window.input.setText("Check my CPU usage")
                with patch.object(friday_main, "collect_system_metrics") as collect, \
                     patch.object(QMessageBox, "information"), \
                     patch.object(QMessageBox, "question") as cloud_confirm:
                    window._send()
                collect.assert_not_called()
                cloud_confirm.assert_not_called()
                self.assertIsNone(window.worker)
                self.assertEqual(window.backend.currentData(), "cloud")
                self.assertEqual(window.store.get_messages(window.conversation_id)[0]["content"],
                                 "Check my CPU usage")
            finally:
                window.close()

    def test_streaming_follows_latest_response_when_user_was_at_bottom(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_scrolling_window(directory)
            try:
                scrollbar = window.transcript.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
                window._append_chunk("latest answer token " * 160)
                self.app.processEvents()
                self.assertEqual(scrollbar.value(), scrollbar.maximum())
                self.assertIn("latest answer token", window.transcript.toPlainText())
            finally:
                window.close()

    def test_streaming_preserves_upward_scroll_and_keeps_user_scrolling_usable(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_scrolling_window(directory)
            try:
                scrollbar = window.transcript.verticalScrollBar()
                initial_position = max(1, scrollbar.maximum() // 2)
                scrollbar.setValue(initial_position)
                cursor = window._stream_cursor
                window._append_chunk("continued answer token " * 160)
                self.app.processEvents()
                self.assertEqual(scrollbar.value(), initial_position)
                self.assertLess(scrollbar.value(), scrollbar.maximum())
                self.assertIs(window._stream_cursor, cursor)
                self.assertIn("continued answer token", window.transcript.toPlainText())

                user_position = min(scrollbar.maximum(), initial_position + 12)
                scrollbar.setValue(user_position)
                window._append_chunk("another streamed segment " * 120)
                self.app.processEvents()
                self.assertEqual(scrollbar.value(), user_position)

                window._render_messages()
                self.app.processEvents()
                self.assertEqual(scrollbar.value(), user_position)
            finally:
                window.close()

    def test_safe_markdown_formatter_supports_requested_blocks_and_escapes_html(self):
        rendered = friday_main.safe_markdown_to_html(
            "# Heading\n\n**bold text**\n\n1. First\n2. Second\n\n"
            "- Bullet A\n- Bullet B\n\nRaw <script>alert(1)</script>"
        )
        self.assertIn("<h1", rendered)
        self.assertIn("<strong>bold text</strong>", rendered)
        self.assertIn("<ol", rendered)
        self.assertIn("<ul", rendered)
        self.assertIn('<p style="margin:0 0 12px 0">', rendered)
        self.assertNotIn("<script", rendered.lower())
        self.assertIn("&lt;script&gt;", rendered.lower())
        self.assertNotIn("<a ", rendered.lower())

    def test_incremental_markdown_stream_formats_blocks_safely_and_keeps_scroll(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_scrolling_window(directory)
            try:
                window.streamed_text = ""
                window._stream_pending_markdown = ""
                window._stream_committed_html = ""
                window._render_messages()
                scrollbar = window.transcript.verticalScrollBar()
                start = max(1, scrollbar.maximum() // 2)
                scrollbar.setValue(start)
                cursor = window._stream_cursor
                for chunk in (
                    "# Streaming heading", "\n\n**bold", " words**\n\n",
                    "1. First item\n2. Second item\n\n",
                    "- Bullet one\n- Bullet two\n\n",
                    "Raw <script>alert(1)</script>.",
                ):
                    window._append_chunk(chunk)
                self.app.processEvents()
                rendered = window.transcript.toHtml().lower()
                plain = window.transcript.toPlainText().lower()
                self.assertIn("streaming heading", plain)
                self.assertIn("bold words", plain)
                self.assertIn("first item", plain)
                self.assertIn("bullet two", plain)
                self.assertIn("<h1", window._stream_committed_html.lower())
                self.assertIn("<strong>bold words</strong>", window._stream_committed_html.lower())
                self.assertIn("<ol", window._stream_committed_html.lower())
                self.assertIn("<ul", window._stream_committed_html.lower())
                self.assertNotIn("<script>alert", rendered)
                self.assertFalse(window.transcript.openExternalLinks())
                self.assertFalse(window.transcript.openLinks())
                self.assertIs(window._stream_cursor, cursor)
                self.assertEqual(scrollbar.value(), start)
            finally:
                window.close()

    def test_streaming_answer_stays_below_friday_label_across_switch_and_final_render(self):
        def transcript_blocks(window):
            block = window.transcript.document().begin()
            texts = []
            while block.isValid():
                texts.append(block.text())
                block = block.next()
            return texts

        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(directory)
            try:
                original_conversation = window.conversation_id
                window.store.add_message(original_conversation, "user", "Question")
                window.active_response_conversation_id = original_conversation
                window._streaming_visible = True
                window._render_messages()
                window._append_chunk("# Stream heading\n\nStreaming answer")

                blocks = transcript_blocks(window)
                label_index = blocks.index("FRIDAY")
                heading_index = next(i for i, text in enumerate(blocks) if "Stream heading" in text)
                answer_index = next(i for i, text in enumerate(blocks) if "Streaming answer" in text)
                self.assertLess(label_index, heading_index)
                self.assertLess(heading_index, answer_index)

                other_conversation = window.store.create_conversation()
                window.conversation_id = other_conversation
                window._render_messages()
                self.assertIsNone(window._stream_cursor)
                self.assertIsNone(window._stream_tail_start)

                window.conversation_id = original_conversation
                window._render_messages()
                self.assertIsNotNone(window._stream_cursor)
                self.assertIsNotNone(window._stream_tail_start)
                window._append_chunk(" continues.")
                self.app.processEvents()

                window._finish_response("# Stream heading\n\nStreaming answer continues.")
                final_blocks = transcript_blocks(window)
                final_label_index = final_blocks.index("FRIDAY")
                final_answer_index = next(
                    i for i, text in enumerate(final_blocks) if "Streaming answer continues." in text
                )
                self.assertGreater(final_answer_index, final_label_index)
                self.assertEqual(
                    sum(text.count("Streaming answer continues.") for text in final_blocks), 1
                )
                self.assertIsNone(window._stream_cursor)
                self.assertIsNone(window._stream_tail_start)
            finally:
                window.close()

    def test_hardware_setup_dialog_requires_an_explicit_button_click(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(directory)
            window.snapshot = HardwareSnapshot(8 * 1024**3, 4 * 1024**3, "AMD64", 4,
                                              None, None, None, False, False, ())
            try:
                with patch.object(QMessageBox, "information") as information, \
                     patch.object(QMessageBox, "warning"):
                    window.active_response_conversation_id = window.conversation_id
                    window._streaming_visible = True
                    window._render_messages()
                    window._append_chunk("Normal response while streaming")
                    window._render_messages()
                    window._finish_response("Normal assistant response")
                    window._show_error("Simulated inference failure")
                    information.assert_not_called()

                    window._show_hardware()
                    information.assert_not_called()
                    window.hardware_button.click()
                    information.assert_called_once()
                    self.assertEqual(information.call_args.args[1], "FRIDAY hardware & setup")
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
                window.show()
                self.assertTrue(window.isVisible())
                worker = window.startup_worker
                self.assertIsNotNone(worker)
                self.assertTrue(worker.wait(3000), "startup discovery worker did not finish promptly")
                self.app.processEvents()
                self.assertFalse(window.snapshot.ollama_running)
                self.assertIn("Ollama is not responding", window.status.text())
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
