from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app import main as friday_main
from app.system.processes import ProcessInspectionError, ProcessRecord


class RecordingLocalClient:
    def __init__(self):
        self.calls = []

    def stream_chat(self, model, messages, stop=None, options=None):
        self.calls.append((model, messages, options))
        yield "Observed process summary."


class ProcessWorkerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_actual_tool_records_are_passed_to_local_model_and_activity_is_visible(self):
        client = RecordingLocalClient()
        records = [ProcessRecord("observed-app.exe", 4321, r"C:\Apps\observed-app.exe", 123456)]
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "List my running applications"},
        ]
        worker = friday_main.ChatWorker(
            client, "local-test", messages, {"num_ctx": 2048, "num_predict": 64},
            backend="local", inspect_processes=True,
        )
        activity = []
        finished = []
        failed = []
        worker.activity.connect(activity.append)
        worker.finished_text.connect(finished.append)
        worker.failed.connect(lambda error, partial: failed.append((error, partial)))
        with patch.object(friday_main, "list_running_processes", return_value=records) as inspect:
            worker.run()

        inspect.assert_called_once_with(stop_event=worker.stop_event)
        self.assertEqual(len(client.calls), 1)
        sent = client.calls[0][1]
        tool_message = next(message["content"] for message in sent
                            if message.get("content", "").startswith("Read-only local process inspection results"))
        payload = json.loads(tool_message.split("\n", 1)[1])
        self.assertEqual(payload["processes"][0]["name"], "observed-app.exe")
        self.assertEqual(payload["processes"][0]["pid"], 4321)
        self.assertEqual(finished, ["Observed process summary."])
        self.assertEqual(failed, [])
        self.assertTrue(any("Inspecting Windows processes locally" in text for text in activity))

    def test_inspection_failure_reports_real_error_and_never_calls_model(self):
        client = RecordingLocalClient()
        worker = friday_main.ChatWorker(
            client, "local-test", [{"role": "user", "content": "list running processes"}],
            {"num_ctx": 2048, "num_predict": 64}, backend="local", inspect_processes=True,
        )
        failed = []
        finished = []
        worker.failed.connect(lambda error, partial: failed.append((error, partial)))
        worker.finished_text.connect(finished.append)
        with patch.object(friday_main, "list_running_processes",
                          side_effect=ProcessInspectionError("AccessDenied: inspection denied")):
            worker.run()
        self.assertIn("AccessDenied: inspection denied", failed[0][0])
        self.assertEqual(failed[0][1], "")
        self.assertEqual(finished, [])
        self.assertEqual(client.calls, [])

    def test_cloud_worker_refuses_inspection_without_collecting_or_sending_data(self):
        client = RecordingLocalClient()
        worker = friday_main.ChatWorker(
            client, "cloud-test", [{"role": "user", "content": "show running apps"}],
            {"num_ctx": 2048, "num_predict": 64}, backend="cloud", inspect_processes=True,
        )
        failed = []
        worker.failed.connect(lambda error, partial: failed.append((error, partial)))
        with patch.object(friday_main, "list_running_processes") as inspect:
            worker.run()
        inspect.assert_not_called()
        self.assertEqual(client.calls, [])
        self.assertIn("local-only", failed[0][0])
        self.assertIn("No process data was collected", failed[0][0])

    def test_process_intent_recognizes_common_requests_only(self):
        for text in (
            "List my running applications",
            "Show running apps",
            "What processes are running?",
            "Inspect Windows processes",
        ):
            with self.subTest(text=text):
                self.assertTrue(friday_main.is_process_inspection_request(text))
        self.assertFalse(friday_main.is_process_inspection_request("Tell me about process scheduling"))


if __name__ == "__main__":
    unittest.main()

