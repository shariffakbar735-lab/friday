from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app import main as friday_main
from app.system.metrics import MetricObservation, SystemMetricsSnapshot


class RecordingLocalClient:
    def __init__(self):
        self.calls = []

    def stream_chat(self, model, messages, stop=None, options=None):
        self.calls.append((model, messages, options))
        yield "A local explanation of the observed readings."


def make_snapshot(*observations, complete=True, cancelled=False):
    category = {
        "cpu_utilization": "cpu", "cpu_logical_core_count": "cpu",
        "physical_ram": "ram", "physical_ram_total": "ram",
        "physical_ram_used": "ram", "physical_ram_available": "ram",
        "physical_ram_percent_used": "ram", "disk_partitions": "disk",
        "battery": "battery", "processes": "processes",
    }
    return SystemMetricsSnapshot(
        "2026-10-24T01:00:00+00:00",
        tuple(dict.fromkeys(category.get(row.metric, row.metric) for row in observations)),
        tuple(observations), complete, cancelled,
    )


class SystemInspectionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def worker(self, client, *, backend="local", metrics=("cpu", "ram"), disk_mountpoint=None):
        return friday_main.ChatWorker(
            client, "local-test", [
                {"role": "system", "content": "Be clear."},
                {"role": "user", "content": "Show my computer system status"},
            ], {"num_ctx": 2048, "num_predict": 128}, backend=backend,
            system_metrics=metrics, disk_mountpoint=disk_mountpoint,
        )

    def test_requested_examples_are_recognized_and_unrelated_chat_is_not_scanned(self):
        expected = {
            "How much RAM is being used?": ("ram",),
            "Check my CPU usage.": ("cpu",),
            "How much disk space is free?": ("disk",),
            "What is my battery percentage?": ("battery",),
        }
        for text, keys in expected.items():
            with self.subTest(text=text):
                self.assertEqual(friday_main.detect_system_request(text).metrics, keys)
        c_drive = friday_main.detect_system_request("How much free disk space is available on C:?")
        self.assertEqual(c_drive.metrics, ("disk",))
        self.assertEqual(c_drive.disk_mountpoint, "C:\\")
        broad = friday_main.detect_system_request("Show my computer's system status")
        self.assertEqual(broad.metrics, ("cpu", "ram", "disk", "battery", "processes"))
        top = friday_main.detect_system_request("List the top processes by memory usage")
        self.assertEqual(top.metrics, ("processes",))
        self.assertTrue(top.top_processes_by_memory)
        self.assertTrue(friday_main.detect_system_request(
            "Which process uses the most memory?"
        ).top_processes_by_memory)
        self.assertIsNone(friday_main.detect_system_request("Explain how CPU scheduling works"))
        self.assertIsNone(friday_main.detect_system_request("What is RAM?"))
        self.assertIsNone(friday_main.detect_system_request("How does battery chemistry work?"))

    def test_observed_metric_values_reach_only_the_local_model(self):
        client = RecordingLocalClient()
        snapshot = make_snapshot(
            MetricObservation("cpu_utilization", 13.75, "percent", "ok", "t1"),
            MetricObservation("physical_ram_percent_used", 62.5, "percent", "ok", "t1"),
        )
        worker = self.worker(client)
        activity, finished, failed = [], [], []
        worker.activity.connect(activity.append)
        worker.finished_text.connect(finished.append)
        worker.failed.connect(lambda error, partial: failed.append((error, partial)))
        with patch.object(friday_main, "collect_system_metrics", return_value=snapshot) as collect:
            worker.run()
        collect.assert_called_once()
        self.assertEqual(collect.call_args.args[0], ("cpu", "ram"))
        self.assertEqual(len(client.calls), 1)
        prompt = client.calls[0][1]
        data_message = next(m["content"] for m in prompt
                            if m.get("content", "").startswith("Read-only local system measurements"))
        data = json.loads(data_message.split("\n", 1)[1])
        values = {item["metric"]: item["value"] for item in data["observations"]}
        self.assertEqual(values["cpu_utilization"], 13.75)
        self.assertEqual(values["physical_ram_percent_used"], 62.5)
        self.assertEqual(finished, ["A local explanation of the observed readings."])
        self.assertEqual(failed, [])
        self.assertTrue(any("Collecting requested system measurements locally" in text for text in activity))

    def test_requested_drive_filter_reaches_local_collector_and_model(self):
        client = RecordingLocalClient()
        gib = 1024 ** 3
        snapshot = make_snapshot(MetricObservation(
            "disk_partitions", {
                "requested_mountpoint": "C:\\", "total_drives_observed": 1,
                "shown_drives": 1, "accessible_drives": 1, "truncated": False,
                "drives": [{
                    "mountpoint": "C:\\", "total_bytes": 100 * gib,
                    "used_bytes": 60 * gib, "free_bytes": 40 * gib,
                    "total_gib": 100.0, "used_gib": 60.0, "free_gib": 40.0,
                    "percent_used": 60.0, "status": "ok",
                }],
            }, "raw bytes, binary GiB (1 GiB = 1024^3 bytes), and percent", "ok", "t1",
        ))
        worker = self.worker(client, metrics=("disk",), disk_mountpoint="C:\\")
        with patch.object(friday_main, "collect_system_metrics", return_value=snapshot) as collect:
            worker.run()
        collect.assert_called_once()
        self.assertEqual(collect.call_args.kwargs["disk_mountpoint"], "C:\\")
        data_message = next(
            message["content"] for message in client.calls[0][1]
            if message.get("content", "").startswith("Read-only local system measurements")
        )
        data = json.loads(data_message.split("\n", 1)[1])
        drive = data["observations"][0]["value"]["drives"][0]
        self.assertEqual(drive["mountpoint"], "C:\\")
        self.assertEqual((drive["total_gib"], drive["used_gib"], drive["free_gib"]),
                         (100.0, 60.0, 40.0))

    def test_partial_snapshot_retains_error_and_valid_reading_for_local_explanation(self):
        client = RecordingLocalClient()
        snapshot = make_snapshot(
            MetricObservation("cpu_utilization", 22.0, "percent", "ok", "t1"),
            MetricObservation("physical_ram", None, None, "error", "t1", error="OSError: RAM read denied"),
            complete=False,
        )
        worker = self.worker(client)
        finished, failed = [], []
        worker.finished_text.connect(finished.append)
        worker.failed.connect(lambda error, partial: failed.append((error, partial)))
        with patch.object(friday_main, "collect_system_metrics", return_value=snapshot):
            worker.run()
        self.assertEqual(len(client.calls), 1)
        prompt = client.calls[0][1]
        data_message = next(m["content"] for m in prompt
                            if m.get("content", "").startswith("Read-only local system measurements"))
        data = json.loads(data_message.split("\n", 1)[1])
        self.assertFalse(data["complete"])
        self.assertEqual(data["observations"][1]["error"], "OSError: RAM read denied")
        self.assertEqual(finished, ["A local explanation of the observed readings."])
        self.assertEqual(failed, [])

    def test_all_measurement_failures_are_reported_without_invoking_a_model(self):
        client = RecordingLocalClient()
        snapshot = make_snapshot(
            MetricObservation("cpu_utilization", None, "percent", "error", "t1", error="PermissionError: CPU denied"),
            complete=False,
        )
        worker = self.worker(client, metrics=("cpu",))
        failed, finished = [], []
        worker.failed.connect(lambda error, partial: failed.append((error, partial)))
        worker.finished_text.connect(finished.append)
        with patch.object(friday_main, "collect_system_metrics", return_value=snapshot):
            worker.run()
        self.assertIn("PermissionError: CPU denied", failed[0][0])
        self.assertEqual(client.calls, [])
        self.assertEqual(finished, [])

    def test_cloud_backend_refuses_system_inspection_before_collection(self):
        client = RecordingLocalClient()
        worker = self.worker(client, backend="cloud")
        failed = []
        worker.failed.connect(lambda error, partial: failed.append(error))
        with patch.object(friday_main, "collect_system_metrics") as collect:
            worker.run()
        collect.assert_not_called()
        self.assertEqual(client.calls, [])
        self.assertIn("local-only", failed[0])
        self.assertIn("No system data was collected", failed[0])

    def test_stop_before_collection_skips_tool_and_model(self):
        client = RecordingLocalClient()
        worker = self.worker(client)
        worker.stop()
        with patch.object(friday_main, "collect_system_metrics") as collect:
            worker.run()
        collect.assert_not_called()
        self.assertEqual(client.calls, [])

    def test_stop_during_collection_prevents_the_following_model_call(self):
        client = RecordingLocalClient()
        worker = self.worker(client)
        snapshot = make_snapshot(
            MetricObservation("cpu_utilization", 11.0, "percent", "ok", "t1"),
            cancelled=True,
        )

        def stop_during_measurement(_requested, *, stop_event, **_kwargs):
            stop_event.set()
            return snapshot

        with patch.object(friday_main, "collect_system_metrics", side_effect=stop_during_measurement) as collect:
            worker.run()
        collect.assert_called_once()
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
