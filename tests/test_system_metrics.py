from __future__ import annotations

import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from app.system import metrics
from app.system.processes import ProcessInspectionError, ProcessRecord


class FakePsutil:
    POWER_TIME_UNLIMITED = -2
    POWER_TIME_UNKNOWN = -1

    def __init__(self, *, cpu=25.0, cores=8, memory=None, partitions=None,
                 disk_usages=None, battery=None):
        self.cpu_percent = Mock(return_value=cpu)
        self.cpu_count = Mock(return_value=cores)
        self.virtual_memory = Mock(return_value=memory or SimpleNamespace(
            total=8 * 1024**3, available=2 * 1024**3,
        ))
        self.disk_partitions = Mock(return_value=list(partitions or []))
        self.disk_usage = Mock(side_effect=lambda mount: (disk_usages or {})[mount])
        self.sensors_battery = Mock(return_value=battery)


class SystemMetricsTests(unittest.TestCase):
    def collect(self, backend, requested, **kwargs):
        return metrics.collect_system_metrics(requested, psutil_module=backend, **kwargs)

    @staticmethod
    def by_name(snapshot):
        return {item.metric: item for item in snapshot.observations}

    def test_cpu_percentage_and_core_count_are_actual_sampled_values(self):
        backend = FakePsutil(cpu=37.5, cores=12)
        snapshot = self.collect(backend, ("cpu",), cpu_interval=0.1)
        result = self.by_name(snapshot)
        self.assertEqual(result["cpu_utilization"].value, 37.5)
        self.assertEqual(result["cpu_utilization"].status, "ok")
        self.assertEqual(result["cpu_utilization"].unit, "percent")
        self.assertEqual(result["cpu_logical_core_count"].value, 12)
        backend.cpu_percent.assert_called_once_with(interval=0.1, percpu=True)
        backend.cpu_count.assert_called_once_with(logical=True)

    def test_per_core_cpu_samples_are_preserved_and_aggregate_is_their_mean(self):
        backend = FakePsutil(cpu=[10.0, 20.0, 30.0], cores=3)
        result = self.by_name(self.collect(backend, ("cpu",)))
        self.assertEqual(result["cpu_utilization"].value, 20.0)
        self.assertEqual(result["cpu_per_core_utilization"].status, "ok")
        self.assertEqual(
            [item["percent"] for item in result["cpu_per_core_utilization"].value["samples"]],
            [10.0, 20.0, 30.0],
        )
        self.assertIn("Arithmetic mean", result["cpu_utilization"].limitation)

    def test_invalid_per_core_sample_keeps_valid_cores_without_inventing_aggregate(self):
        result = self.by_name(self.collect(FakePsutil(cpu=[10.0, None, 30.0]), ("cpu",)))
        self.assertEqual(result["cpu_per_core_utilization"].status, "partial")
        samples = result["cpu_per_core_utilization"].value["samples"]
        self.assertEqual([item["status"] for item in samples], ["ok", "error", "ok"])
        self.assertIsNone(samples[1]["percent"])
        self.assertEqual(result["cpu_utilization"].status, "error")
        self.assertIsNone(result["cpu_utilization"].value)

    def test_unavailable_initial_cpu_sample_is_marked_warming_up(self):
        backend = FakePsutil(cpu=None)
        result = self.by_name(self.collect(backend, ("cpu",)))["cpu_utilization"]
        self.assertIsNone(result.value)
        self.assertEqual(result.status, "warming_up")
        self.assertIn("not produced", result.limitation)

    def test_ram_used_available_and_percentage_use_physical_memory_consistently(self):
        backend = FakePsutil(memory=SimpleNamespace(total=8_000, available=2_000))
        result = self.by_name(self.collect(backend, ("ram",)))
        self.assertEqual(result["physical_ram_total"].value, 8_000)
        self.assertEqual(result["physical_ram_available"].value, 2_000)
        self.assertEqual(result["physical_ram_used"].value, 6_000)
        self.assertEqual(result["physical_ram_percent_used"].value, 75.0)
        self.assertIn("total physical RAM minus", result["physical_ram_used"].limitation)

    def test_invalid_ram_totals_are_reported_as_errors(self):
        backend = FakePsutil(memory=SimpleNamespace(total=0, available=5))
        result = self.by_name(self.collect(backend, ("ram",)))["physical_ram"]
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.value)
        self.assertIn("invalid physical-memory totals", result.error)

    def test_bytes_to_gib_uses_binary_units(self):
        self.assertEqual(metrics.bytes_to_gib(1024 ** 3), 1.0)
        self.assertEqual(metrics.bytes_to_gib(447 * 1024 ** 3), 447.0)

    def test_multiple_drives_report_consistent_raw_bytes_and_gib_per_mountpoint(self):
        gib = 1024 ** 3
        partitions = [SimpleNamespace(mountpoint="C:\\"), SimpleNamespace(mountpoint="D:\\")]
        backend = FakePsutil(
            partitions=partitions,
            disk_usages={
                "C:\\": SimpleNamespace(total=100 * gib, used=60 * gib, free=40 * gib),
                "D:\\": SimpleNamespace(total=200 * gib, used=50 * gib, free=150 * gib),
            },
        )
        result = self.by_name(self.collect(backend, ("disk",)))["disk_partitions"]
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.unit, "raw bytes, binary GiB (1 GiB = 1024^3 bytes), and percent")
        self.assertEqual(result.value["total_drives_observed"], 2)
        c_drive, d_drive = result.value["drives"]
        self.assertEqual(c_drive["mountpoint"], "C:\\")
        self.assertEqual((c_drive["total_bytes"], c_drive["used_bytes"], c_drive["free_bytes"]),
                         (100 * gib, 60 * gib, 40 * gib))
        self.assertEqual((c_drive["total_gib"], c_drive["used_gib"], c_drive["free_gib"]),
                         (100.0, 60.0, 40.0))
        self.assertEqual(c_drive["used_bytes"] + c_drive["free_bytes"], c_drive["total_bytes"])
        self.assertEqual(c_drive["percent_used"], 60.0)
        self.assertEqual(d_drive["mountpoint"], "D:\\")
        self.assertEqual((d_drive["total_gib"], d_drive["used_gib"], d_drive["free_gib"]),
                         (200.0, 50.0, 150.0))
        backend.disk_partitions.assert_called_once_with(all=False)
        self.assertEqual(backend.disk_usage.call_args_list,
                         [call("C:\\"), call("D:\\")])

    def test_drive_selector_reads_only_requested_volume(self):
        partitions = [SimpleNamespace(mountpoint="C:\\"), SimpleNamespace(mountpoint="D:\\")]
        backend = FakePsutil(
            partitions=partitions,
            disk_usages={
                "C:\\": SimpleNamespace(total=100, used=60, free=40),
                "D:\\": SimpleNamespace(total=200, used=50, free=150),
            },
        )
        result = self.by_name(self.collect(backend, ("disk",), disk_mountpoint="c:"))["disk_partitions"]
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.value["requested_mountpoint"], "C:\\")
        self.assertEqual(result.value["total_drives_observed"], 1)
        self.assertEqual(result.value["drives"][0]["mountpoint"], "C:\\")
        self.assertEqual(result.value["drives"][0]["total_bytes"], 100)
        backend.disk_usage.assert_called_once_with("C:\\")

    def test_inconsistent_disk_capacity_tuple_is_reported_as_error(self):
        backend = FakePsutil(
            partitions=[SimpleNamespace(mountpoint="C:\\")],
            disk_usages={"C:\\": SimpleNamespace(total=1000, used=500, free=400)},
        )
        result = self.by_name(self.collect(backend, ("disk",)))["disk_partitions"]
        self.assertEqual(result.status, "error")
        self.assertEqual(result.value["drives"][0]["status"], "error")
        self.assertIn("invalid capacity values", result.value["drives"][0]["error"])

    def test_inaccessible_drive_is_partial_and_other_drive_readings_survive(self):
        partitions = [SimpleNamespace(mountpoint="C:\\"), SimpleNamespace(mountpoint="Z:\\")]
        backend = FakePsutil(
            partitions=partitions,
            disk_usages={"C:\\": SimpleNamespace(total=1000, used=700, free=300)},
        )
        def usage(mount):
            if mount == "Z:\\":
                raise PermissionError("access denied")
            return SimpleNamespace(total=1000, used=700, free=300)
        backend.disk_usage.side_effect = usage
        result = self.by_name(self.collect(backend, ("disk",)))["disk_partitions"]
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.value["accessible_drives"], 1)
        self.assertEqual(result.value["drives"][0]["status"], "ok")
        self.assertIn("PermissionError: access denied", result.value["drives"][1]["error"])

    def test_all_inaccessible_drives_are_failure_not_a_successful_snapshot(self):
        backend = FakePsutil(partitions=[SimpleNamespace(mountpoint="Z:\\")])
        backend.disk_usage.side_effect = PermissionError("drive unavailable")
        snapshot = self.collect(backend, ("disk",))
        result = self.by_name(snapshot)["disk_partitions"]
        self.assertEqual(result.status, "error")
        self.assertFalse(snapshot.has_reportable_observation)
        self.assertIn("All reported drives", result.error)

    def test_disk_enumeration_exception_is_reported(self):
        backend = FakePsutil()
        backend.disk_partitions.side_effect = OSError("partition query failed")
        result = self.by_name(self.collect(backend, ("disk",)))["disk_partitions"]
        self.assertEqual(result.status, "error")
        self.assertIn("partition query failed", result.error)

    def test_present_battery_reports_charge_power_connection_and_meaningful_time(self):
        battery = SimpleNamespace(percent=64.0, power_plugged=True, secsleft=1200)
        result = self.by_name(self.collect(FakePsutil(battery=battery), ("battery",)))["battery"]
        self.assertEqual(result.value["charge_percent"], 64.0)
        self.assertIsNone(result.value["charging_state"])
        self.assertEqual(result.value["power_connection"], "plugged_in")
        self.assertEqual(result.value["seconds_remaining"], 1200)
        self.assertIn("charging_state is unavailable", result.limitation)

    def test_battery_on_battery_and_unknown_time_are_labeled_without_inference(self):
        battery = SimpleNamespace(percent=42, power_plugged=False, secsleft=FakePsutil.POWER_TIME_UNKNOWN)
        result = self.by_name(self.collect(FakePsutil(battery=battery), ("battery",)))["battery"]
        self.assertEqual(result.value["power_connection"], "on_battery")
        self.assertIsNone(result.value["seconds_remaining"])
        self.assertEqual(result.status, "partial")
        self.assertIsNone(result.value["charging_state"])

    def test_absent_battery_is_not_misreported_as_zero_percent(self):
        result = self.by_name(self.collect(FakePsutil(battery=None), ("battery",)))["battery"]
        self.assertEqual(result.status, "not_detected")
        self.assertIsNone(result.value)
        self.assertIn("not mean 0 percent", result.limitation)

    def test_battery_sensor_error_and_invalid_percentage_are_explicit(self):
        backend = FakePsutil(battery=SimpleNamespace(percent=140, power_plugged=True, secsleft=-1))
        result = self.by_name(self.collect(backend, ("battery",)))["battery"]
        self.assertEqual(result.status, "partial")
        self.assertIsNone(result.value["charge_percent"])
        self.assertIn("invalid battery percentage", result.error)
        backend.sensors_battery.side_effect = OSError("battery query failed")
        error = self.by_name(self.collect(backend, ("battery",)))["battery"]
        self.assertEqual(error.status, "error")
        self.assertIn("battery query failed", error.error)

    def test_process_tool_is_reused_sorted_by_memory_and_bounded_by_count(self):
        rows = [
            ProcessRecord("small.exe", 3, None, 100),
            ProcessRecord("large.exe", 1, None, 900),
            ProcessRecord("middle.exe", 2, None, 500),
        ]
        backend = FakePsutil()
        with patch.object(metrics, "list_running_processes", return_value=rows) as reader:
            result = self.by_name(self.collect(
                backend, ("processes",), top_processes_by_memory=True,
                max_process_records=2, max_process_chars=4096,
            ))["processes"]
        reader.assert_called_once()
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.value["total_processes_observed"], 3)
        self.assertEqual(result.value["shown_processes"], 2)
        self.assertEqual([p["name"] for p in result.value["processes"]], ["large.exe", "middle.exe"])
        self.assertEqual([p["pid"] for p in result.value["processes"]], [1, 2])
        self.assertEqual([p["memory_bytes"] for p in result.value["processes"]], [900, 500])
        self.assertTrue(result.value["truncated"])

    def test_process_tool_failure_is_preserved_as_an_error_observation(self):
        with patch.object(metrics, "list_running_processes",
                          side_effect=ProcessInspectionError("AccessDenied: process listing denied")) as reader:
            result = self.by_name(self.collect(FakePsutil(), ("processes",)))["processes"]
        reader.assert_called_once()
        self.assertEqual(result.status, "error")
        self.assertIn("AccessDenied", result.error)

    def test_cpu_and_ram_are_independent_when_one_collector_fails(self):
        backend = FakePsutil(cpu=12.0)
        backend.virtual_memory.side_effect = OSError("memory unavailable")
        snapshot = self.collect(backend, ("cpu", "ram"))
        result = self.by_name(snapshot)
        self.assertEqual(result["cpu_utilization"].status, "ok")
        self.assertEqual(result["physical_ram"].status, "error")
        self.assertFalse(snapshot.complete)
        self.assertTrue(snapshot.has_reportable_observation)

    def test_invalid_cpu_percentage_is_not_reported_as_a_reading(self):
        result = self.by_name(self.collect(FakePsutil(cpu=101), ("cpu",)))["cpu_utilization"]
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.value)

    def test_only_requested_collectors_run(self):
        backend = FakePsutil(cpu=5.0)
        self.collect(backend, ("cpu",))
        backend.virtual_memory.assert_not_called()
        backend.disk_partitions.assert_not_called()
        backend.sensors_battery.assert_not_called()

    def test_unknown_metric_is_an_explicit_error(self):
        snapshot = self.collect(FakePsutil(), ("telepathy",))
        self.assertEqual(snapshot.observations[0].status, "error")
        self.assertFalse(snapshot.has_reportable_observation)

    def test_snapshot_serialization_is_bounded_and_marks_truncated_rows(self):
        rows = [{"pid": n, "name": f"process-{n}", "executable": "x" * 220,
                 "memory_bytes": n} for n in range(1, 30)]
        observation = metrics.MetricObservation(
            "processes", {"total_processes_observed": 100, "shown_processes": len(rows),
                          "truncated": True, "processes": rows, "selection": "memory"},
            "structured process records", "partial", "2026-10-24T00:00:00+00:00",
        )
        snapshot = metrics.SystemMetricsSnapshot(
            "2026-10-24T00:00:00+00:00", ("processes",), (observation,), False,
        )
        encoded = metrics.format_system_snapshot(snapshot, 1600)
        decoded = json.loads(encoded)
        self.assertLessEqual(len(encoded), 1600)
        self.assertTrue(decoded["observations"][0]["value"]["truncated"])
        self.assertEqual(decoded["observations"][0]["status"], "partial")
        self.assertLess(decoded["observations"][0]["value"]["shown_processes"], 29)
        self.assertEqual(
            decoded["observations"][0]["limitation"].count(
                "Process rows were omitted to fit the selected model context."
            ),
            1,
        )

    def test_large_per_core_cpu_snapshot_is_bounded_and_marks_omitted_cores(self):
        samples = [{"logical_cpu": n, "percent": float(n % 100), "status": "ok"}
                   for n in range(200)]
        observation = metrics.MetricObservation(
            "cpu_per_core_utilization",
            {"reported_logical_cpus": 200, "shown_logical_cpus": 200,
             "truncated": False, "samples": samples},
            "percent per logical CPU", "ok", "2026-10-24T00:00:00+00:00",
        )
        snapshot = metrics.SystemMetricsSnapshot(
            "2026-10-24T00:00:00+00:00", ("cpu",), (observation,), True,
        )
        encoded = metrics.format_system_snapshot(snapshot, 1600)
        decoded = json.loads(encoded)
        result = decoded["observations"][0]
        self.assertLessEqual(len(encoded), 1600)
        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["value"]["truncated"])
        self.assertEqual(result["value"]["reported_logical_cpus"], 200)
        self.assertLess(result["value"]["shown_logical_cpus"], 200)
        self.assertEqual(
            result["limitation"].count(
                "Per-core CPU samples were omitted to fit the selected model context."
            ),
            1,
        )

    def test_formatter_fits_at_exact_boundary_and_fails_if_minimum_cannot_fit(self):
        rows = [{"pid": 7, "name": "p" * 300, "executable": None, "memory_bytes": 50}]
        observation = metrics.MetricObservation(
            "processes", {"total_processes_observed": 1, "shown_processes": 1,
                          "truncated": False, "processes": rows},
            "structured process records", "ok", "2026-10-24T00:00:00+00:00",
            limitation="Local process scan.",
        )
        snapshot = metrics.SystemMetricsSnapshot(
            "2026-10-24T00:00:00+00:00", ("processes",), (observation,), True,
        )
        full = json.dumps(metrics.asdict(snapshot), ensure_ascii=False, separators=(",", ":"))
        exact_fit = metrics.format_system_snapshot(snapshot, len(full))
        self.assertEqual(len(exact_fit), len(full))

        boundary_budget = len(full) - 1
        bounded = metrics.format_system_snapshot(snapshot, boundary_budget)
        self.assertLessEqual(len(bounded), boundary_budget)
        boundary_result = json.loads(bounded)["observations"][0]
        self.assertEqual(boundary_result["value"]["shown_processes"], 0)
        self.assertTrue(boundary_result["value"]["truncated"])
        self.assertIn("Local process scan.", boundary_result["limitation"])
        self.assertIn("Process rows were omitted", boundary_result["limitation"])

        minimal_snapshot = metrics.SystemMetricsSnapshot(
            "2026-10-24T00:00:00+00:00", (), (), True,
        )
        minimal_size = len(json.dumps(
            metrics.asdict(minimal_snapshot), ensure_ascii=False, separators=(",", ":")
        ))
        with self.assertRaisesRegex(ValueError, "exceed the safe local-model context budget"):
            metrics.format_system_snapshot(minimal_snapshot, minimal_size - 1)

    def test_cancellation_before_sampling_prevents_os_calls(self):
        backend = FakePsutil()
        stop = threading.Event()
        stop.set()
        snapshot = self.collect(backend, ("cpu", "ram"), stop_event=stop)
        self.assertTrue(snapshot.cancelled)
        self.assertEqual(snapshot.observations, ())
        backend.cpu_percent.assert_not_called()
        backend.virtual_memory.assert_not_called()

    def test_cancellation_during_cpu_sample_prevents_following_metrics(self):
        backend = FakePsutil(cpu=18.0)
        stop = threading.Event()
        backend.cpu_percent.side_effect = lambda **_kwargs: (stop.set(), 18.0)[1]
        snapshot = self.collect(backend, ("cpu", "ram"), stop_event=stop)
        self.assertTrue(snapshot.cancelled)
        self.assertFalse(backend.virtual_memory.called)


if __name__ == "__main__":
    unittest.main()
