from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.system import processes


class FakeAccessDenied(Exception):
    pass


class FakeNoSuchProcess(Exception):
    pass


class FakePsutil:
    AccessDenied = FakeAccessDenied
    NoSuchProcess = FakeNoSuchProcess


class FakeProcess:
    def __init__(self, pid, name="app.exe", executable=r"C:\Apps\app.exe", rss=4096,
                 name_error=None, exe_error=None, memory_error=None):
        self.pid = pid
        self._name = name
        self._executable = executable
        self._rss = rss
        self.name_error = name_error
        self.exe_error = exe_error
        self.memory_error = memory_error

    def name(self):
        if self.name_error:
            raise self.name_error
        return self._name

    def exe(self):
        if self.exe_error:
            raise self.exe_error
        return self._executable

    def memory_info(self):
        if self.memory_error:
            raise self.memory_error
        return SimpleNamespace(rss=self._rss)


class ProcessInspectionTests(unittest.TestCase):
    def setUp(self):
        self.psutil_patch = patch.object(processes, "psutil", FakePsutil)
        self.psutil_patch.start()
        self.addCleanup(self.psutil_patch.stop)

    def test_names_pids_paths_and_memory_are_formatted_from_observed_values(self):
        rows = processes.list_running_processes(
            lambda: [FakeProcess(203, "zeta.exe"), FakeProcess(101, "alpha.exe")]
        )
        encoded = processes.format_process_observations(rows, 4096)
        parsed = json.loads(encoded)
        self.assertEqual(parsed["total_processes_observed"], 2)
        self.assertEqual(parsed["shown_processes"], 2)
        self.assertFalse(parsed["truncated"])
        self.assertEqual(
            [(row["name"], row["pid"]) for row in parsed["processes"]],
            [("alpha.exe", 101), ("zeta.exe", 203)],
        )
        self.assertEqual(parsed["processes"][0]["executable"], r"C:\Apps\app.exe")
        self.assertEqual(parsed["processes"][0]["memory_bytes"], 4096)

    def test_access_denied_fields_are_marked_unavailable_not_invented(self):
        process = FakeProcess(
            17, name_error=FakeAccessDenied("name access denied"),
            memory_error=FakeAccessDenied("memory access denied"),
        )
        rows = processes.list_running_processes(lambda: [process])
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0].name)
        self.assertEqual(rows[0].pid, 17)
        self.assertEqual(rows[0].executable, r"C:\Apps\app.exe")
        self.assertIsNone(rows[0].memory_bytes)

    def test_disappearing_process_is_skipped(self):
        process = FakeProcess(55, exe_error=FakeNoSuchProcess("exited during inspection"))
        self.assertEqual(processes.list_running_processes(lambda: [process]), [])

    def test_enumeration_failure_preserves_the_real_error(self):
        def fail_enumeration():
            raise PermissionError("enumeration blocked")
        with self.assertRaisesRegex(processes.ProcessInspectionError, "PermissionError: enumeration blocked"):
            processes.list_running_processes(fail_enumeration)

    def test_observation_json_is_bounded_and_reports_truncation(self):
        rows = [FakeProcess(pid, name=f"process-{pid}.exe", executable="x" * 80) for pid in range(10)]
        observed = processes.list_running_processes(lambda: rows)
        encoded = processes.format_process_observations(observed, 350)
        parsed = json.loads(encoded)
        self.assertLessEqual(len(encoded), 350)
        self.assertEqual(parsed["total_processes_observed"], 10)
        self.assertTrue(parsed["truncated"])
        self.assertLess(parsed["shown_processes"], parsed["total_processes_observed"])


if __name__ == "__main__":
    unittest.main()
