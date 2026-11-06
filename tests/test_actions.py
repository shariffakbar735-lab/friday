from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.system import actions
from app.system.action_intents import ActionIntentError, parse_action_text
from app.system.actions import (
    ActionService,
    ApplicationRegistry,
    parse_action_request,
)


class WindowsActionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "Downloads"
        self.root.mkdir()
        self.appdata = self.base / "local" / "FRIDAY"
        self.executable = self.base / "notepad.exe"
        self.executable.write_bytes(b"not a real executable; launcher is mocked")
        self.config = self.base / "applications.json"
        self.config.write_text(json.dumps({
            "version": 1,
            "applications": [{
                "id": "notepad", "name": "Notepad", "executable": str(self.executable),
            }],
        }), encoding="utf-8")
        self.launcher = Mock()
        self.opener = Mock()
        self.service = ActionService(
            self.appdata,
            root_overrides={"downloads": self.root},
            application_registry=ApplicationRegistry(self.config),
            process_launcher=self.launcher,
            folder_opener=self.opener,
        )

    def test_structured_requests_are_closed_and_root_ids_are_case_insensitive(self):
        request = parse_action_request({"action": "open_folder", "root_id": "DOWNLOADS"})
        self.assertEqual(request.root_id, "downloads")
        for malformed in (
            {"action": "open_folder", "root_id": "downloads", "extra": "x"},
            {"action": "open_folder", "root_id": "C:\\"},
            {"action": "navigate_parent", "child_name": "ignored"},
            {"action": "launch_application", "app_id": "C:\\Windows\\notepad.exe"},
            None,
        ):
            with self.subTest(malformed=malformed), self.assertRaises(actions.ActionValidationError):
                parse_action_request(malformed)

    def test_supported_natural_language_maps_to_structured_actions(self):
        cases = {
            "Open Notepad": ("launch_application", "notepad", "", ""),
            "Open my Downloads folder": ("open_folder", "", "downloads", ""),
            "List files in my FRIDAY project directory": ("list_directory", "", "friday_project", ""),
            "List files": ("list_directory", "", "current", ""),
            "Go to the parent folder": ("navigate_parent", "", "", ""),
            "Open folder Reports": ("navigate_child", "", "", "Reports"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                request = parse_action_text(text)
                self.assertIsNotNone(request)
                self.assertEqual(
                    (request.action, request.app_id, request.root_id, request.child_name), expected
                )
        self.assertEqual(parse_action_text("OPEN MY DOWNLOADS FOLDER").root_id, "downloads")
        self.assertIsNone(parse_action_text("Explain how Windows known folders work."))
        self.assertIsNone(parse_action_text("Ordinary chat " + ("about local files " * 40)))
        with self.assertRaises(ActionIntentError):
            parse_action_text("Open " + ("x" * 600))

    def test_ambiguous_paths_and_unsupported_actions_do_not_fall_through_to_chat(self):
        for text in (
            r"Open C:\Windows\System32\cmd.exe",
            r"Open \\server\share",
            "Open an unknown directory",
            "Run PowerShell -Command whoami",
        ):
            with self.subTest(text=text), self.assertRaises(ActionIntentError):
                parse_action_text(text)

    def test_approved_application_launches_only_registered_executable_without_shell(self):
        request = parse_action_request({"action": "launch_application", "app_id": "notepad"})
        result = self.service.execute(request)
        self.assertTrue(result.success)
        self.launcher.assert_called_once_with(
            [str(self.executable.resolve())], shell=False, close_fds=True
        )
        self.opener.assert_not_called()

    def test_unregistered_application_and_missing_executable_are_denied(self):
        unknown = self.service.execute(
            parse_action_request({"action": "launch_application", "app_id": "calculator"})
        )
        self.assertFalse(unknown.success)
        self.launcher.assert_not_called()

        missing_config = self.base / "missing-app.json"
        missing_path = self.base / "missing.exe"
        missing_config.write_text(json.dumps({
            "version": 1,
            "applications": [{"id": "editor", "name": "Editor", "executable": str(missing_path)}],
        }), encoding="utf-8")
        missing_service = ActionService(
            self.appdata, application_registry=ApplicationRegistry(missing_config),
            process_launcher=self.launcher,
        )
        missing = missing_service.execute(
            parse_action_request({"action": "launch_application", "app_id": "editor"})
        )
        self.assertFalse(missing.success)
        self.assertIn("unavailable", missing.message)
        self.launcher.assert_not_called()

    def test_application_configuration_rejects_extra_fields_and_relative_paths(self):
        for executable in ("notepad.exe", r"\\server\share\tool.exe"):
            config = self.base / ("bad-" + str(len(executable)) + ".json")
            config.write_text(json.dumps({
                "version": 1,
                "applications": [{"id": "tool", "name": "Tool", "executable": executable}],
            }), encoding="utf-8")
            with self.subTest(executable=executable):
                registry = ApplicationRegistry(config)
                with self.assertRaises(actions.ActionError):
                    registry.get("tool")

    def test_open_approved_folder_uses_mocked_explorer_integration(self):
        result = self.service.execute(
            parse_action_request({"action": "open_folder", "root_id": "downloads"})
        )
        self.assertTrue(result.success)
        self.opener.assert_called_once_with(self.root.resolve())
        self.assertEqual(self.service.current_location.path, self.root.resolve())

    def test_directory_listing_returns_metadata_only_and_does_not_read_file_contents(self):
        secret = self.root / "private-name.txt"
        secret.write_text("DO NOT READ THIS CONTENT", encoding="utf-8")
        (self.root / "subfolder").mkdir()
        result = self.service.execute(
            parse_action_request({"action": "list_directory", "root_id": "downloads"})
        )
        self.assertTrue(result.success)
        by_name = {entry.name: entry for entry in result.entries}
        self.assertEqual(by_name["private-name.txt"].kind, "file")
        self.assertEqual(by_name["private-name.txt"].size_bytes, len("DO NOT READ THIS CONTENT"))
        self.assertEqual(by_name["subfolder"].kind, "folder")
        self.assertNotIn("DO NOT READ THIS CONTENT", result.display_text)
        self.assertIn('"size_bytes"', result.display_text)

    def test_parent_child_navigation_stays_inside_approved_root(self):
        (self.root / "Reports").mkdir()
        listed = self.service.execute(
            parse_action_request({"action": "list_directory", "root_id": "downloads"})
        )
        self.assertTrue(listed.success)
        child = self.service.execute(
            parse_action_request({"action": "navigate_child", "child_name": "Reports"})
        )
        self.assertTrue(child.success)
        self.assertEqual(self.service.current_location.path, (self.root / "Reports").resolve())
        parent = self.service.execute(parse_action_request({"action": "navigate_parent"}))
        self.assertTrue(parent.success)
        self.assertEqual(self.service.current_location.path, self.root.resolve())
        denied = self.service.execute(parse_action_request({"action": "navigate_parent"}))
        self.assertFalse(denied.success)
        self.assertIn("above the approved", denied.message)

    def test_traversal_and_alternate_separators_are_rejected(self):
        self.service.execute(parse_action_request({"action": "list_directory", "root_id": "downloads"}))
        for name in ("..", "../outside", r"..\outside", "Reports/child", "bad:name"):
            with self.subTest(name=name):
                result = self.service.execute(
                    parse_action_request({"action": "navigate_child", "child_name": name})
                )
                self.assertFalse(result.success)
        self.assertEqual(self.service.current_location.path, self.root.resolve())

    def test_symlink_or_junction_child_is_rejected_without_following_it(self):
        outside = self.base / "outside"
        outside.mkdir()
        link = self.root / "escape"
        link.mkdir()
        self.service.execute(parse_action_request({"action": "list_directory", "root_id": "downloads"}))
        real_check = actions._is_reparse_point

        def reparse_check(path):
            return Path(path) == link or real_check(Path(path))

        with patch("app.system.actions._is_reparse_point", side_effect=reparse_check):
            result = self.service.execute(
                parse_action_request({"action": "navigate_child", "child_name": "escape"})
            )
        self.assertFalse(result.success)
        self.assertIn("Symlinks and junctions", result.message)
        self.assertEqual(self.service.current_location.path, self.root.resolve())

    def test_directory_listing_is_bounded_and_marks_omitted_entries(self):
        for index in range(actions.MAX_DIRECTORY_ENTRIES + 5):
            (self.root / f"file-{index:03}.txt").touch()
        result = self.service.execute(
            parse_action_request({"action": "list_directory", "root_id": "downloads"})
        )
        self.assertTrue(result.success)
        self.assertEqual(len(result.entries), actions.MAX_DIRECTORY_ENTRIES)
        self.assertTrue(result.truncated)
        self.assertIn("more were not inspected", result.message)

    def test_unc_device_and_nonlocal_roots_are_denied(self):
        self.assertTrue(actions._is_network_or_device_path(r"\\server\share"))
        self.assertTrue(actions._is_network_or_device_path(r"\\?\C:\Windows"))
        unc_service = ActionService(
            self.appdata, root_overrides={"downloads": Path(r"\\server\share")},
        )
        result = unc_service.execute(
            parse_action_request({"action": "list_directory", "root_id": "downloads"})
        )
        self.assertFalse(result.success)
        self.assertIn("Network, device", result.message)

    def test_audit_records_outcome_without_paths_or_directory_names(self):
        (self.root / "private-listing-name.txt").touch()
        self.service.execute(
            parse_action_request({"action": "list_directory", "root_id": "downloads"})
        )
        log_text = (self.appdata / "actions.jsonl").read_text(encoding="utf-8")
        record = json.loads(log_text.splitlines()[0])
        self.assertEqual(set(record), {"timestamp", "action", "authorization", "execution_result"})
        self.assertEqual(record["action"], "list_directory")
        self.assertEqual(record["authorization"], "allowed")
        self.assertEqual(record["execution_result"], "success")
        self.assertNotIn(str(self.root), log_text)
        self.assertNotIn("private-listing-name.txt", log_text)

    def test_permission_decision_is_separate_from_execution(self):
        request = parse_action_request({"action": "open_folder", "root_id": "downloads"})
        decision = self.service.authorize(request)
        self.assertTrue(decision.allowed)
        self.assertIsNotNone(decision.plan)
        self.opener.assert_not_called()
        unavailable_service = ActionService(
            self.appdata, root_overrides={"downloads": self.base / "not-configured"}
        )
        denied = unavailable_service.authorize(
            parse_action_request({"action": "open_folder", "root_id": "downloads"})
        )
        self.assertFalse(denied.allowed)
        self.assertIsNone(denied.plan)


if __name__ == "__main__":
    unittest.main()
