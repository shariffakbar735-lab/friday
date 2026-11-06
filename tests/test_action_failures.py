from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.system.actions import ActionError, ActionService, ApplicationRegistry, parse_action_request


class ActionFailureTests(unittest.TestCase):
    def test_missing_folder_returns_controlled_denial(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            service = ActionService(Path(directory) / "data", root_overrides={"downloads": missing})
            result = service.execute(parse_action_request({"action": "list_directory", "root_id": "downloads"}))
            self.assertFalse(result.success)
            self.assertIn("Folder is unavailable", result.message)

    def test_inaccessible_directory_returns_real_error_without_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Downloads"
            root.mkdir()
            service = ActionService(Path(directory) / "data", root_overrides={"downloads": root})
            with patch("app.system.actions.os.scandir", side_effect=PermissionError("listing denied")):
                result = service.execute(parse_action_request({"action": "list_directory", "root_id": "downloads"}))
            self.assertFalse(result.success)
            self.assertIn("PermissionError: listing denied", result.message)

    def test_registry_rejects_unexpected_fields_duplicate_ids_and_oversized_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "applications.json"
            valid_entry = {"id": "editor", "name": "Editor", "executable": str(Path(directory) / "editor.exe")}
            for payload in (
                {"version": 1, "applications": [{**valid_entry, "arguments": ["--unsafe"]}]},
                {"version": 1, "applications": [valid_entry, valid_entry]},
            ):
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(payload=payload), self.assertRaises(ActionError):
                    ApplicationRegistry(path).entries()
            path.write_text(" " * 65537, encoding="utf-8")
            with self.assertRaisesRegex(ActionError, "size limit"):
                ApplicationRegistry(path).entries()


if __name__ == "__main__":
    unittest.main()
