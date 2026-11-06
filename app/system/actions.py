"""Validated, read-only Windows actions for FRIDAY.

Natural-language interpretation is deliberately kept outside this module. The
executor accepts only exact ActionRequest schemas and never executes shell text.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import stat
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

MAX_DIRECTORY_ENTRIES = 200
MAX_CONFIG_BYTES = 64 * 1024
_REPARSE_POINT = 0x400
_APP_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_CHILD_NAME = re.compile(r'^[^\\/:*?"<>|\x00-\x1f]{1,255}$')
_RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_KNOWN_FOLDER_IDS = {
    "desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "downloads": "374DE290-123F-4565-9164-39C4925E467B",
}


class ActionError(Exception):
    """A user-presentable action failure that performs no fallback execution."""


class ActionValidationError(ActionError):
    pass


class ActionDenied(ActionError):
    pass


@dataclass(frozen=True)
class ActionRequest:
    action: str
    app_id: str = ""
    root_id: str = ""
    child_name: str = ""

    def to_payload(self) -> dict[str, str]:
        payload = {"action": self.action}
        if self.app_id:
            payload["app_id"] = self.app_id
        if self.root_id:
            payload["root_id"] = self.root_id
        if self.child_name:
            payload["child_name"] = self.child_name
        return payload


def parse_action_request(payload: object) -> ActionRequest:
    """Validate a closed structured request; reject unknown fields and shapes."""
    schemas = {
        "launch_application": {"action", "app_id"},
        "open_folder": {"action", "root_id"},
        "list_directory": {"action", "root_id"},
        "navigate_parent": {"action"},
        "navigate_child": {"action", "child_name"},
    }
    if not isinstance(payload, dict) or not isinstance(payload.get("action"), str):
        raise ActionValidationError("Action request must be an object with a string action.")
    action = payload["action"]
    expected = schemas.get(action)
    if expected is None or set(payload) != expected:
        raise ActionValidationError("Unknown action or unexpected/missing request fields.")
    values: dict[str, str] = {}
    for field in expected - {"action"}:
        value = payload[field]
        if not isinstance(value, str) or not value or len(value) > 255:
            raise ActionValidationError(f"{field} must be a non-empty string of at most 255 characters.")
        values[field] = value
    if action == "launch_application" and not _APP_ID.fullmatch(values["app_id"]):
        raise ActionValidationError("Application identifiers must be simple registered IDs.")
    if action in {"open_folder", "list_directory"}:
        values["root_id"] = values["root_id"].casefold()
    if action in {"open_folder", "list_directory"} and values["root_id"] not in {
        "desktop", "documents", "downloads", "friday_project", "current"
    }:
        raise ActionValidationError("Folder root is not in FRIDAY's supported root set.")
    return ActionRequest(action, **values)


@dataclass(frozen=True)
class ApplicationEntry:
    app_id: str
    display_name: str
    executable: Path


@dataclass(frozen=True)
class DirectoryEntry:
    name: str
    kind: str
    size_bytes: int | None

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "type": self.kind, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class ActionResult:
    action: str
    success: bool
    message: str
    entries: tuple[DirectoryEntry, ...] = ()
    truncated: bool = False

    @property
    def display_text(self) -> str:
        if not self.entries and not self.truncated:
            return self.message
        payload = {
            "entries": [entry.as_dict() for entry in self.entries],
            "truncated": self.truncated,
            "maximum_entries": MAX_DIRECTORY_ENTRIES,
        }
        return self.message + "\n\n" + json.dumps(payload, ensure_ascii=True, indent=2)


@dataclass(frozen=True)
class AuthorizedAction:
    request: ActionRequest
    root_id: str = ""
    root: Path | None = None
    target: Path | None = None
    application: ApplicationEntry | None = None


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    reason: str
    plan: AuthorizedAction | None = None


@dataclass(frozen=True)
class _Location:
    root_id: str
    root: Path
    path: Path


class ApplicationRegistry:
    """Load explicitly allowlisted executables from per-user FRIDAY config."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self._entries: dict[str, ApplicationEntry] | None = None

    def _create_template_if_missing(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        template = {
            "version": 1,
            "applications": [{
                "id": "notepad",
                "name": "Notepad",
                "executable": "%WINDIR%\\System32\\notepad.exe",
            }],
        }
        try:
            with self.config_path.open("x", encoding="utf-8") as stream:
                json.dump(template, stream, ensure_ascii=True, indent=2)
                stream.write("\n")
        except FileExistsError:
            pass

    def entries(self) -> dict[str, ApplicationEntry]:
        if self._entries is not None:
            return self._entries
        self._create_template_if_missing()
        try:
            if self.config_path.stat().st_size > MAX_CONFIG_BYTES:
                raise ActionError("Application allowlist exceeds its size limit.")
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ActionError(f"Cannot read the application allowlist: {type(exc).__name__}: {exc}") from exc
        if not isinstance(payload, dict) or set(payload) != {"version", "applications"}:
            raise ActionError("Application allowlist must contain only version and applications fields.")
        if payload["version"] != 1 or not isinstance(payload["applications"], list):
            raise ActionError("Unsupported application allowlist format.")

        entries: dict[str, ApplicationEntry] = {}
        for record in payload["applications"]:
            if not isinstance(record, dict) or set(record) != {"id", "name", "executable"}:
                raise ActionError("Each allowlist entry must contain exactly id, name, and executable.")
            app_id, name, raw_path = record["id"], record["name"], record["executable"]
            if (not isinstance(app_id, str) or not _APP_ID.fullmatch(app_id)
                    or app_id in entries or not isinstance(name, str) or not name.strip()
                    or len(name) > 80 or not isinstance(raw_path, str) or not raw_path.strip()):
                raise ActionError("Application allowlist contains an invalid or duplicate entry.")
            expanded = _expand_windows_variables(raw_path.strip())
            executable = Path(expanded)
            if _is_network_or_device_path(expanded) or not executable.is_absolute():
                raise ActionError("Application executable paths must be absolute local paths.")
            if executable.suffix.casefold() != ".exe":
                raise ActionError("Only explicitly registered .exe applications are supported.")
            entries[app_id] = ApplicationEntry(app_id, name.strip(), executable)
        self._entries = entries
        return entries

    def get(self, app_id: str) -> ApplicationEntry | None:
        return self.entries().get(app_id)


def _expand_windows_variables(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        replacement = os.environ.get(key)
        if replacement is None:
            raise ActionError(f"Application path uses an unset environment variable: {key}.")
        return replacement
    return re.sub(r"%([^%]+)%", replace, value)


def _is_network_or_device_path(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return normalized.startswith("\\\\")


def _is_reparse_point(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        if path.is_symlink():
            return True
    except OSError:
        return True
    if bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT):
        return True
    junction_check = getattr(path, "is_junction", None)
    try:
        return bool(junction_check and junction_check())
    except OSError:
        return True


def _path_within(root: Path, target: Path) -> bool:
    try:
        root_norm = os.path.normcase(os.path.abspath(str(root)))
        target_norm = os.path.normcase(os.path.abspath(str(target)))
        return os.path.commonpath((root_norm, target_norm)) == root_norm
    except (OSError, ValueError):
        return False


def _validated_directory(path: Path) -> Path:
    if _is_network_or_device_path(str(path)) or not path.is_absolute():
        raise ActionDenied("Network, device, and relative paths are not authorized.")
    if _is_reparse_point(path):
        raise ActionDenied("A symlink or junction cannot be used as an authorized folder root.")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ActionDenied(f"Folder is unavailable: {type(exc).__name__}: {exc}") from exc
    if not resolved.is_dir():
        raise ActionDenied("The authorized target is not a directory.")
    return resolved


def _validated_child(root: Path, target: Path) -> Path:
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ActionDenied("Target is outside its authorized folder root.") from exc
    current = root
    for component in relative.parts:
        current = current / component
        if _is_reparse_point(current):
            raise ActionDenied("Symlinks and junctions are not followed during navigation.")
    try:
        resolved = target.resolve(strict=True)
    except OSError as exc:
        raise ActionDenied(f"Folder is unavailable: {type(exc).__name__}: {exc}") from exc
    if not _path_within(root, resolved):
        raise ActionDenied("Resolved target escaped its authorized folder root.")
    if not resolved.is_dir():
        raise ActionDenied("The requested target is not a directory.")
    return resolved


def _get_windows_known_folder(root_id: str) -> Path:
    if os.name != "nt":
        raise ActionDenied("Windows known folders are unavailable on this platform.")
    guid_text = _KNOWN_FOLDER_IDS.get(root_id)
    if guid_text is None:
        raise ActionDenied("Unknown Windows known folder.")

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
            ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8),
        ]

    guid = GUID.from_buffer_copy(uuid.UUID(guid_text).bytes_le)
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    ole32 = ctypes.WinDLL("ole32", use_last_error=True)
    get_path = shell32.SHGetKnownFolderPath
    get_path.argtypes = [ctypes.POINTER(GUID), ctypes.c_uint32, ctypes.c_void_p,
                         ctypes.POINTER(ctypes.c_void_p)]
    get_path.restype = ctypes.c_long
    free = ole32.CoTaskMemFree
    free.argtypes = [ctypes.c_void_p]
    free.restype = None
    buffer = ctypes.c_void_p()
    result = get_path(ctypes.byref(guid), 0, None, ctypes.byref(buffer))
    if result != 0:
        raise ActionDenied(f"Windows could not resolve the {root_id} folder (HRESULT {result}).")
    try:
        return Path(ctypes.wstring_at(buffer.value))
    finally:
        free(buffer)


def _default_project_root() -> Path | None:
    configured = os.environ.get("FRIDAY_PROJECT_ROOT")
    if configured:
        return Path(os.path.expandvars(os.path.expanduser(configured)))
    if getattr(sys, "frozen", False):
        return None
    candidate = Path(__file__).resolve().parents[2]
    if _is_friday_project(candidate):
        return candidate
    return None


def _is_friday_project(path: Path) -> bool:
    return all((path / part).is_file() for part in ("app/main.py", "README.md", "requirements.txt"))


class ActionService:
    """Policy and executor for a small, explicit Windows action allowlist."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        root_overrides: Mapping[str, str | Path] | None = None,
        known_folder_resolver: Callable[[str], Path] | None = None,
        application_registry: ApplicationRegistry | None = None,
        process_launcher: Callable[..., object] | None = None,
        folder_opener: Callable[[Path], None] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.root_overrides = {key: Path(value) for key, value in (root_overrides or {}).items()}
        self.known_folder_resolver = known_folder_resolver or _get_windows_known_folder
        self.registry = application_registry or ApplicationRegistry(self.data_dir / "applications.json")
        self.process_launcher = process_launcher or _launch_process
        self.folder_opener = folder_opener or _open_folder_in_explorer
        self.current_location: _Location | None = None
        self._audit_lock = threading.Lock()

    def record_rejected_request(self) -> None:
        self._record_audit("invalid_request", "denied", "rejected")

    def authorize(self, request: ActionRequest) -> AuthorizationDecision:
        try:
            request = parse_action_request(request.to_payload())
            if request.action == "launch_application":
                entry = self.registry.get(request.app_id)
                if entry is None:
                    raise ActionDenied(
                        f"'{request.app_id}' is not approved. Add its absolute executable path to "
                        f"{self.registry.config_path} and restart FRIDAY."
                    )
                try:
                    executable = _validated_executable(entry.executable)
                except ActionDenied as exc:
                    raise ActionDenied(
                        f"{exc} Update the approved executable path in {self.registry.config_path}."
                    ) from exc
                return AuthorizationDecision(True, "Application is allowlisted.",
                                             AuthorizedAction(request, application=ApplicationEntry(entry.app_id, entry.display_name, executable)))

            if request.action in {"open_folder", "list_directory"}:
                if request.root_id == "current":
                    if self.current_location is None:
                        raise ActionDenied("No current folder. Open an approved folder first.")
                    root_id = self.current_location.root_id
                    root = self._resolve_root(root_id)
                    target = _validated_child(root, self.current_location.path)
                else:
                    root_id = request.root_id
                    root = self._resolve_root(root_id)
                    target = root
                return AuthorizationDecision(True, "Target is inside an approved folder root.",
                                             AuthorizedAction(request, root_id, root, target))

            if request.action in {"navigate_parent", "navigate_child"}:
                location = self.current_location
                if location is None:
                    raise ActionDenied("No current folder. Open or list an approved folder first.")
                root = self._resolve_root(location.root_id)
                current = _validated_child(root, location.path)
                if request.action == "navigate_parent":
                    if current == root:
                        raise ActionDenied("Navigation cannot go above the approved folder root.")
                    target = _validated_child(root, current.parent)
                else:
                    name = request.child_name
                    if (not _CHILD_NAME.fullmatch(name) or name in {".", ".."}
                            or name.endswith((".", " "))
                            or name.split(".", 1)[0].upper() in _RESERVED_WINDOWS_NAMES):
                        raise ActionDenied("Child folder name is not a safe single directory name.")
                    target = _validated_child(root, current / name)
                return AuthorizationDecision(True, "Navigation remains inside the approved root.",
                                             AuthorizedAction(request, location.root_id, root, target))
            raise ActionDenied("Unknown action type.")
        except (ActionError, AttributeError, TypeError, OSError, ValueError) as exc:
            return AuthorizationDecision(False, str(exc), None)

    def execute(self, request: ActionRequest) -> ActionResult:
        try:
            request = parse_action_request(request.to_payload())
        except (ActionError, AttributeError, TypeError, OSError, ValueError) as exc:
            self._record_audit("invalid_request", "denied", "rejected")
            return ActionResult("invalid_request", False, f"No action was run: {exc}")

        decision = self.authorize(request)
        if not decision.allowed or decision.plan is None:
            self._record_audit(request.action, "denied", "denied")
            return ActionResult(request.action, False, f"No action was run: {decision.reason}")
        try:
            result = self._execute_authorized(decision.plan)
        except Exception as exc:
            self._record_audit(request.action, "allowed", "failed")
            return ActionResult(request.action, False,
                                f"{request.action.replace('_', ' ').capitalize()} failed: {type(exc).__name__}: {exc}")
        self._record_audit(request.action, "allowed", "success" if result.success else "failed")
        return result

    def _resolve_root(self, root_id: str) -> Path:
        if root_id in self.root_overrides:
            candidate = self.root_overrides[root_id]
        elif root_id in _KNOWN_FOLDER_IDS:
            candidate = self.known_folder_resolver(root_id)
        elif root_id == "friday_project":
            candidate = _default_project_root()
            if candidate is None or not _is_friday_project(candidate):
                raise ActionDenied(
                    "FRIDAY's project folder is not configured. Set FRIDAY_PROJECT_ROOT to the "
                    "local project directory containing app/main.py, README.md, and requirements.txt."
                )
        else:
            raise ActionDenied("Folder root is not approved.")
        return _validated_directory(Path(candidate))

    def _execute_authorized(self, plan: AuthorizedAction) -> ActionResult:
        request = plan.request
        if request.action == "launch_application":
            if plan.application is None:
                raise ActionDenied("No approved application was resolved.")
            executable = _validated_executable(plan.application.executable)
            self.process_launcher([str(executable)], shell=False, close_fds=True)
            return ActionResult(request.action, True, f"Started approved application: {plan.application.display_name}.")

        if plan.root is None or plan.target is None:
            raise ActionDenied("No authorized folder target was resolved.")
        target = _validated_child(plan.root, plan.target)
        if request.action == "open_folder":
            self.folder_opener(target)
            self.current_location = _Location(plan.root_id, plan.root, target)
            return ActionResult(request.action, True, f"Opened approved folder in File Explorer: {target}")
        if request.action in {"navigate_parent", "navigate_child"}:
            self.current_location = _Location(plan.root_id, plan.root, target)

        entries, truncated = _list_directory(target)
        self.current_location = _Location(plan.root_id, plan.root, target)
        qualifier = " (first entries only; more were not inspected)" if truncated else ""
        message = f"Listed {len(entries)} direct entries in {target}{qualifier}."
        if not entries and not truncated:
            message = f"The authorized folder is empty: {target}"
        return ActionResult(request.action, True, message, tuple(entries), truncated)

    def _record_audit(self, action: str, authorization: str, execution_result: str) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "action": action,
            "authorization": authorization,
            "execution_result": execution_result,
        }
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with self._audit_lock, (self.data_dir / "actions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=True) + "\n")
        except OSError:
            # Auditing failure must not cause a second or misleading OS action.
            pass


def _validated_executable(path: Path) -> Path:
    if _is_network_or_device_path(str(path)) or not path.is_absolute():
        raise ActionDenied("Executable must be an absolute local path.")
    if _is_reparse_point(path):
        raise ActionDenied("A symlink or junction cannot be launched as an approved executable.")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ActionDenied(f"Approved executable is unavailable: {type(exc).__name__}: {exc}") from exc
    if not resolved.is_file() or resolved.suffix.casefold() != ".exe":
        raise ActionDenied("Approved executable is not a local .exe file.")
    return resolved


def _launch_process(command: list[str], *, shell: bool, close_fds: bool) -> object:
    return subprocess.Popen(command, shell=shell, close_fds=close_fds)


def _open_folder_in_explorer(path: Path) -> None:
    if os.name != "nt" or not hasattr(os, "startfile"):
        raise ActionError("Opening folders in File Explorer is supported only on Windows.")
    os.startfile(str(path))


def _list_directory(path: Path) -> tuple[list[DirectoryEntry], bool]:
    entries: list[DirectoryEntry] = []
    truncated = False
    with os.scandir(path) as iterator:
        for item in iterator:
            if len(entries) >= MAX_DIRECTORY_ENTRIES:
                truncated = True
                break
            try:
                info = item.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            attrs = getattr(info, "st_file_attributes", 0)
            is_link = item.is_symlink() or bool(attrs & _REPARSE_POINT)
            if is_link:
                kind, size = "link (not followed)", None
            elif stat.S_ISDIR(info.st_mode):
                kind, size = "folder", None
            elif stat.S_ISREG(info.st_mode):
                kind, size = "file", info.st_size
            else:
                kind, size = "other", None
            entries.append(DirectoryEntry(item.name, kind, size))
    entries.sort(key=lambda entry: entry.name.casefold())
    return entries, truncated
