"""Read-only, local inspection of Windows process metadata."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

try:
    import psutil
except ImportError:  # Keep FRIDAY usable if an optional/partial install is damaged.
    psutil = None  # type: ignore[assignment]


class ProcessInspectionError(RuntimeError):
    """A process inspection operation could not produce reliable observations."""


@dataclass(frozen=True)
class ProcessRecord:
    name: str | None
    pid: int
    executable: str | None
    memory_bytes: int | None


def _optional_field(call: Callable[[], Any], denied_type: type[BaseException]) -> Any:
    try:
        return call()
    except denied_type:
        return None


def list_running_processes(process_iter: Callable[[], Iterable[Any]] | None = None,
                           stop_event: threading.Event | None = None) -> list[ProcessRecord]:
    """Return observed process metadata only; never opens or changes a process."""
    if process_iter is None:
        if os.name != "nt":
            raise ProcessInspectionError("Windows process inspection is only available on Windows.")
        if psutil is None:
            raise ProcessInspectionError("The psutil dependency is unavailable; install the declared FRIDAY requirements.")
        process_iter = psutil.process_iter

    if psutil is None:
        raise ProcessInspectionError("The psutil dependency is unavailable; install the declared FRIDAY requirements.")

    access_denied = psutil.AccessDenied
    no_such_process = psutil.NoSuchProcess
    try:
        iterator = iter(process_iter())
    except Exception as exc:
        raise ProcessInspectionError(f"Could not enumerate Windows processes: {type(exc).__name__}: {exc}") from exc

    records: list[ProcessRecord] = []
    while True:
        if stop_event is not None and stop_event.is_set():
            break
        try:
            process = next(iterator)
        except StopIteration:
            break
        except (access_denied, no_such_process):
            continue
        except Exception as exc:
            raise ProcessInspectionError(
                f"Could not enumerate Windows processes: {type(exc).__name__}: {exc}"
            ) from exc

        try:
            pid = process.pid
            if not isinstance(pid, int) or isinstance(pid, bool) or pid < 0:
                continue
            name = _optional_field(process.name, access_denied)
            executable = _optional_field(process.exe, access_denied)
            memory_info = _optional_field(process.memory_info, access_denied)
            memory_bytes = getattr(memory_info, "rss", None) if memory_info is not None else None
            if not isinstance(memory_bytes, int) or memory_bytes < 0:
                memory_bytes = None
            records.append(ProcessRecord(
                name=name if isinstance(name, str) and name else None,
                pid=pid,
                executable=executable if isinstance(executable, str) and executable else None,
                memory_bytes=memory_bytes,
            ))
        except no_such_process:
            # The process exited between enumeration and field access; omit it.
            continue
        except access_denied:
            # PID is normally readable. If even required identity access fails, omit it.
            continue
        except Exception as exc:
            raise ProcessInspectionError(
                f"Could not read process metadata: {type(exc).__name__}: {exc}"
            ) from exc

    return sorted(records, key=lambda row: ((row.name or "").casefold(), row.pid))


def format_process_observations(records: list[ProcessRecord], max_chars: int) -> str:
    """Serialize a bounded, explicitly countable observation set for local Ollama."""
    max_chars = max(256, int(max_chars))
    payload: dict[str, Any] = {
        "total_processes_observed": len(records),
        "shown_processes": 0,
        "truncated": bool(records),
        "processes": [],
    }

    def encode() -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    for record in records:
        payload["processes"].append(asdict(record))
        payload["shown_processes"] = len(payload["processes"])
        payload["truncated"] = payload["shown_processes"] < len(records)
        if len(encode()) > max_chars:
            payload["processes"].pop()
            payload["shown_processes"] = len(payload["processes"])
            payload["truncated"] = payload["shown_processes"] < len(records)
            break

    result = encode()
    # Even a single record may exceed the model budget (e.g. an unusually long path).
    while payload["processes"] and len(result) > max_chars:
        payload["processes"].pop()
        payload["shown_processes"] = len(payload["processes"])
        payload["truncated"] = payload["shown_processes"] < len(records)
        result = encode()
    return result
