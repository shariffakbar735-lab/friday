"""Read-only local system measurements for explicit FRIDAY queries."""
from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

try:
    import psutil
except ImportError:  # Keep history/UI available if a partial install is damaged.
    psutil = None  # type: ignore[assignment]

from app.system.processes import (
    ProcessInspectionError,
    format_process_observations,
    list_running_processes,
)


SUPPORTED_METRICS = frozenset({"cpu", "ram", "disk", "battery", "processes"})
MAX_DISK_PARTITIONS = 16
MAX_PROCESS_RECORDS = 40
MAX_CPU_CORE_SAMPLES = 128
BYTES_PER_GIB = 1024 ** 3


def bytes_to_gib(size_bytes: int) -> float:
    """Convert a non-negative byte count to binary GiB without changing the raw value."""
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
        raise ValueError("A byte count must be a non-negative integer.")
    return size_bytes / BYTES_PER_GIB


def _normalize_drive_letter(value: str) -> str | None:
    candidate = value.strip().replace("/", "\\")
    if re.fullmatch(r"[A-Za-z]:(?:\\)?", candidate):
        return candidate[0].upper() + ":\\"
    return None


@dataclass(frozen=True)
class MetricObservation:
    """One observed value, or a transparent unavailable/error result."""

    metric: str
    value: Any
    unit: str | None
    status: str
    observed_at: str
    limitation: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class SystemMetricsSnapshot:
    """Ephemeral collection result; callers should not persist this raw object."""

    observed_at: str
    requested: tuple[str, ...]
    observations: tuple[MetricObservation, ...]
    complete: bool
    cancelled: bool = False

    @property
    def has_reportable_observation(self) -> bool:
        return any(item.status != "error" for item in self.observations)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _error_text(exc: BaseException) -> str:
    text = str(exc).strip() or "No additional error detail was provided."
    return f"{type(exc).__name__}: {text[:400]}"


def _observation(metric: str, value: Any, unit: str | None, status: str,
                 observed_at: str, *, limitation: str | None = None,
                 error: str | None = None) -> MetricObservation:
    return MetricObservation(metric, value, unit, status, observed_at, limitation, error)


def _collect_cpu(backend: Any, timestamp: str, stop_event: threading.Event | None,
                 interval: float) -> list[MetricObservation]:
    results: list[MetricObservation] = []
    sample_interval = max(0.05, float(interval))
    try:
        if backend is None:
            raise RuntimeError("psutil is unavailable; install the declared FRIDAY requirements.")
        if stop_event is not None and stop_event.is_set():
            return results
        # A short blocking sample runs only in ChatWorker, never the GUI thread.
        sample = backend.cpu_percent(interval=sample_interval, percpu=True)
        if sample is None:
            results.append(_observation(
                "cpu_utilization", None, "percent", "warming_up", timestamp,
                limitation="The operating system has not produced a CPU sample yet.",
            ))
            results.append(_observation(
                "cpu_per_core_utilization", None, "percent per logical CPU", "warming_up", timestamp,
                limitation="Per-core utilization is not available until the operating system produces a sample.",
            ))
        elif isinstance(sample, (list, tuple)):
            core_samples: list[dict[str, Any]] = []
            valid_values: list[float] = []
            for index, value in enumerate(sample):
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(float(value)) or not 0 <= float(value) <= 100):
                    core_samples.append({
                        "logical_cpu": index, "percent": None, "status": "error",
                        "error": "The operating system returned an invalid per-core percentage.",
                    })
                else:
                    measured = float(value)
                    valid_values.append(measured)
                    core_samples.append({"logical_cpu": index, "percent": measured, "status": "ok"})
            invalid_count = len(core_samples) - len(valid_values)
            shown_core_samples = core_samples[:MAX_CPU_CORE_SAMPLES]
            core_truncated = len(core_samples) > len(shown_core_samples)
            if not core_samples:
                results.append(_observation(
                    "cpu_utilization", None, "percent", "error", timestamp,
                    error="The operating system returned no per-core CPU samples.",
                ))
                results.append(_observation(
                    "cpu_per_core_utilization",
                    {"reported_logical_cpus": 0, "shown_logical_cpus": 0,
                     "truncated": False, "samples": []},
                    "percent per logical CPU", "error", timestamp,
                    error="The operating system returned no per-core CPU samples.",
                ))
            else:
                results.append(_observation(
                    "cpu_per_core_utilization",
                    {"reported_logical_cpus": len(core_samples),
                     "shown_logical_cpus": len(shown_core_samples),
                     "truncated": core_truncated,
                     "samples": shown_core_samples},
                    "percent per logical CPU",
                    "error" if invalid_count and not valid_values else "partial" if invalid_count or core_truncated else "ok",
                    timestamp,
                    limitation=("Logical CPU indexes are zero-based; only the first "
                                f"{len(shown_core_samples)} of {len(core_samples)} samples are shown."
                                if core_truncated else "Logical CPU indexes are zero-based."),
                    error=f"{invalid_count} per-core sample(s) were invalid." if invalid_count else None,
                ))
                if invalid_count:
                    results.append(_observation(
                        "cpu_utilization", None, "percent", "error", timestamp,
                        error="Aggregate CPU utilization was not calculated because one or more per-core samples were invalid.",
                    ))
                else:
                    results.append(_observation(
                        "cpu_utilization", sum(valid_values) / len(valid_values), "percent", "ok", timestamp,
                        limitation=(f"Arithmetic mean of {len(valid_values)} per-core samples collected over "
                                    f"{sample_interval:g} seconds."),
                    ))
        else:
            if (isinstance(sample, bool) or not isinstance(sample, (int, float))
                    or not math.isfinite(float(sample)) or not 0 <= float(sample) <= 100):
                results.append(_observation(
                    "cpu_utilization", None, "percent", "error", timestamp,
                    error="The CPU sampler returned an invalid percentage.",
                ))
            else:
                results.append(_observation(
                    "cpu_utilization", float(sample), "percent", "ok", timestamp,
                    limitation=f"Sampled over {sample_interval:g} seconds.",
                ))
            results.append(_observation(
                "cpu_per_core_utilization", None, "percent per logical CPU", "unavailable", timestamp,
                limitation="This sampler did not return per-core readings.",
            ))
    except Exception as exc:
        results.append(_observation("cpu_utilization", None, "percent", "error", timestamp,
                                    error=_error_text(exc)))
        results.append(_observation("cpu_per_core_utilization", None, "percent per logical CPU", "error",
                                    timestamp, error=_error_text(exc)))

    if stop_event is not None and stop_event.is_set():
        return results
    try:
        if backend is None:
            raise RuntimeError("psutil is unavailable; install the declared FRIDAY requirements.")
        cores = backend.cpu_count(logical=True)
        if cores is None:
            results.append(_observation("cpu_logical_core_count", None, "logical cores", "unavailable",
                                        timestamp, limitation="The operating system did not report a logical core count."))
        elif isinstance(cores, bool) or not isinstance(cores, int) or cores <= 0:
            results.append(_observation("cpu_logical_core_count", None, "logical cores", "error",
                                        timestamp, error="The CPU sampler returned an invalid logical core count."))
        else:
            results.append(_observation("cpu_logical_core_count", cores, "logical cores", "ok", timestamp))
    except Exception as exc:
        results.append(_observation("cpu_logical_core_count", None, "logical cores", "error",
                                    timestamp, error=_error_text(exc)))
    return results


def _collect_ram(backend: Any, timestamp: str) -> list[MetricObservation]:
    try:
        if backend is None:
            raise RuntimeError("psutil is unavailable; install the declared FRIDAY requirements.")
        memory = backend.virtual_memory()
        total = memory.total
        available = memory.available
        if (isinstance(total, bool) or not isinstance(total, int) or total <= 0
                or isinstance(available, bool) or not isinstance(available, int)
                or available < 0 or available > total):
            raise ValueError("The operating system returned invalid physical-memory totals.")
        used = total - available
        percent = (used / total) * 100.0
        return [
            _observation("physical_ram_total", total, "bytes", "ok", timestamp),
            _observation("physical_ram_used", used, "bytes", "ok", timestamp,
                         limitation="Used is calculated as total physical RAM minus OS-reported available physical RAM."),
            _observation("physical_ram_available", available, "bytes", "ok", timestamp),
            _observation("physical_ram_percent_used", percent, "percent", "ok", timestamp),
        ]
    except Exception as exc:
        return [_observation("physical_ram", None, None, "error", timestamp,
                             error=_error_text(exc))]


def _collect_disks(backend: Any, timestamp: str, stop_event: threading.Event | None,
                   max_partitions: int, requested_mountpoint: str | None = None) -> list[MetricObservation]:
    normalized_request = None
    if requested_mountpoint is not None:
        normalized_request = _normalize_drive_letter(requested_mountpoint)
        if normalized_request is None:
            return [_observation(
                "disk_partitions", None, None, "error", timestamp,
                error="The requested disk selector is not a drive letter such as C:.",
            )]
    try:
        if backend is None:
            raise RuntimeError("psutil is unavailable; install the declared FRIDAY requirements.")
        partitions = backend.disk_partitions(all=False)
    except Exception as exc:
        return [_observation("disk_partitions", None, "raw bytes, binary GiB, and percent", "error",
                             timestamp, error=_error_text(exc))]

    unique: list[Any] = []
    seen: set[str] = set()
    for partition in partitions:
        mountpoint = getattr(partition, "mountpoint", None)
        if isinstance(mountpoint, str) and mountpoint and mountpoint.casefold() not in seen:
            seen.add(mountpoint.casefold())
            unique.append(partition)
    if normalized_request is not None:
        unique = [partition for partition in unique
                  if _normalize_drive_letter(partition.mountpoint or "") == normalized_request]
    cap = max(1, int(max_partitions))
    partition_truncated = len(unique) > cap
    selected = unique[:cap]
    drives: list[dict[str, Any]] = []
    failed = 0
    for partition in selected:
        if stop_event is not None and stop_event.is_set():
            break
        mountpoint = partition.mountpoint
        try:
            usage = backend.disk_usage(mountpoint)
            total, used, free = usage.total, usage.used, usage.free
            if (any(isinstance(number, bool) or not isinstance(number, int)
                   for number in (total, used, free)) or total <= 0
                    or used < 0 or used > total or free < 0 or used + free != total):
                raise ValueError("The filesystem returned invalid capacity values.")
            drives.append({
                "mountpoint": mountpoint,
                "total_bytes": total,
                "used_bytes": used,
                "free_bytes": free,
                "total_gib": bytes_to_gib(total),
                "used_gib": bytes_to_gib(used),
                "free_gib": bytes_to_gib(free),
                "percent_used": (used / total) * 100.0,
                "status": "ok",
            })
        except Exception as exc:
            failed += 1
            drives.append({"mountpoint": mountpoint, "status": "error", "error": _error_text(exc)})

    if not unique and normalized_request is not None:
        return [_observation("disk_partitions", {
            "requested_mountpoint": normalized_request,
            "total_drives_observed": 0, "shown_drives": 0,
            "accessible_drives": 0, "truncated": False, "drives": [],
        }, "structured bytes, GiB, and percent", "not_detected", timestamp,
            limitation=f"The requested local drive {normalized_request} was not returned by the operating system.")]
    if not unique:
        return [_observation("disk_partitions", {
            "total_drives_observed": 0, "shown_drives": 0, "truncated": False, "drives": [],
        }, "raw bytes, binary GiB, and percent", "unavailable", timestamp,
            limitation="No local partitions were returned by the operating system.")]

    was_cancelled = stop_event is not None and stop_event.is_set()
    truncated = partition_truncated or was_cancelled
    shown = sum(1 for drive in drives if drive["status"] == "ok")
    status = "error" if failed and shown == 0 and not truncated else (
        "partial" if failed or truncated else "ok"
    )
    limitation_parts = []
    if failed:
        limitation_parts.append(f"{failed} drive(s) could not be read; their errors are included.")
    if partition_truncated:
        limitation_parts.append(f"Partition listing bounded to {cap} of {len(unique)} reported drives.")
    if was_cancelled:
        limitation_parts.append("Drive collection stopped at the user's request.")
    return [_observation("disk_partitions", {
        "total_drives_observed": len(unique),
        **({"requested_mountpoint": normalized_request} if normalized_request else {}),
        "shown_drives": len(drives),
        "accessible_drives": shown,
        "truncated": truncated,
        "drives": drives,
    }, "raw bytes, binary GiB (1 GiB = 1024^3 bytes), and percent", status, timestamp,
        limitation=" ".join(limitation_parts) or None,
        error="All reported drives were inaccessible." if failed and shown == 0 else None)]


def _collect_battery(backend: Any, timestamp: str) -> list[MetricObservation]:
    try:
        if backend is None:
            raise RuntimeError("psutil is unavailable; install the declared FRIDAY requirements.")
        battery = backend.sensors_battery()
        if battery is None:
            return [_observation("battery", None, None, "not_detected", timestamp,
                                 limitation="No battery was reported; this does not mean 0 percent charge.")]
        raw_percent = battery.percent
        percent = None
        percent_error = None
        if (isinstance(raw_percent, bool) or not isinstance(raw_percent, (int, float))
                or not math.isfinite(float(raw_percent)) or not 0 <= float(raw_percent) <= 100):
            percent_error = "The operating system returned an invalid battery percentage."
        else:
            percent = float(raw_percent)

        plugged = getattr(battery, "power_plugged", None)
        power_connection = "plugged_in" if plugged is True else "on_battery" if plugged is False else None
        seconds_left = getattr(battery, "secsleft", None)
        unlimited = getattr(backend, "POWER_TIME_UNLIMITED", -2)
        unknown = getattr(backend, "POWER_TIME_UNKNOWN", -1)
        if (isinstance(seconds_left, bool) or not isinstance(seconds_left, (int, float))
                or not math.isfinite(float(seconds_left)) or seconds_left < 0
                or seconds_left in (unlimited, unknown)):
            seconds_left = None
        else:
            seconds_left = int(seconds_left)

        value = {
            "charge_percent": percent,
            "charging_state": None,
            "power_connection": power_connection,
            "seconds_remaining": seconds_left,
        }
        limitations = [
            "psutil reports AC connection, not whether charge is actively increasing; charging_state is unavailable."
        ]
        if power_connection is None:
            limitations.append("Power-connection state was unavailable.")
        if seconds_left is None:
            limitations.append("No meaningful OS-supplied time-remaining estimate was available.")
        status = "error" if percent_error and power_connection is None and seconds_left is None else "partial"
        return [_observation("battery", value, "percent, connection state, seconds", status, timestamp,
                             limitation=" ".join(limitations), error=percent_error)]
    except Exception as exc:
        return [_observation("battery", None, None, "error", timestamp,
                             error=_error_text(exc))]


def _collect_processes(timestamp: str, stop_event: threading.Event | None,
                       *, top_by_memory: bool, max_chars: int,
                       max_records: int) -> list[MetricObservation]:
    try:
        records = list_running_processes(stop_event=stop_event)
        if stop_event is not None and stop_event.is_set():
            return []
        if top_by_memory:
            records = sorted(records, key=lambda row: (
                row.memory_bytes is None,
                -(row.memory_bytes or 0),
                (row.name or "").casefold(),
                row.pid,
            ))
        bounded = records[:max(1, int(max_records))]
        encoded = format_process_observations(bounded, max_chars=max(256, int(max_chars)))
        value = json.loads(encoded)
        value["total_processes_observed"] = len(records)
        value["truncated"] = bool(value["truncated"] or len(records) > len(bounded))
        value["selection"] = "descending resident memory" if top_by_memory else "name and PID"
        inaccessible = any(row.name is None or row.executable is None or row.memory_bytes is None
                           for row in bounded)
        status = "partial" if value["truncated"] or inaccessible else "ok"
        limitation = "Some process fields were unavailable." if inaccessible else None
        if value["truncated"]:
            suffix = f"Only {value['shown_processes']} of {len(records)} observed process(es) are shown."
            limitation = f"{limitation} {suffix}" if limitation else suffix
        return [_observation("processes", value, "structured process records", status, timestamp,
                             limitation=limitation)]
    except ProcessInspectionError as exc:
        return [_observation("processes", None, None, "error", timestamp, error=_error_text(exc))]
    except Exception as exc:
        return [_observation("processes", None, None, "error", timestamp, error=_error_text(exc))]


def collect_system_metrics(
    requested: Iterable[str],
    *,
    stop_event: threading.Event | None = None,
    top_processes_by_memory: bool = False,
    max_process_chars: int = 1800,
    max_process_records: int = MAX_PROCESS_RECORDS,
    max_disk_partitions: int = MAX_DISK_PARTITIONS,
    disk_mountpoint: str | None = None,
    cpu_interval: float = 0.1,
    psutil_module: Any = None,
) -> SystemMetricsSnapshot:
    """Collect only requested metrics; each OS collector runs off the GUI thread."""
    keys = tuple(dict.fromkeys(str(item) for item in requested))
    timestamp = _now()
    backend = psutil if psutil_module is None else psutil_module
    observations: list[MetricObservation] = []

    for key in keys:
        if stop_event is not None and stop_event.is_set():
            return SystemMetricsSnapshot(timestamp, keys, tuple(observations), False, True)
        if key == "cpu":
            observations.extend(_collect_cpu(backend, timestamp, stop_event, cpu_interval))
        elif key == "ram":
            observations.extend(_collect_ram(backend, timestamp))
        elif key == "disk":
            observations.extend(_collect_disks(
                backend, timestamp, stop_event, max_disk_partitions, disk_mountpoint,
            ))
        elif key == "battery":
            observations.extend(_collect_battery(backend, timestamp))
        elif key == "processes":
            observations.extend(_collect_processes(
                timestamp, stop_event, top_by_memory=top_processes_by_memory,
                max_chars=max_process_chars, max_records=max_process_records,
            ))
        else:
            observations.append(_observation(
                key, None, None, "error", timestamp, error=f"Unknown system metric requested: {key!r}.",
            ))

    cancelled = stop_event is not None and stop_event.is_set()
    complete = not cancelled and all(item.status in {"ok", "not_detected"} for item in observations)
    return SystemMetricsSnapshot(timestamp, keys, tuple(observations), complete, cancelled)


def format_system_snapshot(snapshot: SystemMetricsSnapshot, max_chars: int) -> str:
    """Serialize structured observations for local Ollama, with explicit truncation."""
    limit = int(max_chars)
    if limit <= 0:
        raise ValueError("The structured system measurement budget must be positive.")
    payload = asdict(snapshot)

    def encode() -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    text = encode()
    for observation in payload["observations"]:
        value = observation.get("value")
        if not isinstance(value, dict):
            continue
        if observation["metric"] == "processes":
            rows_key, shown_key = "processes", "shown_processes"
            notice = "Process rows were omitted to fit the selected model context."
        elif observation["metric"] == "disk_partitions":
            rows_key, shown_key = "drives", "shown_drives"
            notice = "Drive rows were omitted to fit the selected model context."
        elif observation["metric"] == "cpu_per_core_utilization":
            rows_key, shown_key = "samples", "shown_logical_cpus"
            notice = "Per-core CPU samples were omitted to fit the selected model context."
        else:
            continue
        rows = value.get(rows_key)
        if not isinstance(rows, list):
            continue
        while len(text) > limit and rows:
            rows.pop()
            value[shown_key] = len(rows)
            value["truncated"] = True
            observation["status"] = "partial"
            existing = observation.get("limitation") or ""
            if notice not in existing:
                observation["limitation"] = f"{existing} {notice}".strip()
            payload["complete"] = False
            text = encode()
    if len(text) > limit:
        raise ValueError("Structured system measurements exceed the safe local-model context budget.")
    return text
