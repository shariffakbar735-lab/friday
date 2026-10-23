"""Best-effort hardware discovery and conservative local-model fit estimates."""
from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.ai.ollama_client import Model, OllamaClient, OllamaError


@dataclass(frozen=True)
class HardwareSnapshot:
    ram_total_bytes: int | None
    ram_available_bytes: int | None
    cpu_architecture: str
    cpu_count: int
    gpu_name: str | None
    gpu_type: str | None
    vram_bytes: int | None
    ollama_installed: bool
    ollama_running: bool
    notes: tuple[str, ...] = ()

    @property
    def cpu_only(self) -> bool:
        return not self.gpu_name


@dataclass(frozen=True)
class ModelFit:
    name: str
    size_bytes: int | None
    estimated_memory_bytes: int | None
    fits_estimate: bool | None
    explanation: str


def _windows_memory() -> tuple[int | None, int | None]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    try:
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
    except (AttributeError, OSError):
        pass
    return None, None


def _system_memory() -> tuple[int | None, int | None]:
    if os.name == "nt":
        return _windows_memory()
    try:
        values: dict[str, int] = {}
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                key, raw = line.split(":", 1)
                if key in {"MemTotal", "MemAvailable"}:
                    values[key] = int(raw.strip().split()[0]) * 1024
        return values.get("MemTotal"), values.get("MemAvailable")
    except (OSError, ValueError, IndexError):
        return None, None


def _gpu_detection() -> tuple[str | None, str | None, int | None, str | None]:
    """Return name/type/VRAM and a nonfatal diagnostic; never require a GPU driver."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, check=True,
            )
            first = result.stdout.strip().splitlines()[0]
            name, memory_mb = (part.strip() for part in first.split(",", 1))
            return name, "NVIDIA", int(float(memory_mb) * 1024 * 1024), None
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass

    if os.name == "nt":
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell:
            script = "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"
            try:
                result = subprocess.run([powershell, "-NoProfile", "-Command", script],
                                        capture_output=True, text=True, timeout=5, check=True)
                raw = json.loads(result.stdout or "[]")
                adapters = raw if isinstance(raw, list) else [raw]
                for adapter in adapters:
                    name = str(adapter.get("Name", "")).strip()
                    if (not name or "Microsoft Basic" in name or "Remote Display" in name
                            or "Sharing Monitor" in name or "Virtual" in name):
                        continue
                    lowered = name.lower()
                    kind = "NVIDIA" if "nvidia" in lowered else "AMD" if "amd" in lowered or "radeon" in lowered else "Intel" if "intel" in lowered else "Other"
                    reported = adapter.get("AdapterRAM")
                    vram = int(reported) if isinstance(reported, (int, float)) and reported > 0 else None
                    return name, kind, vram, ("Adapter-reported video memory can be inaccurate or truncated." if vram else None)
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError, TypeError):
                return None, None, None, "GPU/driver query failed; local CPU inference remains available."
    return None, None, None, "No supported GPU was detected; FRIDAY can still use Ollama CPU inference."


def detect_hardware(client: OllamaClient | None = None,
                    gpu_detector: Callable[[], tuple[str | None, str | None, int | None, str | None]] = _gpu_detection) -> HardwareSnapshot:
    notes: list[str] = []
    total, available = _system_memory()
    if total is None:
        notes.append("System memory could not be measured; model fit estimates are unavailable.")
    try:
        gpu_name, gpu_type, vram, gpu_note = gpu_detector()
    except Exception as exc:  # hardware/driver probes are optional and must never block launch
        gpu_name, gpu_type, vram, gpu_note = None, None, None, f"GPU detection failed ({type(exc).__name__}); CPU inference remains available."
    if gpu_note:
        notes.append(gpu_note)
    ollama_installed = bool(shutil.which("ollama"))
    if not ollama_installed and os.name == "nt":
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            ollama_installed = Path(local_app, "Programs", "Ollama", "ollama.exe").is_file()
    try:
        (client or OllamaClient()).list_models()
        ollama_running = True
    except OllamaError:
        ollama_running = False
    return HardwareSnapshot(total, available, platform.machine() or "unknown", os.cpu_count() or 1,
                            gpu_name, gpu_type, vram, ollama_installed, ollama_running, tuple(notes))


def estimate_model_fits(snapshot: HardwareSnapshot, models: list[Model], profile: str = "balanced") -> list[ModelFit]:
    """Estimate weights plus runtime headroom; estimates are advisory, not guarantees."""
    if snapshot.ram_available_bytes is None:
        return [ModelFit(m.name, m.size_bytes, None, None,
                         "Unknown: available system memory could not be measured; check Ollama/model requirements.")
                for m in models]
    profile_context = {"fast": 4096, "balanced": 8192, "low-resource": 2048}.get(profile, 4096)
    # Rough KV-cache/runtime reserve, explicitly not a vendor-provided requirement.
    context_reserve = min(2 * 1024**3, max(384 * 1024**2, profile_context * 128 * 1024))
    budget = snapshot.ram_available_bytes
    if snapshot.vram_bytes:
        budget += int(snapshot.vram_bytes * 0.65)
    results: list[ModelFit] = []
    for model in models:
        if not model.size_bytes:
            results.append(ModelFit(model.name, model.size_bytes, None, None,
                                    "Ollama did not report the model's disk size; fit is unknown."))
            continue
        estimate = int(model.size_bytes * 1.25) + context_reserve
        fits = estimate <= budget
        results.append(ModelFit(model.name, model.size_bytes, estimate, fits,
                                "Advisory estimate uses reported model size plus runtime/context headroom; actual use varies by quantization, context, OS, and Ollama offload."))
    return results


def recommend_model(fits: list[ModelFit], profile: str) -> str | None:
    eligible = [item for item in fits if item.fits_estimate is True]
    if not eligible:
        return None
    eligible.sort(key=lambda item: item.size_bytes or 0)
    if profile in {"fast", "low-resource"}:
        return eligible[0].name
    return eligible[-1].name
