"""User-selectable inference profiles; values bound context/output, not speed guarantees."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceProfile:
    key: str
    display_name: str
    num_ctx: int
    num_predict: int
    system_instruction: str
    description: str


PROFILES = {
    "fast": PerformanceProfile(
        "fast", "Fast", 4096, 256,
        "Answer directly and concisely. Prefer the shortest complete useful response.",
        "Prioritizes shorter outputs and a smaller context window; latency still depends on model and hardware.",
    ),
    "balanced": PerformanceProfile(
        "balanced", "Balanced", 8192, 512,
        "Give a clear, useful answer with appropriate detail. Avoid unnecessary repetition.",
        "A moderate context/output budget; actual memory use and speed vary by model and hardware.",
    ),
    "low-resource": PerformanceProfile(
        "low-resource", "Low-resource", 2048, 128,
        "Use a compact answer. Focus on the most important points and ask if more detail is needed.",
        "Uses the smallest context/output budget; does not guarantee a particular response time.",
    ),
}
