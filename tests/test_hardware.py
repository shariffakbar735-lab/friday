from __future__ import annotations

import unittest

from app.ai.hardware import HardwareSnapshot, detect_hardware, estimate_model_fits, recommend_model
from app.ai.ollama_client import Model, OllamaError
from app.ai.profiles import PROFILES


class OfflineClient:
    def list_models(self):
        raise OllamaError("not running")


class HardwareTests(unittest.TestCase):
    def test_gpu_absence_is_supported_and_ollama_failure_is_nonfatal(self) -> None:
        snapshot = detect_hardware(
            OfflineClient(),
            gpu_detector=lambda: (None, None, None, "simulated: no GPU detected"),
        )
        self.assertTrue(snapshot.cpu_only)
        self.assertFalse(snapshot.ollama_running)
        self.assertIn("simulated: no GPU detected", snapshot.notes)

    def test_gpu_probe_exception_degrades_to_cpu_without_crashing(self) -> None:
        def broken_gpu_probe():
            raise OSError("driver query denied")
        snapshot = detect_hardware(OfflineClient(), gpu_detector=broken_gpu_probe)
        self.assertTrue(snapshot.cpu_only)
        self.assertIn("GPU detection failed", " ".join(snapshot.notes))

    def test_model_recommendation_uses_model_size_and_ram_without_gpu(self) -> None:
        snapshot = HardwareSnapshot(
            ram_total_bytes=16 * 1024**3,
            ram_available_bytes=8 * 1024**3,
            cpu_architecture="AMD64", cpu_count=8, gpu_name=None, gpu_type=None,
            vram_bytes=None, ollama_installed=True, ollama_running=True,
        )
        models = [Model("small:latest", 1 * 1024**3),
                  Model("medium:latest", 5 * 1024**3),
                  Model("large:latest", 20 * 1024**3)]
        fits = estimate_model_fits(snapshot, models, "balanced")
        self.assertTrue(fits[0].fits_estimate)
        self.assertTrue(fits[1].fits_estimate)
        self.assertFalse(fits[2].fits_estimate)
        self.assertEqual(recommend_model(fits, "fast"), "small:latest")
        self.assertEqual(recommend_model(fits, "low-resource"), "small:latest")
        self.assertEqual(recommend_model(fits, "balanced"), "medium:latest")
        self.assertIn("advisory", fits[0].explanation.lower())

    def test_missing_memory_or_model_size_is_reported_as_unknown(self) -> None:
        no_memory = HardwareSnapshot(None, None, "unknown", 1, None, None, None, False, False)
        fit = estimate_model_fits(no_memory, [Model("opaque")])[0]
        self.assertIsNone(fit.fits_estimate)
        self.assertIsNone(fit.estimated_memory_bytes)

    def test_performance_profiles_have_explicit_bounded_context_and_outputs(self) -> None:
        self.assertEqual(set(PROFILES), {"fast", "balanced", "low-resource"})
        self.assertLess(PROFILES["low-resource"].num_ctx, PROFILES["balanced"].num_ctx)
        self.assertLess(PROFILES["low-resource"].num_predict, PROFILES["balanced"].num_predict)
        for profile in PROFILES.values():
            self.assertGreater(profile.num_ctx, 0)
            self.assertGreater(profile.num_predict, 0)
            self.assertTrue(any(term in profile.description.lower() for term in ("speed", "latency", "response time")))


if __name__ == "__main__":
    unittest.main()
