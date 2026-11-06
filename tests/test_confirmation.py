from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from app.system.actions import ActionDenied
from app.system.confirmation import ConfirmationGate, ConfirmationRequest


class ConfirmationGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.target = Path(self.temp.name) / "target.txt"
        self.other = Path(self.temp.name) / "other.txt"
        self.request = ConfirmationRequest.create(
            "overwrite_file", (str(self.target),), {"replace": True, "content_hash": "abc"}
        )

    def test_confirmation_summary_shows_exact_absolute_targets_and_parameters(self):
        summary = self.request.confirmation_text()
        self.assertIn(str(self.target.absolute()), summary)
        self.assertIn("overwrite_file", summary)
        self.assertIn('"replace":true', summary)

    def test_cancelled_confirmation_cannot_execute_side_effect(self):
        gate = ConfirmationGate()
        token = gate.issue(self.request, accepted=False)
        side_effect = Mock()
        with self.assertRaises(ActionDenied):
            gate.execute_if_approved(
                self.request, token, revalidate=lambda _request: True, execute=side_effect
            )
        side_effect.assert_not_called()

    def test_changed_target_or_parameters_invalidates_prior_approval(self):
        gate = ConfirmationGate()
        changed_requests = (
            ConfirmationRequest.create(
                "overwrite_file", (str(self.other),), {"replace": True, "content_hash": "abc"}
            ),
            ConfirmationRequest.create(
                "overwrite_file", (str(self.target),), {"replace": False, "content_hash": "abc"}
            ),
        )
        for changed in changed_requests:
            token = gate.issue(self.request, accepted=True)
            side_effect = Mock()
            with self.subTest(changed=changed), self.assertRaises(ActionDenied):
                gate.execute_if_approved(changed, token, revalidate=lambda _request: True, execute=side_effect)
            side_effect.assert_not_called()

    def test_approval_is_single_use_and_revalidates_immediately_before_execution(self):
        gate = ConfirmationGate()
        token = gate.issue(self.request, accepted=True)
        calls = []
        value = gate.execute_if_approved(
            self.request, token,
            revalidate=lambda request: calls.append(("revalidate", request.fingerprint)) or True,
            execute=lambda request: calls.append(("execute", request.fingerprint)) or "done",
        )
        self.assertEqual(value, "done")
        self.assertEqual([item[0] for item in calls], ["revalidate", "execute"])
        with self.assertRaises(ActionDenied):
            gate.execute_if_approved(
                self.request, token, revalidate=lambda _request: True, execute=lambda _request: "again"
            )

    def test_failed_revalidation_prevents_execution(self):
        gate = ConfirmationGate()
        token = gate.issue(self.request, accepted=True)
        side_effect = Mock()
        with self.assertRaisesRegex(ActionDenied, "changed after approval"):
            gate.execute_if_approved(
                self.request, token, revalidate=lambda _request: False, execute=side_effect
            )
        side_effect.assert_not_called()

    def test_confirmation_payload_is_bounded_and_rejects_non_finite_values(self):
        with self.assertRaises(ValueError):
            ConfirmationRequest.create(
                "overwrite_file", tuple(str(self.target) for _ in range(9)), {}
            )
        with self.assertRaisesRegex(ValueError, "JSON-safe"):
            ConfirmationRequest.create("overwrite_file", (str(self.target),), {"value": float("nan")})


if __name__ == "__main__":
    unittest.main()
