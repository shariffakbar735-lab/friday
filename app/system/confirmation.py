"""Reusable, exact-payload confirmation primitives for future file operations.

No destructive operation is implemented or enabled by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from app.system.actions import ActionDenied, _is_network_or_device_path

T = TypeVar("T")
MAX_CONFIRMATION_TARGETS = 8
MAX_CONFIRMATION_PARAMETER_BYTES = 16 * 1024
MAX_PENDING_APPROVALS = 128


@dataclass(frozen=True)
class ConfirmationRequest:
    action: str
    absolute_targets: tuple[str, ...]
    parameters_json: str

    @classmethod
    def create(cls, action: str, targets: tuple[str, ...], parameters: dict[str, object]) -> "ConfirmationRequest":
        if not isinstance(action, str) or not action.strip() or len(action) > 128:
            raise ValueError("Confirmation action must be a short non-empty name.")
        if (not isinstance(targets, tuple) or not targets
                or len(targets) > MAX_CONFIRMATION_TARGETS):
            raise ValueError(f"Confirmation requires between 1 and {MAX_CONFIRMATION_TARGETS} exact target paths.")
        normalized: list[str] = []
        for value in targets:
            if (not isinstance(value, str) or not value or len(value) > 32767
                    or _is_network_or_device_path(value)):
                raise ValueError("Confirmation targets must be local absolute paths.")
            path = Path(value)
            if not path.is_absolute():
                raise ValueError("Confirmation targets must be absolute paths.")
            normalized.append(os.path.abspath(str(path)))
        if not isinstance(parameters, dict):
            raise ValueError("Confirmation parameters must be an object.")
        try:
            parameters_json = json.dumps(
                parameters, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Confirmation parameters must be JSON-safe.") from exc
        if len(parameters_json.encode("utf-8")) > MAX_CONFIRMATION_PARAMETER_BYTES:
            raise ValueError("Confirmation parameters exceed their size limit.")
        return cls(action.strip(), tuple(normalized), parameters_json)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            [self.action, self.absolute_targets, self.parameters_json],
            separators=(",", ":"), ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def confirmation_text(self) -> str:
        targets = "\n".join(f"Target: {path}" for path in self.absolute_targets)
        return f"Action: {self.action}\n{targets}\nParameters: {self.parameters_json}"


@dataclass(frozen=True)
class ApprovalToken:
    nonce: str
    fingerprint: str


class ConfirmationGate:
    """Issue and consume explicit, single-use approval for one exact request."""

    def __init__(self) -> None:
        self._issued: dict[str, str] = {}

    def issue(self, request: ConfirmationRequest, accepted: bool) -> ApprovalToken | None:
        if not isinstance(accepted, bool):
            raise TypeError("An explicit boolean confirmation decision is required.")
        if not accepted:
            return None
        if len(self._issued) >= MAX_PENDING_APPROVALS:
            raise ActionDenied("Too many pending confirmations; no approval was issued.")
        token = ApprovalToken(secrets.token_urlsafe(24), request.fingerprint)
        self._issued[token.nonce] = token.fingerprint
        return token

    def execute_if_approved(
        self,
        request: ConfirmationRequest,
        token: ApprovalToken | None,
        *,
        revalidate: Callable[[ConfirmationRequest], bool],
        execute: Callable[[ConfirmationRequest], T],
    ) -> T:
        if token is None or self._issued.pop(token.nonce, None) != request.fingerprint:
            raise ActionDenied("No current approval matches this exact action and target.")
        if token.fingerprint != request.fingerprint:
            raise ActionDenied("Approval was invalidated because the action or target changed.")
        if not revalidate(request):
            raise ActionDenied("Permissions or targets changed after approval; no operation was run.")
        return execute(request)
