"""Narrow natural-language parser for the deterministic Windows action allowlist."""
from __future__ import annotations

import re

from app.system.actions import ActionRequest, ActionValidationError, parse_action_request


class ActionIntentError(ActionValidationError):
    """Action-like text was not precise enough to execute safely."""


_PREFIX = r"(?:(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?)?"
_ROOT_ALIASES = {
    "desktop": "desktop",
    "desktop folder": "desktop",
    "documents": "documents",
    "documents folder": "documents",
    "my documents": "documents",
    "downloads": "downloads",
    "downloads folder": "downloads",
    "my downloads": "downloads",
    "friday": "friday_project",
    "friday project": "friday_project",
    "friday project folder": "friday_project",
    "friday project directory": "friday_project",
    "project": "friday_project",
    "project folder": "friday_project",
    "project directory": "friday_project",
    "current folder": "current",
    "current directory": "current",
}


def _root_id(value: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", value.strip(" .!?;:").casefold())
    return _ROOT_ALIASES.get(cleaned)


def _request(action: str, **fields: str) -> ActionRequest:
    return parse_action_request({"action": action, **fields})


def _application_id(target: str) -> str:
    if re.search(r"(?:^|\s)-[a-z0-9]|[;&|<>`\"']", target):
        raise ActionIntentError("Executable arguments and shell syntax are not accepted.")
    name = re.sub(r"\s+(?:app|application)$", "", target).strip()
    app_id = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    if not app_id:
        raise ActionIntentError("Application name is missing.")
    return app_id


def parse_action_text(text: str) -> ActionRequest | None:
    """Parse a small supported command set; return None for ordinary chat.

    Unsupported or ambiguous action-like requests raise ActionIntentError so
    they cannot fall through to an LLM that might claim an action occurred.
    """
    if not isinstance(text, str):
        return None
    original_normalized = re.sub(r"\s+", " ", text.strip())
    normalized = original_normalized.casefold()
    if not normalized:
        return None
    original_normalized = re.sub(r"[.!?]+$", "", original_normalized).strip()
    normalized = re.sub(r"[.!?]+$", "", normalized).strip()
    if len(normalized) > 512:
        action_like = re.match(
            _PREFIX + r"(?:open|launch|start|run|list|show|display|navigate|go\s+(?:to|into))\b",
            normalized,
        )
        if action_like:
            raise ActionIntentError("Action request exceeds the safe 512-character limit; no action was taken.")
        return None

    match = re.fullmatch(
        _PREFIX + r"(?:go|navigate)\s+to\s+(?:the\s+)?parent(?:\s+(?:folder|directory))?",
        normalized,
    )
    if match:
        return _request("navigate_parent")

    match = re.fullmatch(
        _PREFIX + r"(?:go\s+into|navigate\s+into|open\s+(?:the\s+)?(?:sub)?folder)\s+(.+)",
        original_normalized,
        re.IGNORECASE,
    )
    if match:
        name = match.group(1).strip()
        if name.casefold().startswith("my "):
            name = name[3:].strip()
        return _request("navigate_child", child_name=name)

    match = re.fullmatch(
        _PREFIX
        + r"(?:list|show|display|what(?:'s|\s+is)\s+in)\s+"
        + r"(?:(?:the)\s+)?(?:files?|folders?|items?|contents|directory\s+entries)"
        + r"(?:\s+(?:in|of)\s+(?:my\s+)?(.+))?",
        normalized,
    )
    if match:
        destination = match.group(1)
        root = _root_id(destination) if destination else "current"
        if root is None:
            raise ActionIntentError("That directory is not an approved root. Use Desktop, Documents, Downloads, or the configured FRIDAY project folder.")
        return _request("list_directory", root_id=root)

    match = re.fullmatch(
        _PREFIX + r"(?:open|show)\s+(?:(?:my|the)\s+)?(.+)", normalized
    )
    if match:
        target = match.group(1).strip()
        root = _root_id(target)
        if root is not None:
            return _request("open_folder", root_id=root)
        if any(separator in target for separator in ("/", "\\", ":")):
            raise ActionIntentError("Arbitrary paths are not accepted. Choose an approved folder or navigate within one.")
        if re.search(r"\b(?:folder|directory|file)\b", target):
            raise ActionIntentError("That folder or file was not identified as an approved location. No action was taken.")
        return _request("launch_application", app_id=_application_id(target))

    match = re.fullmatch(
        _PREFIX + r"(?:launch|start|run)\s+(?:the\s+)?(.+)", normalized
    )
    if match:
        target = match.group(1).strip()
        if any(separator in target for separator in ("/", "\\", ":")):
            raise ActionIntentError("Executable paths and shell commands are not accepted.")
        return _request("launch_application", app_id=_application_id(target))

    action_like = re.match(
        _PREFIX + r"(?:open|launch|start|run|list|show|display|navigate|go\s+(?:to|into))\b",
        normalized,
    )
    if action_like:
        raise ActionIntentError("I could not safely interpret that action. Specify one approved app or folder; no action was taken.")
    return None
