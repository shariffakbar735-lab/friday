"""Optional OpenAI-compatible streaming backend; never used unless explicitly selected."""
from __future__ import annotations

import json
import ipaddress
import os
import threading
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from typing import Iterator

from app.ai.http_stream import cancellable_response


class CloudConfigurationError(RuntimeError):
    pass


class OpenAICompatibleClient:
    """Minimal SSE client configured by FRIDAY_CLOUD_* environment variables."""

    def __init__(self) -> None:
        self.api_key = os.environ.get("FRIDAY_CLOUD_API_KEY", "").strip()
        self.model = os.environ.get("FRIDAY_CLOUD_MODEL", "").strip()
        self.endpoint = os.environ.get("FRIDAY_CLOUD_BASE_URL", "").strip().rstrip("/")

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model and self.endpoint)

    def stream_chat(self, messages: list[dict[str, str]], stop: threading.Event,
                    max_tokens: int) -> Iterator[str]:
        if not self.configured:
            raise CloudConfigurationError(
                "Cloud mode is optional and not configured. Set FRIDAY_CLOUD_BASE_URL, "
                "FRIDAY_CLOUD_MODEL, and FRIDAY_CLOUD_API_KEY in your environment, or select Local Ollama."
            )
        parts = urlsplit(self.endpoint)
        if not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
            raise CloudConfigurationError("Cloud endpoint must be a base URL without embedded credentials, query, or fragment.")
        try:
            loopback = ipaddress.ip_address(parts.hostname).is_loopback
        except ValueError:
            loopback = parts.hostname.lower().rstrip(".") == "localhost"
        if parts.scheme != "https" and not (parts.scheme == "http" and loopback):
            raise CloudConfigurationError("Use HTTPS for cloud endpoints; plain HTTP is permitted only for loopback testing.")
        url = self.endpoint if parts.path.rstrip("/").endswith("/chat/completions") else self.endpoint.rstrip("/") + "/chat/completions"
        payload = json.dumps({"model": self.model, "messages": messages,
                              "stream": True, "max_tokens": max_tokens}).encode("utf-8")
        request = urllib.request.Request(url, data=payload, method="POST", headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
            "Accept": "text/event-stream",
        })
        if stop.is_set():
            return
        try:
            response = urllib.request.urlopen(request, timeout=90)
        except urllib.error.HTTPError as exc:
            try:
                status = exc.code
            finally:
                exc.close()
            raise RuntimeError(f"Cloud provider returned HTTP {status}.") from None
        except (OSError, urllib.error.URLError):
            raise RuntimeError("Could not reach the configured cloud provider; verify its URL and connection.") from None
        with cancellable_response(response, stop):
            try:
                for raw_line in response:
                    if stop.is_set():
                        break
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        item = json.loads(data)
                        if not isinstance(item, dict):
                            raise ValueError
                        if item.get("error"):
                            raise RuntimeError("Cloud provider returned an error event.")
                        choices = item.get("choices", [])
                        text = choices[0].get("delta", {}).get("content", "") if choices else ""
                        if not isinstance(text, str):
                            raise ValueError
                    except (json.JSONDecodeError, IndexError, AttributeError, TypeError, ValueError) as exc:
                        raise RuntimeError("Cloud provider returned an invalid streaming event.") from exc
                    if text:
                        yield text
            except (OSError, urllib.error.URLError):
                if stop.is_set():
                    return
                raise RuntimeError("Cloud connection was interrupted while receiving a response.") from None
