"""Small, dependency-free client for the local Ollama HTTP API."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator

from app.ai.http_stream import cancellable_response, iter_response_lines


class OllamaError(RuntimeError):
    """A clear, user-displayable Ollama connection or API error."""


@dataclass(frozen=True)
class Model:
    name: str
    size_bytes: int | None = None


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def list_models(self) -> list[Model]:
        request = urllib.request.Request(f"{self.base_url}/api/tags")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                data = json.load(response)
        except (OSError, urllib.error.URLError) as exc:
            raise OllamaError("Cannot reach Ollama at " + self.base_url + ". Start Ollama, then retry.") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OllamaError("Ollama returned an invalid model catalog.") from exc
        if not isinstance(data, dict) or not isinstance(data.get("models", []), list):
            raise OllamaError("Ollama returned an invalid model catalog.")
        models: list[Model] = []
        for item in data.get("models", []):
            if not isinstance(item, dict):
                raise OllamaError("Ollama returned an invalid model catalog.")
            if not item.get("name"):
                continue
            size = item.get("size")
            models.append(Model(item["name"], size if isinstance(size, int) and size >= 0 else None))
        return models

    def stream_chat(self, model: str, messages: list[dict[str, str]], stop: threading.Event | None = None,
                    options: dict[str, int | float] | None = None) -> Iterator[str]:
        body: dict[str, object] = {"model": model, "messages": messages, "stream": True}
        if options:
            body["options"] = options
        # Ollama's Qwen3 models can spend the entire bounded output budget in a
        # separate thinking channel; request answer-only mode for visible chat.
        if model.split(":", 1)[0].lower().startswith("qwen3"):
            body["think"] = False
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        if stop is not None and stop.is_set():
            return
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read(2048).decode("utf-8", "replace")
            finally:
                exc.close()
            raise OllamaError(f"Ollama rejected the request ({exc.code}): {detail}") from exc
        except (OSError, urllib.error.URLError) as exc:
            raise OllamaError("Could not connect to Ollama. Check that it is running and retry.") from exc
        with cancellable_response(response, stop):
            completed = False
            try:
                for raw_line in iter_response_lines(response, stop, self.timeout):
                    if stop is not None and stop.is_set():
                        break
                    if not raw_line.strip():
                        continue
                    try:
                        item = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise OllamaError("Ollama returned an invalid streaming response.") from exc
                    if not isinstance(item, dict):
                        raise OllamaError("Ollama returned an invalid streaming response.")
                    if item.get("error"):
                        raise OllamaError(str(item["error"]))
                    message = item.get("message") or {}
                    if not isinstance(message, dict):
                        raise OllamaError("Ollama returned an invalid streaming response.")
                    text = message.get("content", "")
                    if not isinstance(text, str):
                        raise OllamaError("Ollama returned an invalid streaming response.")
                    done = item.get("done", False)
                    if not isinstance(done, bool):
                        raise OllamaError("Ollama returned an invalid streaming response.")
                    if text:
                        yield text
                    if done:
                        completed = True
                        break
            except (OSError, urllib.error.URLError) as exc:
                if stop is not None and stop.is_set():
                    return
                if isinstance(exc, TimeoutError):
                    raise OllamaError(f"Ollama response timed out after {self.timeout:g} seconds.") from exc
                raise OllamaError("Ollama connection was interrupted while receiving a response.") from exc
            if stop is not None and stop.is_set():
                return
            if not completed:
                raise OllamaError("Ollama stream ended without a final done:true completion event.")
