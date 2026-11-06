"""Small, dependency-free client for the local Ollama HTTP API."""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterator

from app.ai.http_stream import cancellable_response, iter_response_lines
from app.ai.reasoning import FinalAnswerFilter


_LOGGER = logging.getLogger(__name__)
_FINAL_CHANNELS = {"final", "answer", "content"}


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
                    options: dict[str, int | float] | None = None,
                    thinking_callback: Callable[[str], None] | None = None) -> Iterator[str]:
        body: dict[str, object] = {"model": model, "messages": messages, "stream": True}
        if options:
            body["options"] = options
        # Qwen3 supports the Boolean thinking toggle; keep it model-specific
        # because other models can expose named thinking levels instead.
        if model.split(":", 1)[0].casefold().startswith("qwen3"):
            body["think"] = False
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        if stop is not None and stop.is_set():
            return
        started_at = time.perf_counter()
        first_visible_at: float | None = None
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            _LOGGER.warning("Ollama request failed category=http_status status=%d", status)
            raise OllamaError(f"Ollama rejected the request (HTTP {status}).") from None
        except (OSError, urllib.error.URLError) as exc:
            _LOGGER.warning("Ollama request failed category=unavailable")
            raise OllamaError("Could not connect to Ollama. Check that it is running and retry.") from exc
        with cancellable_response(response, stop):
            completed = False
            event_index = 0
            final_filter = FinalAnswerFilter()
            try:
                for raw_line in iter_response_lines(response, stop, self.timeout):
                    if stop is not None and stop.is_set():
                        break
                    if not raw_line.strip():
                        continue
                    event_index += 1
                    try:
                        item = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        _LOGGER.warning(
                            "Ollama stream failure category=malformed_json event_index=%d", event_index
                        )
                        raise OllamaError(
                            "Ollama returned an invalid streaming response (malformed JSON)."
                        ) from None
                    if not isinstance(item, dict):
                        _LOGGER.warning(
                            "Ollama stream failure category=unexpected_event_shape event_index=%d", event_index
                        )
                        raise OllamaError(
                            "Ollama returned an invalid streaming response (unexpected event shape)."
                        )
                    if item.get("error"):
                        error_text = item["error"]
                        _LOGGER.warning(
                            "Ollama stream failure category=server_error event_index=%d", event_index
                        )
                        if (isinstance(error_text, str)
                                and "model" in error_text.casefold()
                                and "not found" in error_text.casefold()):
                            raise OllamaError("Ollama model not found. Select an installed model and retry.")
                        raise OllamaError("Ollama reported a server error during generation.")
                    message = item.get("message")
                    if message is None and item.get("done") is True:
                        # Ollama-compatible servers may omit message on a
                        # terminal metadata-only event. It is valid only when
                        # done is explicitly true and there is no text payload.
                        message = {}
                    if not isinstance(message, dict):
                        _LOGGER.warning(
                            "Ollama stream failure category=unexpected_event_shape event_index=%d missing_field=message",
                            event_index,
                        )
                        raise OllamaError(
                            "Ollama returned an invalid streaming response (missing or invalid message object)."
                        )
                    text = message.get("content", "")
                    thinking = message.get("thinking", "")
                    if not isinstance(text, str) or (thinking is not None and not isinstance(thinking, str)):
                        _LOGGER.warning(
                            "Ollama stream failure category=unexpected_event_shape event_index=%d wrong_field_type=message.content_or_thinking",
                            event_index,
                        )
                        raise OllamaError(
                            "Ollama returned an invalid streaming response (invalid text field)."
                        )
                    channels: list[str] = []
                    for source in (item, message):
                        if "channel" in source:
                            channel = source["channel"]
                            if not isinstance(channel, str):
                                _LOGGER.warning(
                                    "Ollama stream failure category=unexpected_event_shape event_index=%d field=channel",
                                    event_index,
                                )
                                raise OllamaError(
                                    "Ollama returned an invalid streaming response (invalid channel field)."
                                )
                            channels.append(channel.casefold())
                    done = item.get("done", False)
                    if not isinstance(done, bool):
                        _LOGGER.warning(
                            "Ollama stream failure category=unexpected_event_shape event_index=%d field=done",
                            event_index,
                        )
                        raise OllamaError(
                            "Ollama returned an invalid streaming response (invalid done field)."
                        )
                    if text and all(channel in _FINAL_CHANNELS for channel in channels):
                        visible = final_filter.feed_final_channel(text)
                        if visible:
                            if first_visible_at is None:
                                first_visible_at = time.perf_counter()
                                _LOGGER.info(
                                    "Ollama stream timing category=first_visible elapsed_ms=%.1f",
                                    (first_visible_at - started_at) * 1000,
                                )
                            yield visible
                    if thinking and thinking_callback is not None:
                        thinking_callback(thinking)
                    if done:
                        final_text = final_filter.finish()
                        if final_text:
                            if first_visible_at is None:
                                first_visible_at = time.perf_counter()
                                _LOGGER.info(
                                    "Ollama stream timing category=first_visible elapsed_ms=%.1f",
                                    (first_visible_at - started_at) * 1000,
                                )
                            yield final_text
                        completed = True
                        _LOGGER.info(
                            "Ollama stream timing category=completed total_ms=%.1f",
                            (time.perf_counter() - started_at) * 1000,
                        )
                        break
            except (OSError, urllib.error.URLError) as exc:
                if stop is not None and stop.is_set():
                    return
                if isinstance(exc, TimeoutError):
                    _LOGGER.warning(
                        "Ollama stream failure category=read_timeout event_index=%d", event_index
                    )
                    raise OllamaError(
                        f"Ollama response timed out after {self.timeout:g} seconds; retry the request."
                    ) from None
                message = str(exc).casefold()
                category = "premature_eof" if any(
                    phrase in message for phrase in ("content length", "inside a chunk", "ended unexpectedly")
                ) else "connection_interrupted"
                _LOGGER.warning(
                    "Ollama stream failure category=%s event_index=%d", category, event_index
                )
                raise OllamaError(
                    "Ollama connection was interrupted while receiving a response (premature EOF); retry the request."
                    if category == "premature_eof"
                    else "Ollama connection was interrupted while receiving a response; retry the request."
                ) from None
            except ValueError:
                _LOGGER.warning(
                    "Ollama stream failure category=invalid_http_framing event_index=%d", event_index
                )
                raise OllamaError("Ollama returned an invalid HTTP streaming response.") from None
            if stop is not None and stop.is_set():
                return
            if not completed:
                _LOGGER.warning(
                    "Ollama stream failure category=missing_done event_index=%d", event_index
                )
                raise OllamaError(
                    "Ollama stream ended without a final done:true completion event; retry the request."
                )
