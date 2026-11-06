"""Filter inline reasoning markers before any model text reaches the UI or history."""
from __future__ import annotations

import re


class FinalAnswerFilter:
    """Withhold ambiguous model text until it is safe to expose as final output.

    Ollama's structured ``message.thinking`` field and non-final channels are
    filtered by the client. This parser is a defensive fallback for model
    templates that place ``<think>`` markers inside ``message.content``.
    Unclosed reasoning blocks and incomplete marker prefixes are discarded.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"
    _OPEN_RE = re.compile(re.escape(_OPEN), re.IGNORECASE)
    _CLOSE_RE = re.compile(re.escape(_CLOSE), re.IGNORECASE)
    _META_OPENING_RE = re.compile(
        r"^\s*(?:okay|alright|interesting|noticing|important|let['’]s\s+see)\b",
        re.IGNORECASE,
    )
    _LABELED_THOUGHT_RE = re.compile(
        r"^\s*(?:first|initial|my|internal)\s+"
        r"(?:thought|analysis|reasoning)\s*[:\-]",
        re.IGNORECASE,
    )
    _META_SUBJECT_RE = re.compile(
        r"\b(?:user|conversation|pattern|prompt|request|concise|brief|"
        r"overcomplicat|avoid|asked|said|mentioned)\b",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self._buffer = ""
        self._in_thinking = False

    @staticmethod
    def _suffix_prefix_length(value: str, markers: tuple[str, ...]) -> int:
        longest = 0
        for marker in markers:
            maximum = min(len(value), len(marker) - 1)
            for length in range(maximum, 0, -1):
                if value[-length:].casefold() == marker[:length].casefold():
                    longest = max(longest, length)
                    break
        return longest

    def feed(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("Final-answer filter accepts text only.")
        self._buffer += text

        while self._buffer:
            if self._in_thinking:
                close_match = self._CLOSE_RE.search(self._buffer)
                if close_match is None:
                    break
                self._buffer = self._buffer[close_match.end():]
                self._in_thinking = False
                continue

            open_match = self._OPEN_RE.search(self._buffer)
            close_match = self._CLOSE_RE.search(self._buffer)
            if close_match is not None and (open_match is None or close_match.start() < open_match.start()):
                # An unmatched close is malformed. Discard its preceding text
                # conservatively rather than risking a leaked reasoning prefix.
                self._buffer = self._buffer[close_match.end():]
                continue
            if open_match is not None:
                self._buffer = self._buffer[open_match.end():]
                self._in_thinking = True
                continue
            break

        # Without an explicit final boundary, this text may still be a
        # reasoning preamble whose closing </think> arrives in a later chunk.
        # Holding it is intentional: correctness is more important than token
        # latency, and the UI displays its Thinking indicator meanwhile.
        return ""

    def feed_final_channel(self, text: str) -> str:
        """Stream structured final-channel text without unnecessary buffering."""
        if not isinstance(text, str):
            raise TypeError("Final-answer filter accepts text only.")
        self._buffer += text
        visible: list[str] = []
        while self._buffer:
            if self._in_thinking:
                close_match = self._CLOSE_RE.search(self._buffer)
                if close_match is None:
                    keep = self._suffix_prefix_length(self._buffer, (self._CLOSE,))
                    self._buffer = self._buffer[-keep:] if keep else ""
                    break
                self._buffer = self._buffer[close_match.end():]
                self._in_thinking = False
                continue
            open_match = self._OPEN_RE.search(self._buffer)
            close_match = self._CLOSE_RE.search(self._buffer)
            if close_match is not None and (open_match is None or close_match.start() < open_match.start()):
                self._buffer = self._buffer[close_match.end():]
                continue
            if open_match is not None:
                visible.append(self._buffer[:open_match.start()])
                self._buffer = self._buffer[open_match.end():]
                self._in_thinking = True
                continue
            if ((self._META_OPENING_RE.match(self._buffer)
                    or self._LABELED_THOUGHT_RE.match(self._buffer))
                    and self._META_SUBJECT_RE.search(self._buffer)):
                # Some model templates omit the opening tag but still emit a
                # recognizable meta-analysis preamble before a later close
                # marker. This is only a fallback; structured fields remain
                # authoritative and arbitrary unmarked prose is unknowable.
                break
            keep = self._suffix_prefix_length(self._buffer, (self._OPEN,))
            if keep:
                visible.append(self._buffer[:-keep])
                self._buffer = self._buffer[-keep:]
            else:
                visible.append(self._buffer)
                self._buffer = ""
            break
        return "".join(visible)

    def finish(self) -> str:
        """Flush a completed response; never flush an open think block/tag prefix."""
        if self._in_thinking:
            self._buffer = ""
            return ""
        pending = self._buffer
        self._buffer = ""
        if not pending:
            return ""
        close_match = self._CLOSE_RE.search(pending)
        open_match = self._OPEN_RE.search(pending)
        if close_match is not None and (open_match is None or close_match.start() < open_match.start()):
            pending = pending[close_match.end():]
        if self._in_thinking:
            return ""
        partial_marker = self._suffix_prefix_length(pending, (self._OPEN, self._CLOSE))
        return pending[:-partial_marker] if partial_marker else pending


def sanitize_final_text(text: str) -> str:
    """Remove marked reasoning from a complete response, including old history."""
    parser = FinalAnswerFilter()
    return parser.feed(text) + parser.finish()
