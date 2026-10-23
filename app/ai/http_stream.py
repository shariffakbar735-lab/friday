"""Cancellable, bounded reads from urllib HTTP streaming responses."""
from __future__ import annotations

import select
import socket
import ssl
import threading
import time
from contextlib import contextmanager
from typing import Iterator


_READ_POLL_SECONDS = 0.05
_MAX_LINE_BYTES = 4 * 1024 * 1024


def _response_socket(response):
    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    return fp, getattr(raw, "_sock", None)


class CancellableBodyReader:
    """Read decoded HTTP body lines while polling cancellation at short intervals."""

    def __init__(self, response, stop: threading.Event | None, timeout: float) -> None:
        self.response = response
        self.fp, self.sock = _response_socket(response)
        if self.fp is None or self.sock is None:
            raise OSError("HTTP response has no readable socket")
        self.stop = stop
        self.timeout = max(float(timeout), _READ_POLL_SECONDS)
        self._buffer = bytearray()
        self._last_data = time.monotonic()
        self._chunked = bool(getattr(response, "chunked", False))
        self._remaining = 0
        self._chunk_finished = False
        self._body_remaining = getattr(response, "length", None)
        # BufferedReader retains any bytes read together with response headers. read1()
        # drains those bytes first and otherwise performs one nonblocking socket read.
        self.sock.setblocking(False)

    def _fill(self) -> bool:
        while not self._buffer:
            if self.stop is not None and self.stop.is_set():
                return False
            pending = getattr(self.sock, "pending", None)
            if not (callable(pending) and pending() > 0):
                remaining = self.timeout - (time.monotonic() - self._last_data)
                if remaining <= 0:
                    raise TimeoutError("HTTP response timed out")
                try:
                    readable, _, _ = select.select(
                        [self.sock], [], [], min(_READ_POLL_SECONDS, remaining))
                except (OSError, ValueError) as exc:
                    if self.stop is not None and self.stop.is_set():
                        return False
                    raise OSError("HTTP response socket is no longer readable") from exc
                if not readable:
                    continue
            try:
                block = self.fp.read1(65536)
            except (BlockingIOError, ssl.SSLWantReadError, socket.timeout):
                continue
            if not block:
                return False
            self._buffer.extend(block)
            self._last_data = time.monotonic()
        return True

    def _raw_line(self) -> bytes | None:
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._buffer[:newline + 1])
                del self._buffer[:newline + 1]
                return line
            if len(self._buffer) > _MAX_LINE_BYTES:
                raise ValueError("HTTP stream line exceeded the safety limit")
            if not self._fill():
                if self._buffer:
                    line = bytes(self._buffer)
                    self._buffer.clear()
                    return line
                return None

    def _raw_exact(self, size: int) -> bytes | None:
        while len(self._buffer) < size:
            if not self._fill():
                return None
        result = bytes(self._buffer[:size])
        del self._buffer[:size]
        return result

    def _next_chunk(self) -> bool:
        line = self._raw_line()
        if line is None:
            raise OSError("HTTP chunk header ended unexpectedly")
        try:
            size = int(line.strip().split(b";", 1)[0], 16)
        except ValueError as exc:
            raise ValueError("HTTP response contained an invalid chunk header") from exc
        if size < 0:
            raise ValueError("HTTP response contained an invalid chunk size")
        if size == 0:
            while True:
                trailer = self._raw_line()
                if trailer is None or trailer in (b"\r\n", b"\n", b""):
                    break
            self._chunk_finished = True
            return False
        self._remaining = size
        return True

    def _body_piece(self) -> bytes | None:
        if self._chunked:
            if self._chunk_finished:
                return None
            if self._remaining == 0:
                if not self._next_chunk():
                    return None
            amount = min(self._remaining, 65536)
            piece = self._raw_exact(amount)
            if piece is None:
                raise OSError("HTTP response ended inside a chunk")
            self._remaining -= len(piece)
            if self._remaining == 0:
                ending = self._raw_exact(2)
                if ending != b"\r\n":
                    raise ValueError("HTTP response contained an invalid chunk terminator")
            return piece
        if self._body_remaining is not None:
            if self._body_remaining <= 0:
                return None
            amount = min(self._body_remaining, 65536)
            piece = self._raw_exact(amount)
            if piece is None:
                raise OSError("HTTP response ended before its declared content length")
            self._body_remaining -= len(piece)
            return piece
        if self._buffer:
            piece = bytes(self._buffer)
            self._buffer.clear()
            return piece
        if not self._fill():
            return None
        piece = bytes(self._buffer)
        self._buffer.clear()
        return piece

    def readline(self) -> bytes | None:
        line = bytearray()
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line.extend(self._buffer[:newline + 1])
                del self._buffer[:newline + 1]
                return bytes(line)
            if self._buffer:
                line.extend(self._buffer)
                self._buffer.clear()
                if len(line) > _MAX_LINE_BYTES:
                    raise ValueError("HTTP stream line exceeded the safety limit")
            piece = self._body_piece()
            if piece is None:
                return bytes(line) if line else None
            self._buffer.extend(piece)
            if len(line) + len(self._buffer) > _MAX_LINE_BYTES:
                raise ValueError("HTTP stream line exceeded the safety limit")


def iter_response_lines(response, stop: threading.Event | None,
                        timeout: float) -> Iterator[bytes]:
    """Yield body lines from content-length, close-delimited, or chunked HTTP streams."""
    reader = CancellableBodyReader(response, stop, timeout)
    while stop is None or not stop.is_set():
        line = reader.readline()
        if line is None:
            return
        yield line


@contextmanager
def cancellable_response(response, stop: threading.Event | None) -> Iterator[object]:
    """Always close the urllib response; reads themselves poll cancellation every 50 ms."""
    try:
        yield response
    finally:
        try:
            response.close()
        except OSError:
            pass
