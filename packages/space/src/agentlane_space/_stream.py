"""Bounded native SPACE event parsing and output collection."""

import base64
import binascii
from collections.abc import AsyncIterator

import httpx
from pydantic import ValidationError

from ._errors import SpaceError, SpaceErrorKind
from ._types import CommandResult, ServerFrame

# Bound the encoded event separately from the smaller decoded result budget.
_MAX_FRAME_BYTES = 4 * 1024 * 1024


async def event_frames(response: httpx.Response) -> AsyncIterator[ServerFrame]:
    """Read complete SSE records without unbounded line buffering."""
    pending = bytearray()
    data: list[bytes] = []
    frame_size = 0
    async for chunk in response.aiter_bytes():
        # HTTP chunks can split lines or contain several events. Keep bytes
        # intact until an entire SSE data record is available for JSON decoding.
        pending.extend(chunk)
        while b"\n" in pending:
            line, _, rest = pending.partition(b"\n")
            pending = bytearray(rest)
            line = line.rstrip(b"\r")
            frame_size += len(line)
            if frame_size > _MAX_FRAME_BYTES:
                raise _protocol_error()
            if not line:
                if data:
                    # SSE joins repeated data fields with newlines. Heartbeats
                    # contain no data and therefore produce no protocol frame.
                    try:
                        yield ServerFrame.model_validate_json(b"\n".join(data))
                    except ValidationError:
                        raise _protocol_error() from None
                data = []
                frame_size = 0
            elif line.startswith(b"data:"):
                value = bytes(line[5:])
                data.append(value[1:] if value.startswith(b" ") else value)
        # Include an unfinished line so a missing delimiter cannot bypass the cap.
        if len(pending) + frame_size > _MAX_FRAME_BYTES:
            raise _protocol_error()

    if pending or data:
        # EOF does not finish a partial event. Accepting it could hide lost bytes.
        raise _protocol_error()


class OutputCollector:
    """Collect a complete bounded result or a bounded output tail."""

    def __init__(self, *, limit: int, tail: bool) -> None:
        self._limit = limit
        self._tail = tail
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._combined = bytearray()
        self._truncated = False

    def append(self, frame: ServerFrame) -> None:
        if frame.data_base64 is None:
            raise _protocol_error()
        try:
            chunk = base64.b64decode(frame.data_base64, validate=True)
        except (ValueError, binascii.Error):
            raise _protocol_error() from None
        target = self._stdout if frame.type == "stdout" else self._stderr
        self._append(target, chunk)

        # Preserve arrival order across streams; concatenating stdout and stderr
        # after completion would change the visible Bash output.
        self._append(self._combined, chunk)

    def _append(self, target: bytearray, chunk: bytes) -> None:
        if len(target) + len(chunk) > self._limit:
            if not self._tail:
                # Worker JSON requires every byte. Tail mode is for display
                # output, such as Bash results backed by a complete guest log.
                raise SpaceError(
                    "SPACE result exceeds the response limit.",
                    kind=SpaceErrorKind.PROTOCOL,
                )
            self._truncated = True
        target.extend(chunk)
        if len(target) > self._limit:
            del target[: -self._limit]

    def result(self, exit_frame: ServerFrame) -> CommandResult:
        if exit_frame.exit_code is None or exit_frame.status not in {
            "exited",
            "signaled",
            "timed_out",
        }:
            raise _protocol_error()
        return CommandResult(
            stdout=bytes(self._stdout),
            stderr=bytes(self._stderr),
            combined=bytes(self._combined),
            exit_code=exit_frame.exit_code,
            # A command may choose exit code 124 itself. Only SPACE's status
            # establishes that the server enforced a timeout.
            timed_out=exit_frame.status == "timed_out",
            truncated=self._truncated,
        )


def _protocol_error() -> SpaceError:
    return SpaceError(
        "SPACE returned an incomplete or invalid event stream.",
        kind=SpaceErrorKind.PROTOCOL,
    )
