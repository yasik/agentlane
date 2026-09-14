"""Cancellable waits for borrowed synchronous process streams.

A borrowed stream may not support interruption. Its daemon thread can remain
alive until the operation returns, but it must not prevent process shutdown.
Each bridge keeps at most one pending reader and one pending writer call.
"""

import asyncio
import os
import threading
from collections.abc import Callable
from contextvars import copy_context
from typing import Protocol


async def call_stream[T](operation: Callable[[], T]) -> T:
    """Wait for one blocking operation without owning a default-executor thread."""
    context = copy_context()
    loop = asyncio.get_running_loop()
    result: asyncio.Future[T] = loop.create_future()

    def finish(value: T) -> None:
        if not result.done():
            result.set_result(value)

    def fail(error: BaseException) -> None:
        if not result.done():
            result.set_exception(error)

    def run() -> None:
        try:
            try:
                value = context.run(operation)
            except BaseException as exc:
                loop.call_soon_threadsafe(fail, exc)
            else:
                loop.call_soon_threadsafe(finish, value)
        except RuntimeError:
            # The session ended before its borrowed blocking stream returned.
            pass

    threading.Thread(target=run, name="agentlane-bridge-io", daemon=True).start()
    return await result


class TextOutput(Protocol):
    """Synchronous output surface needed by the bounded event writer."""

    def write(self, value: str, /) -> int:
        """Write text and return the number of characters written."""
        ...

    def flush(self) -> None:
        """Flush completed writes."""
        ...


class ProcessOutput:
    """Write UTF-8 to a borrowed process descriptor without a buffered-I/O lock.

    The bridge does not close the descriptor. A timed-out daemon write may remain
    blocked until the host drains or closes its pipe, but interpreter shutdown
    cannot wait on a TextIOWrapper lock held by that write.
    """

    def __init__(self, descriptor: int) -> None:
        self._descriptor = descriptor

    def write(self, value: str, /) -> int:
        """Write the complete UTF-8 record, including partial descriptor writes."""
        remaining = memoryview(value.encode("utf-8"))
        while remaining:
            written = os.write(self._descriptor, remaining)
            if written == 0:
                raise BrokenPipeError("Bridge output descriptor accepted no bytes.")
            remaining = remaining[written:]
        return len(value)

    def flush(self) -> None:
        """Descriptor writes have no Python-side output buffer."""
