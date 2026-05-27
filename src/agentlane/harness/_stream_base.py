"""Shared stream-handle mechanics for harness run streams."""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from typing import Any, Self, cast

from ._run import RunResult


class BaseRunStream[T](AsyncIterator[T]):
    """Shared queue, result, closure, and cleanup behavior for run streams."""

    def __init__(
        self,
        *,
        end_sentinel: object,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._result_future: asyncio.Future[RunResult] = (
            asyncio.get_running_loop().create_future()
        )
        self._closed = False
        self._cleaned_up = False
        self._end_sentinel = end_sentinel
        self._on_close = on_close
        self._cleanup_callbacks: list[Callable[[], None]] = []

    def __aiter__(self) -> Self:
        """Return the stream itself as the async iterator."""
        return self

    async def __anext__(self) -> T:
        """Return the next streamed event."""
        item = await self._queue.get()
        if item is self._end_sentinel:
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            raise item
        return cast(T, item)

    async def result(self) -> RunResult:
        """Return the final harness result for this stream."""
        return await self._result_future

    async def aclose(self) -> None:
        """Request early closure of the stream."""
        if self._closed:
            return

        self._closed = True
        if self._on_close is not None:
            self._on_close()
        self._run_cleanups()
        if not self._result_future.done():
            self._result_future.set_exception(asyncio.CancelledError())
        self._queue.put_nowait(self._end_sentinel)

    def emit(self, event: T) -> None:
        """Push one live event into the stream."""
        if self._closed:
            return
        self._queue.put_nowait(event)

    def finish(self, result: RunResult) -> None:
        """Resolve the stream successfully with the final run result."""
        if not self._result_future.done():
            self._result_future.set_result(result)
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(self._end_sentinel)
        self._run_cleanups()

    def fail(self, exc: BaseException) -> None:
        """Fail the stream and surface the error to consumers."""
        if not self._result_future.done():
            self._result_future.set_exception(exc)
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(exc)
        self._run_cleanups()

    def add_cleanup(self, callback: Callable[[], None]) -> None:
        """Register one cleanup callback for stream termination."""
        self._cleanup_callbacks.append(callback)

    def _run_cleanups(self) -> None:
        """Run cleanup callbacks at most once."""
        if self._cleaned_up:
            return
        self._cleaned_up = True
        for callback in self._cleanup_callbacks:
            with suppress(Exception):
                callback()


def close_stream_callback(stream: BaseRunStream[Any]) -> Callable[[], None]:
    """Return a cleanup callback that closes a child run stream."""

    def close_stream() -> None:
        asyncio.create_task(_close_stream(stream))

    return close_stream


async def _close_stream(stream: BaseRunStream[Any]) -> None:
    await stream.aclose()
    with suppress(BaseException):
        await stream.result()
