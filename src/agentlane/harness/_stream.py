"""Harness run-stream handle for live agent execution."""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from typing import cast

from agentlane.models import ModelStreamEvent

from ._run import RunResult

_STREAM_END = object()


class RunStream(AsyncIterator[ModelStreamEvent]):
    """Async stream handle for one harness run.

    `RunStream` exposes the live per-event model stream while keeping the final
    harness `RunResult` available separately via `result()`.
    """

    def __init__(
        self,
        *,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """Initialize one run stream handle."""
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._result_future: asyncio.Future[RunResult] = (
            asyncio.get_running_loop().create_future()
        )
        self._closed = False
        self._cleaned_up = False
        self._on_close = on_close
        self._cleanup_callbacks: list[Callable[[], None]] = []

    def __aiter__(self) -> "RunStream":
        """Return the stream itself as the async iterator."""
        return self

    async def __anext__(self) -> ModelStreamEvent:
        """Return the next streamed model event."""
        item = await self._queue.get()
        if item is _STREAM_END:
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            raise item
        return cast(ModelStreamEvent, item)

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
        self._queue.put_nowait(_STREAM_END)

    def emit(self, event: ModelStreamEvent) -> None:
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
            self._queue.put_nowait(_STREAM_END)
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
