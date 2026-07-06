import asyncio
import json
from collections.abc import AsyncIterator
from io import StringIO
from typing import Any, Self, cast

from agentlane.harness import RunEvent, RunResult, RunState
from agentlane.harness.tools import ToolApprovalEvent
from agentlane.runtime import CancellationToken


class FakeRunEventStream:
    """Small controllable stream for backend lifecycle tests."""

    _END = object()

    def __init__(self) -> None:
        self.aclose_calls = 0
        self.close_error: Exception | None = None
        self.result_awaits = 0
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._result: asyncio.Future[RunResult] = (
            asyncio.get_running_loop().create_future()
        )

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> RunEvent:
        item = await self._queue.get()
        if item is self._END:
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            raise item
        return cast(RunEvent, item)

    async def result(self) -> RunResult:
        self.result_awaits += 1
        return await self._result

    async def aclose(self) -> None:
        self.aclose_calls += 1
        if self.close_error is not None:
            raise self.close_error

        if not self._result.done():
            self._result.set_exception(asyncio.CancelledError())

        self._queue.put_nowait(self._END)

    def emit(self, event: RunEvent) -> None:
        self._queue.put_nowait(event)

    def finish(self, result: RunResult) -> None:
        if not self._result.done():
            self._result.set_result(result)
        self._queue.put_nowait(self._END)

    def fail(self, exc: BaseException) -> None:
        if not self._result.done():
            self._result.set_exception(exc)
        self._queue.put_nowait(exc)


class FakeAgent:
    def __init__(self) -> None:
        self.run_state: RunState | None = None
        self.prompts: list[str] = []
        self.reset_calls = 0
        self.streams: list[FakeRunEventStream] = []
        self.approval_events: AsyncIterator[ToolApprovalEvent] | None = None
        self.cancellation_tokens: list[CancellationToken | None] = []

    def reset(self) -> None:
        self.reset_calls += 1
        self.run_state = None

    async def run_events(
        self,
        input: str,
        /,
        *,
        approval_events: AsyncIterator[ToolApprovalEvent],
        cancellation_token: CancellationToken | None = None,
    ) -> FakeRunEventStream:
        self.prompts.append(input)
        self.approval_events = approval_events
        self.cancellation_tokens.append(cancellation_token)
        stream = FakeRunEventStream()
        self.streams.append(stream)
        return stream


def emitted_events(buffer: StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in buffer.getvalue().splitlines()]


async def wait_for_event_count(buffer: StringIO, count: int) -> list[dict[str, Any]]:
    for _ in range(100):
        events = emitted_events(buffer)
        if len(events) >= count:
            return events
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"Expected at least {count} events, got {emitted_events(buffer)}"
    )


async def wait_for_stream(agent: FakeAgent, index: int = 0) -> FakeRunEventStream:
    for _ in range(100):
        if len(agent.streams) > index:
            return agent.streams[index]
        await asyncio.sleep(0.01)
    raise AssertionError("Expected fake stream to be created.")
