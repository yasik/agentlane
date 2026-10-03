"""Cancellation at public harness startup, preparation, and cleanup boundaries."""

import asyncio
from contextlib import suppress
from contextvars import ContextVar, Token
from typing import Any, Literal

import anyio
import pytest
from anyio.abc import TaskGroup

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    RunInput,
    Runner,
    RunnerHooks,
    RunResult,
    RunState,
    Task,
)
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.shims import PreparedTurn, Shim
from agentlane.models.run import RunContext
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine

from .tools_test_utils import (
    SequenceModel,
    StreamingSequenceModel,
    make_assistant_response,
)

type _RunMode = Literal["run", "stream", "events"]
type _WaitPhase = Literal["start", "prepare", "end"]


class _LifecycleShim(Shim):
    def __init__(
        self,
        label: str,
        calls: list[str],
        *,
        wait_at: _WaitPhase | None = None,
        fail_start: bool = False,
        cleanup_error: BaseException | None = None,
    ) -> None:
        self._label = label
        self._calls = calls
        self._wait_at = wait_at
        self._fail_start = fail_start
        self._cleanup_error = cleanup_error
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.settled = False
        self.tasks: list[asyncio.Task[Any] | None] = []

    @property
    def name(self) -> str:
        return self._label

    async def on_run_start(
        self, state: RunState, transient_state: RunContext[Any]
    ) -> None:
        del state, transient_state
        self.tasks.append(asyncio.current_task())
        self._calls.append(f"{self.name}:start")
        await self._wait("start")
        if self._fail_start:
            raise ValueError("primary startup failure")

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        del turn
        self.tasks.append(asyncio.current_task())
        self._calls.append(f"{self.name}:prepare")
        await self._wait("prepare")

    async def on_run_end(
        self, result: RunResult | None, transient_state: RunContext[Any]
    ) -> None:
        del result, transient_state
        self.tasks.append(asyncio.current_task())
        self._calls.append(f"{self.name}:end")
        await self._wait("end")
        # Cleanup must still be able to await after its run token is cancelled.
        await asyncio.sleep(0)
        if self._cleanup_error is not None:
            raise self._cleanup_error

    async def _wait(self, phase: _WaitPhase) -> None:
        if self._wait_at != phase:
            return

        self.entered.set()
        try:
            await self.release.wait()
        finally:
            self.settled = True


class _ContextShim(_LifecycleShim):
    def __init__(self, calls: list[str], *, wait_at: _WaitPhase | None) -> None:
        super().__init__("context", calls, wait_at=wait_at)
        self._value: ContextVar[str] = ContextVar("shim_value", default="unset")
        self._context_token: Token[str] | None = None
        self.values: list[str] = []

    async def on_run_start(
        self, state: RunState, transient_state: RunContext[Any]
    ) -> None:
        self._context_token = self._value.set("active")
        await super().on_run_start(state, transient_state)

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        self.values.append(self._value.get())
        await super().prepare_turn(turn)

    async def on_run_end(
        self, result: RunResult | None, transient_state: RunContext[Any]
    ) -> None:
        self.values.append(self._value.get())
        assert self._context_token is not None
        self._value.reset(self._context_token)
        self.values.append(self._value.get())
        await super().on_run_end(result, transient_state)


class _TaskGroupShim(_LifecycleShim):
    def __init__(self, calls: list[str], *, wait_at: _WaitPhase | None) -> None:
        super().__init__("task-group", calls, wait_at=wait_at)
        self._group: TaskGroup | None = None
        self._child_release = asyncio.Event()
        self.child_closed = False

    async def on_run_start(
        self, state: RunState, transient_state: RunContext[Any]
    ) -> None:
        await super().on_run_start(state, transient_state)
        self._group = anyio.create_task_group()
        await self._group.__aenter__()
        self._group.start_soon(self._child)

    async def on_run_end(
        self, result: RunResult | None, transient_state: RunContext[Any]
    ) -> None:
        assert self._group is not None
        self._child_release.set()
        try:
            await self._group.__aexit__(None, None, None)
        finally:
            await super().on_run_end(result, transient_state)

    async def _child(self) -> None:
        try:
            await self._child_release.wait()
        finally:
            self.child_closed = True


class _ObserverJoinToken(CancellationToken):
    def __init__(self) -> None:
        super().__init__()
        self.observing = asyncio.Event()
        self.joining = asyncio.Event()

    async def wait_cancelled(self) -> None:
        self.observing.set()
        try:
            await super().wait_cancelled()
        finally:
            # Pause observer shutdown so the owner can be cancelled at its join.
            self.joining.set()
            await asyncio.Event().wait()


class _AwaitObserverShim(_LifecycleShim):
    def __init__(self, calls: list[str], token: _ObserverJoinToken) -> None:
        super().__init__("observer", calls)
        self._token = token

    async def on_run_start(
        self, state: RunState, transient_state: RunContext[Any]
    ) -> None:
        await super().on_run_start(state, transient_state)
        await self._token.observing.wait()


class _EndHooks(RunnerHooks):
    def __init__(self, calls: list[str], *, error: BaseException | None = None) -> None:
        self._calls = calls
        self._error = error

    async def on_agent_end(self, task: Task, result: RunResult | None) -> None:
        del task, result
        self._calls.append("hook:end")
        if self._error is not None:
            raise self._error


def _cleanup_cancellation(*, nested: bool) -> BaseException:
    cancelled = asyncio.CancelledError("cleanup cancelled")
    if not nested:
        return cancelled

    return BaseExceptionGroup(
        "cleanup group",
        [
            RuntimeError("secondary cleanup failure"),
            BaseExceptionGroup("nested cleanup group", [cancelled]),
        ],
    )


async def _run(
    agent: DefaultAgent, mode: _RunMode, token: CancellationToken | None = None
) -> RunResult:
    if mode == "run":
        return await agent.run("test", cancellation_token=token)

    stream = (
        await agent.run_stream("test", cancellation_token=token)
        if mode == "stream"
        else await agent.run_events("test", cancellation_token=token)
    )
    try:
        async for _event in stream:
            pass
        return await stream.result()
    finally:
        await stream.aclose()
        with suppress(BaseException):
            await stream.result()


async def _assert_cancelled(task: asyncio.Task[RunResult], mode: _RunMode) -> None:
    if mode == "run":
        with pytest.raises(RuntimeError, match="delivery status `canceled`"):
            await asyncio.wait_for(task, 1)
    else:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)


async def _assert_no_task_leaks(before: set[asyncio.Task[Any]]) -> None:
    # Stream completion schedules nested close callbacks and token relay cleanup.
    for _ in range(5):
        await asyncio.sleep(0)

    pending = asyncio.all_tasks() - before
    try:
        assert not pending
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_runtime_stop_during_shim_cleanup_resolves_cancelled_delivery() -> None:
    before = asyncio.all_tasks()
    runtime = SingleThreadedRuntimeEngine()
    await runtime.start()
    calls: list[str] = []
    slow = _LifecycleShim("slow", calls, wait_at="end")
    agent = DefaultAgent(
        runtime=runtime,
        descriptor=AgentDescriptor(
            name="fixture",
            model=SequenceModel([make_assistant_response("done")]),
            shims=(slow, _LifecycleShim("later", calls)),
        ),
        hooks=_EndHooks(calls),
    )
    task = asyncio.create_task(agent.run("test"))
    try:
        await asyncio.wait_for(slow.entered.wait(), 1)
        await asyncio.wait_for(runtime.stop(), 1)
        await _assert_cancelled(task, "run")
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.stop()

    assert slow.settled
    assert calls[-3:] == ["slow:end", "later:end", "hook:end"]
    await _assert_no_task_leaks(before)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("origin", ["shim", "hook"])
async def test_cleanup_cancellation_survives_other_failures_and_runs_later_cleanup(
    mode: _RunMode, nested: bool, origin: str
) -> None:
    before = asyncio.all_tasks()
    calls: list[str] = []
    cancellation = _cleanup_cancellation(nested=nested)
    model_type = SequenceModel if mode == "run" else StreamingSequenceModel
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="fixture",
            model=model_type([make_assistant_response("done")]),
            shims=(
                _LifecycleShim(
                    "first", calls, cleanup_error=ValueError("first cleanup failure")
                ),
                _LifecycleShim(
                    "second",
                    calls,
                    cleanup_error=cancellation if origin == "shim" else None,
                ),
                _LifecycleShim("last", calls),
            ),
        ),
        hooks=_EndHooks(calls, error=cancellation if origin == "hook" else None),
    )
    await _assert_cancelled(asyncio.create_task(_run(agent, mode)), mode)

    assert calls[-4:] == ["first:end", "second:end", "last:end", "hook:end"]
    assert agent.run_state is None
    await _assert_no_task_leaks(before)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
async def test_primary_startup_error_survives_nested_cleanup_cancellation(
    mode: _RunMode,
) -> None:
    before = asyncio.all_tasks()
    calls: list[str] = []
    model_type = SequenceModel if mode == "run" else StreamingSequenceModel
    model = model_type([make_assistant_response("must not run")])
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="fixture",
            model=model,
            shims=(
                _LifecycleShim(
                    "first", calls, cleanup_error=_cleanup_cancellation(nested=True)
                ),
                _LifecycleShim("failed", calls, fail_start=True),
                _LifecycleShim("not-started", calls),
            ),
        ),
        hooks=_EndHooks(calls, error=RuntimeError("last cleanup failure")),
    )
    with pytest.raises((RuntimeError, ValueError), match="primary startup failure"):
        await asyncio.wait_for(_run(agent, mode), 1)

    assert calls == [
        "first:start",
        "failed:start",
        "first:end",
        "failed:end",
        "hook:end",
    ]
    assert not model.calls
    await _assert_no_task_leaks(before)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
@pytest.mark.parametrize("phase", ["start", "prepare"])
async def test_token_cancellation_interrupts_shim_work_and_waits_for_cleanup(
    mode: _RunMode, phase: _WaitPhase
) -> None:
    before = asyncio.all_tasks()
    calls: list[str] = []
    token = CancellationToken()
    blocked = _LifecycleShim("blocked", calls, wait_at=phase)
    model_type = SequenceModel if mode == "run" else StreamingSequenceModel
    model = model_type([make_assistant_response("must not run")])
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="fixture",
            model=model,
            shims=(
                _LifecycleShim("first", calls, cleanup_error=ValueError("secondary")),
                blocked,
                _LifecycleShim("last", calls),
            ),
        ),
        hooks=_EndHooks(calls),
    )
    task = asyncio.create_task(_run(agent, mode, token))
    try:
        await asyncio.wait_for(blocked.entered.wait(), 1)
        token.cancel()
        await _assert_cancelled(task, mode)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert blocked.settled
    assert [call for call in calls if call.endswith(":end")] == (
        ["first:end", "blocked:end", "hook:end"]
        if phase == "start"
        else ["first:end", "blocked:end", "last:end", "hook:end"]
    )
    assert not model.calls
    assert agent.run_state is None
    await _assert_no_task_leaks(before)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
async def test_precancelled_token_skips_shim_startup(mode: _RunMode) -> None:
    before = asyncio.all_tasks()
    calls: list[str] = []
    token = CancellationToken()
    token.cancel()
    model_type = SequenceModel if mode == "run" else StreamingSequenceModel
    model = model_type([make_assistant_response("must not run")])
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="fixture", model=model, shims=(_LifecycleShim("first", calls),)
        )
    )
    await _assert_cancelled(asyncio.create_task(_run(agent, mode, token)), mode)

    assert not calls
    assert not model.calls
    await _assert_no_task_leaks(before)


@pytest.mark.asyncio
@pytest.mark.parametrize("input_kind", ["text", "history", "state"])
async def test_terminal_run_input_shapes_forward_cancellation(input_kind: str) -> None:
    before = asyncio.all_tasks()
    calls: list[str] = []
    token = CancellationToken()
    blocked = _LifecycleShim("blocked", calls, wait_at="prepare")
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="fixture",
            model=SequenceModel([make_assistant_response("must not run")]),
            shims=(blocked,),
        )
    )
    run_input: RunInput
    if input_kind == "state":
        run_input = RunState(instructions=None, history=["test"], responses=[])
    elif input_kind == "history":
        run_input = ["test"]
    else:
        run_input = "test"

    task = asyncio.create_task(agent.run(run_input, cancellation_token=token))
    try:
        await asyncio.wait_for(blocked.entered.wait(), 1)
        token.cancel()
        await _assert_cancelled(task, "run")
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert blocked.settled
    assert calls == ["blocked:start", "blocked:prepare", "blocked:end"]
    await _assert_no_task_leaks(before)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["stream", "events"])
async def test_cancelled_drain_resolves_active_and_queued_streams_and_can_restart(
    mode: _RunMode,
) -> None:
    before = asyncio.all_tasks()
    calls: list[str] = []
    token = CancellationToken()
    blocked = _LifecycleShim("blocked", calls, wait_at="start")
    model = StreamingSequenceModel([make_assistant_response("recovered")])
    agent = Agent(
        SingleThreadedRuntimeEngine(),
        Runner(),
        descriptor=AgentDescriptor(name="fixture", model=model, shims=(blocked,)),
    )
    enqueue = (
        agent.enqueue_input_stream if mode == "stream" else agent.enqueue_input_events
    )
    first = await enqueue("first", cancellation_token=token)
    await asyncio.wait_for(blocked.entered.wait(), 1)
    second = await enqueue("second")
    assert agent.pending_input_count == 1

    token.cancel()
    try:
        for stream in (first, second):
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(stream.result(), 1)

        assert not agent.is_running
        assert agent.pending_input_count == 0
        assert agent.run_state is None
        assert not model.calls

        blocked.release.set()
        recovered = await enqueue("retry")
        try:
            result = await asyncio.wait_for(recovered.result(), 1)
            assert result.final_output == "recovered"
        finally:
            await recovered.aclose()
    finally:
        for stream in (first, second):
            await stream.aclose()
            with suppress(BaseException):
                await stream.result()

    assert calls.count("blocked:end") == 2
    await _assert_no_task_leaks(before)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("resource", ["context", "task-group"])
async def test_shim_lifecycle_keeps_task_and_context(
    mode: _RunMode, cancel: bool, resource: str
) -> None:
    before = asyncio.all_tasks()
    calls: list[str] = []
    token = CancellationToken()
    shim_type = _ContextShim if resource == "context" else _TaskGroupShim
    shim = shim_type(calls, wait_at="prepare" if cancel else None)
    model_type = SequenceModel if mode == "run" else StreamingSequenceModel
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="fixture",
            model=model_type([make_assistant_response("done")]),
            shims=(shim,),
        )
    )
    task = asyncio.create_task(_run(agent, mode, token))
    try:
        if cancel:
            await asyncio.wait_for(shim.entered.wait(), 1)
            token.cancel()
            await _assert_cancelled(task, mode)
        else:
            result = await asyncio.wait_for(task, 1)
            assert result.final_output == "done"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert len(shim.tasks) == 3
    assert all(task is shim.tasks[0] for task in shim.tasks)
    if isinstance(shim, _ContextShim):
        assert shim.values == ["active", "active", "unset"]
    else:
        assert shim.child_closed
    await _assert_no_task_leaks(before)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
@pytest.mark.parametrize("fail_start", [False, True])
async def test_finished_shim_work_does_not_link_cleanup_to_cancelled_token(
    mode: _RunMode, fail_start: bool
) -> None:
    before = asyncio.all_tasks()
    calls: list[str] = []
    token = CancellationToken()
    shim = _LifecycleShim("cleanup", calls, wait_at="end", fail_start=fail_start)
    model_type = SequenceModel if mode == "run" else StreamingSequenceModel
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="fixture",
            model=model_type([make_assistant_response("done")]),
            shims=(shim,),
        )
    )
    task = asyncio.create_task(_run(agent, mode, token))
    try:
        await asyncio.wait_for(shim.entered.wait(), 1)
        token.cancel()
        for _ in range(3):
            await asyncio.sleep(0)
        assert not task.done()

        shim.release.set()
        if fail_start:
            with pytest.raises(
                (RuntimeError, ValueError), match="primary startup failure"
            ):
                await asyncio.wait_for(task, 1)
        else:
            result = await asyncio.wait_for(task, 1)
            assert result.final_output == "done"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert shim.settled
    await _assert_no_task_leaks(before)


@pytest.mark.asyncio
async def test_owner_cancellation_during_observer_join_is_not_suppressed() -> None:
    before = asyncio.all_tasks()
    calls: list[str] = []
    token = _ObserverJoinToken()
    shim = _AwaitObserverShim(calls, token)
    model = SequenceModel([make_assistant_response("must not run")])
    agent = DefaultAgent(
        descriptor=AgentDescriptor(name="fixture", model=model, shims=(shim,))
    )
    task = asyncio.create_task(agent.run("test", cancellation_token=token))
    try:
        await asyncio.wait_for(token.joining.wait(), 1)
        owner = shim.tasks[0]
        assert owner is not None
        owner.cancel()
        await _assert_cancelled(task, "run")
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert calls == ["observer:start", "observer:end"]
    assert not model.calls
    await _assert_no_task_leaks(before)
