"""Lifecycle helpers for harness agent execution."""

import asyncio
from asyncio import Future, get_running_loop
from collections import deque
from dataclasses import dataclass
from typing import Any, cast

from pydantic import BaseModel

from agentlane.models import (
    Model,
    ModelResponse,
    OutputSchema,
    PromptSpec,
    Tools,
)
from agentlane.runtime import CancellationToken

from ._hooks import RunnerHooks
from ._run import RunInput, RunResult, RunState
from ._runner import Runner
from ._task import Task


@dataclass(slots=True)
class AgentDescriptor:
    """Static agent configuration shared by the public agent and lifecycle."""

    name: str
    """Human-readable agent name."""

    description: str | None = None
    """Short description of the agent responsibility."""

    model: Model[ModelResponse] | None = None
    """Canonical model client used by the default runner."""

    instructions: str | PromptSpec[Any] | None = None
    """Optional instructions used to seed new conversations."""

    model_args: dict[str, Any] | None = None
    """Model request arguments forwarded as-is to the model call."""

    schema: type[BaseModel] | OutputSchema[Any] | None = None
    """Structured-output schema forwarded to the model."""

    tools: Tools | None = None
    """Canonical tool configuration visible to the model and later phases."""

    skills: tuple[object, ...] | None = None
    """Skills associated with the agent in later phases."""

    context: object | None = None
    """Opaque context reference reserved for later phases."""

    memory: object | None = None
    """Opaque memory reference reserved for later phases."""


@dataclass(slots=True)
class _QueuedRunInput:
    """One pending inbound run input waiting for runner execution."""

    run_input: RunInput
    """Public run input queued for the next runner invocation."""

    future: Future[RunResult]
    """Future resolved with the final run result for this input."""

    cancellation_token: CancellationToken | None
    """Optional cancellation token for the runner turn."""


class AgentLifecycle:
    """Owns per-agent run state and next-turn input queueing."""

    def __init__(
        self,
        *,
        descriptor: AgentDescriptor,
        run_state: RunState | None = None,
    ) -> None:
        """Initialize lifecycle state for one agent instance."""
        self._descriptor = descriptor
        self._run_state = _copy_run_state(run_state)
        self._pending_inputs: deque[_QueuedRunInput] = deque()
        self._is_running = False
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        """Return whether the lifecycle is currently executing a runner turn."""
        return self._is_running

    @property
    def pending_input_count(self) -> int:
        """Return the number of queued run inputs not yet started."""
        return len(self._pending_inputs)

    def run_state_snapshot(self) -> RunState | None:
        """Return a shallow copy of the current resumable run state."""
        return _copy_run_state(self._run_state)

    async def enqueue_input(
        self,
        *,
        agent: Task,
        runner: Runner,
        hooks: RunnerHooks | None,
        run_input: RunInput,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        """Queue one run input and return the final run result."""
        queued_input = _QueuedRunInput(
            run_input=run_input,
            future=get_running_loop().create_future(),
            cancellation_token=cancellation_token,
        )
        should_drain = False

        async with self._lock:
            self._pending_inputs.append(queued_input)
            if not self._is_running:
                # Only the first enqueuer becomes the active drainer. Later
                # arrivals simply append inputs and wait for their own futures.
                self._is_running = True
                should_drain = True

        if should_drain:
            await self._drain_pending_inputs(
                agent=agent,
                runner=runner,
                hooks=hooks,
            )

        return await queued_input.future

    async def _drain_pending_inputs(
        self,
        *,
        agent: Task,
        runner: Runner,
        hooks: RunnerHooks | None,
    ) -> None:
        """Drain queued turns sequentially under the current runtime guarantee.

        The lifecycle intentionally processes one queued input per runner
        invocation. It does not batch multiple queued inputs into one larger
        runner call.
        """
        active_input: _QueuedRunInput | None = None
        try:
            while True:
                async with self._lock:
                    if not self._pending_inputs:
                        self._is_running = False
                        return
                    active_input = self._pending_inputs.popleft()
                    # The lifecycle hands each queued input a private working
                    # copy so failed turns do not mutate the persisted run
                    # state. Only successful runner completion is committed.
                    working_state = _next_run_state(
                        self._run_state,
                        active_input.run_input,
                    )

                try:
                    result = await runner.run(
                        agent=agent,
                        state=working_state,
                        hooks=hooks,
                        cancellation_token=active_input.cancellation_token,
                    )
                except Exception as exc:  # noqa: BLE001
                    _set_future_exception(active_input.future, exc)
                else:
                    # Persist the advanced state only after the runner returns
                    # successfully for this queued input.
                    self._run_state = working_state
                    _set_future_result(active_input.future, result)
                finally:
                    active_input = None
        except BaseException as exc:
            if active_input is not None:
                _set_future_exception(active_input.future, exc)
            async with self._lock:
                pending_inputs = list(self._pending_inputs)
                self._pending_inputs.clear()
                self._is_running = False
            for queued_input in pending_inputs:
                _set_future_exception(queued_input.future, exc)
            raise


def _next_run_state(
    current_state: RunState | None,
    run_input: RunInput,
) -> RunState:
    """Return the next working state for one queued input."""
    if isinstance(run_input, RunState):
        if current_state is not None:
            raise ValueError(
                "Cannot resume a `RunState` when the agent already has persisted run state."
            )
        resumed_state = _copy_run_state(run_input)
        if resumed_state is None:
            raise AssertionError("RunState copy unexpectedly returned None.")
        return resumed_state

    if current_state is None:
        return RunState(
            original_input=_copy_original_input(run_input),
            continuation_history=[],
            responses=[],
        )

    next_state = _copy_run_state(current_state)
    if next_state is None:
        raise AssertionError("Current run state copy unexpectedly returned None.")
    _append_run_input(next_state.continuation_history, run_input)
    return next_state


def _append_run_input(history: list[object], run_input: str | list[object]) -> None:
    """Append one raw run input onto continuation history."""
    if isinstance(run_input, str):
        history.append(run_input)
        return
    for item in run_input:
        history.append(_copy_item(item))


def _copy_original_input(original_input: str | list[object]) -> str | list[object]:
    """Copy one original run input for state ownership."""
    if isinstance(original_input, str):
        return original_input
    return [_copy_item(item) for item in original_input]


def _copy_run_state(run_state: RunState | None) -> RunState | None:
    """Copy one run state for lifecycle isolation."""
    if run_state is None:
        return None
    return RunState(
        original_input=_copy_original_input(run_state.original_input),
        continuation_history=[
            _copy_item(item) for item in run_state.continuation_history
        ],
        responses=list(run_state.responses),
        turn_count=run_state.turn_count,
    )


def _copy_item(item: object) -> object:
    """Copy one generic run item when shallow ownership is needed."""
    if isinstance(item, list):
        return list(cast(list[object], item))
    return item


def _set_future_result(future: Future[RunResult], value: RunResult) -> None:
    """Resolve a future once when it is still pending."""
    if not future.done():
        future.set_result(value)


def _set_future_exception(
    future: Future[RunResult],
    exc: BaseException,
) -> None:
    """Fail a future once when it is still pending."""
    if not future.done():
        future.set_exception(exc)
