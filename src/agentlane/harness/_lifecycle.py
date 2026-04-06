"""Lifecycle helpers for harness agent execution.

The lifecycle owns the per-agent run state and input queue. Its core
responsibility is sequencing: when multiple inputs arrive for the same
``AgentId``, only one runner turn executes at a time. Later arrivals are
queued and drained in order after the active turn completes.

Key invariants maintained here:

- **Single-writer**: only one runner invocation per ``AgentId`` at a time.
- **Copy-on-write**: every runner turn receives a private working copy of
  ``RunState``. The persisted baseline is updated only after a successful
  return.
- **Fair drain**: queued inputs are processed one-at-a-time in FIFO order.
  The lifecycle never batches multiple inputs into a single runner call.
"""

import asyncio
from asyncio import Future, get_running_loop
from collections import deque
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from agentlane.models import (
    Model,
    ModelResponse,
    OutputSchema,
    PromptSpec,
)
from agentlane.runtime import CancellationToken

from ._hooks import RunnerHooks
from ._run import (
    RunInput,
    RunResult,
    RunState,
    copy_item,
    copy_original_input,
    copy_run_state,
)
from ._runner import Runner
from ._task import Task
from ._tooling import INHERIT_TOOLS, ToolConfig


@dataclass(slots=True)
class AgentDescriptor:
    """Static agent configuration shared by the public agent and lifecycle.

    This groups all the fields that remain constant for the lifetime of one
    agent instance. ``RunState`` (the mutable part) is managed separately by
    the lifecycle so recovered state and static config never conflate.
    """

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

    tools: ToolConfig = INHERIT_TOOLS
    """Tool visibility policy for this agent.

    When omitted, a future child agent may inherit tools from its parent.
    ``None`` means "expose no tools explicitly".
    """

    skills: tuple[object, ...] | None = None
    """Skills associated with the agent in later phases."""

    context: object | None = None
    """Opaque context reference reserved for later phases."""

    memory: object | None = None
    """Opaque memory reference reserved for later phases."""


@dataclass(slots=True)
class _QueuedRunInput:
    """One pending inbound run input waiting for runner execution.

    Each enqueued input gets its own ``Future`` so the caller can ``await``
    the result of its specific turn, even if several inputs are queued ahead.
    """

    run_input: RunInput
    """Public run input queued for the next runner invocation."""

    future: Future[RunResult]
    """Future resolved with the final run result for this input."""

    cancellation_token: CancellationToken | None
    """Optional cancellation token for the runner turn."""


class AgentLifecycle:
    """Owns per-agent run state and next-turn input queueing.

    The lifecycle coordinates two concerns:

    1. **State ownership** — it holds the persisted ``RunState`` baseline and
       creates isolated working copies for each runner turn.
    2. **Queue drain** — when the agent is already running, new inputs are
       appended to an internal queue. The first enqueuer becomes the
       "drainer" and processes all pending inputs sequentially.
    """

    def __init__(
        self,
        *,
        descriptor: AgentDescriptor,
        run_state: RunState | None = None,
    ) -> None:
        """Initialize lifecycle state for one agent instance."""
        self._descriptor = descriptor

        # Defensive copy so the caller's original is never mutated.
        self._run_state = copy_run_state(run_state)

        self._pending_inputs: deque[_QueuedRunInput] = deque()
        self._is_running = False

        # Guards `_pending_inputs` and `_is_running`. Held only for queue
        # bookkeeping — never during expensive operations like state copies
        # or runner invocations.
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
        """Return an isolated copy suitable for persistence or inspection."""
        return copy_run_state(self._run_state)

    async def enqueue_input(
        self,
        *,
        agent: Task,
        runner: Runner,
        hooks: RunnerHooks | None,
        run_input: RunInput,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        """Queue one run input and wait for the final result.

        If the agent is idle, the calling coroutine becomes the drainer and
        processes this input (and any that arrive while it runs) in a loop.
        If the agent is already running, the input is simply appended; the
        active drainer will pick it up after the current turn finishes.
        """
        queued_input = _QueuedRunInput(
            run_input=run_input,
            future=get_running_loop().create_future(),
            cancellation_token=cancellation_token,
        )
        should_drain = False

        async with self._lock:
            self._pending_inputs.append(queued_input)
            if not self._is_running:
                # First enqueuer claims the drain responsibility.
                self._is_running = True
                should_drain = True

        if should_drain:
            await self._drain_pending_inputs(
                agent=agent,
                runner=runner,
                hooks=hooks,
            )

        # Every caller awaits its own future, regardless of whether it was
        # the drainer or a later arrival.
        return await queued_input.future

    async def _drain_pending_inputs(
        self,
        *,
        agent: Task,
        runner: Runner,
        hooks: RunnerHooks | None,
    ) -> None:
        """Drain queued turns sequentially until the queue is empty.

        Each iteration:
          1. Pop one input under the lock (minimal critical section).
          2. Build a private working ``RunState`` outside the lock.
          3. Invoke the runner for that single input.
          4. On success, commit the working state as the new baseline.
          5. Resolve or fail the caller's future.

        If a ``BaseException`` (e.g. ``KeyboardInterrupt``) escapes the
        loop, all remaining queued futures are failed and the lifecycle
        resets to idle.
        """
        active_input: _QueuedRunInput | None = None

        try:
            while True:
                async with self._lock:
                    if not self._pending_inputs:
                        self._is_running = False
                        return
                    active_input = self._pending_inputs.popleft()

                # Build working state (outside lock)
                working_state = _next_run_state(
                    self._run_state,
                    active_input.run_input,
                )

                # Execute one runner turn
                try:
                    result = await runner.run(
                        agent=agent,
                        state=working_state,
                        hooks=hooks,
                        cancellation_token=active_input.cancellation_token,
                    )
                except Exception as exc:  # noqa: BLE001
                    # Runner failures are contained: only this input's
                    # future is failed. The persisted baseline is unchanged,
                    # so subsequent queued inputs start from the last good
                    # state.
                    _set_future_exception(active_input.future, exc)
                else:
                    # Commit the working state only after success, ensuring
                    # the baseline always reflects a completed turn.
                    self._run_state = working_state
                    _set_future_result(active_input.future, result)
                finally:
                    active_input = None

        except BaseException as exc:
            # Catastrophic failure (cancellation, KeyboardInterrupt, etc.).
            # Fail any in-flight and remaining queued futures, then reset.
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
    """Build the next working state for one queued input.

    Three cases:
      1. ``RunState`` input + no baseline → resume from the provided state.
      2. Plain input + no baseline → start a brand-new conversation.
      3. Plain input + existing baseline → continue with new user input
         appended to ``continuation_history``.

    Resuming into an agent that already has a baseline is an error — the
    caller should create a fresh agent instance instead.
    """
    # Case 1: resume from a serialized RunState
    if isinstance(run_input, RunState):
        if current_state is not None:
            raise ValueError(
                "Cannot resume a `RunState` when the agent already has persisted run state."
            )

        resumed_state = copy_run_state(run_input)
        if resumed_state is None:
            raise AssertionError("RunState copy unexpectedly returned None.")

        return resumed_state

    # Case 2: first input — initialize a new conversation
    if current_state is None:
        return RunState(
            original_input=copy_original_input(run_input),
            continuation_history=[],
            responses=[],
        )

    # Case 3: continuation — fork from the current baseline
    next_state = copy_run_state(current_state)
    if next_state is None:
        raise AssertionError("Current run state copy unexpectedly returned None.")

    _append_run_input(next_state.continuation_history, run_input)
    return next_state


def _append_run_input(history: list[object], run_input: str | list[object]) -> None:
    """Append one raw run input onto continuation history.

    Strings are appended directly. Lists are iterated and each item is
    shallow-copied to prevent cross-state aliasing of mutable containers.
    """
    if isinstance(run_input, str):
        history.append(run_input)
        return

    for item in run_input:
        history.append(copy_item(item))


def _set_future_result(future: Future[RunResult], value: RunResult) -> None:
    """Resolve a future once, guarding against double-resolution."""
    if not future.done():
        future.set_result(value)


def _set_future_exception(
    future: Future[RunResult],
    exc: BaseException,
) -> None:
    """Fail a future once, guarding against double-resolution."""
    if not future.done():
        future.set_exception(exc)
