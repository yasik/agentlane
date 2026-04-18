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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self

from pydantic import BaseModel

from agentlane.models import (
    Model,
    ModelResponse,
    OutputSchema,
    PromptSpec,
    ToolSpec,
)
from agentlane.runtime import CancellationToken

from ._cancellation import cancel_task_callback, cancellation_relay_task
from ._handoff import (
    DefaultAgentToolInput,
    DelegatedTaskInput,
    agent_tool_description,
    default_agent_tool_details,
    default_agent_tool_instructions,
    handoff_description,
    normalize_delegation_tool_name,
)
from ._hooks import RunnerHooks
from ._run import (
    RunHistoryItem,
    RunInput,
    RunResult,
    RunState,
    ShimState,
    copy_history_item,
    copy_original_input,
    copy_run_state,
)
from ._stream import RunStream
from ._task import Task
from ._tooling import INHERIT_TOOLS, ToolConfig
from .shims import Shim, ShimBindingContext
from .shims._manager import BoundShimManager


class _EmptyAgentToolArgs(BaseModel):
    """Default args model for predefined parameterless agent tools."""


@dataclass(slots=True)
class DefaultHandoff:
    """Configuration for the generic fresh-agent handoff tool."""

    name: str = "handoff"
    """Model-facing tool name for the generic transfer path."""

    description: str = (
        "Transfer the conversation to a fresh helper agent and continue there."
    )
    """Model-facing tool description for the generic transfer path."""

    instructions: str | PromptSpec[Any] | None = None
    """Instructions used by the fresh delegated handoff agent."""

    model: Model[ModelResponse] | None = None
    """Optional model override for the delegated handoff agent."""

    model_args: dict[str, Any] | None = None
    """Optional model args override for the delegated handoff agent."""

    schema: type[BaseModel] | OutputSchema[Any] | None = None
    """Optional structured output schema for the delegated handoff agent."""

    tools: ToolConfig = INHERIT_TOOLS
    """Tool visibility for the delegated handoff agent."""


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

    shims: Sequence[Shim] | None = None
    """Ordered mutating shim definitions bound per agent instance."""

    handoffs: tuple[Self, ...] | None = None
    """Predefined delegated child agents exposed as model-visible tools."""

    default_handoff: DefaultHandoff | None = None
    """Optional generic handoff configuration for fresh spawned sub-agents."""

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        args_model: type[BaseModel] | None = None,
    ) -> ToolSpec[Any]:
        """Expose this agent descriptor as declarative agent-as-tool metadata.

        Predefined agent tools no longer reserve a synthetic ``task`` field.
        The delegated child receives exactly the validated payload described by
        ``args_model``. When no args model is supplied, the tool is treated as
        parameterless.
        """
        return AgentTool(
            descriptor=self,
            name=name,
            description=description,
            args_model=args_model or _EmptyAgentToolArgs,
        )


class AgentTool(ToolSpec[Any]):
    """Declarative tool schema that routes one call to another agent run."""

    def __init__(
        self,
        *,
        descriptor: AgentDescriptor,
        name: str | None = None,
        description: str | None = None,
        args_model: type[BaseModel],
    ) -> None:
        tool_name = name or normalize_delegation_tool_name(descriptor.name)
        tool_description = description or agent_tool_description(
            descriptor.name,
            descriptor.description,
        )
        super().__init__(
            name=tool_name,
            description=tool_description,
            args_model=args_model,
        )
        self.descriptor = descriptor


class DefaultAgentTool(ToolSpec[DefaultAgentToolInput]):
    """Declarative tool schema for one generic spawned helper agent.

    The model sees one normal tool, typically named ``agent``. Its arguments
    define the delegated helper's name and optional task metadata. The runner
    parses those arguments into ``DefaultAgentToolInput`` and forwards that
    structured payload to the spawned child agent exactly like any other
    agent-as-tool call.
    """

    def __init__(
        self,
        *,
        name: str = "agent",
        description: str = (
            "Spawn a focused helper agent by name and optional task, then continue with the result."
        ),
        instructions: str | None = None,
        model: Model[ModelResponse] | None = None,
        model_args: dict[str, Any] | None = None,
        output_schema: type[BaseModel] | OutputSchema[Any] | None = None,
        tools: ToolConfig = INHERIT_TOOLS,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            args_model=DefaultAgentToolInput,
        )
        self.instructions = instructions
        self.model = model
        self.model_args = model_args
        self.output_schema = output_schema
        self.tools = tools

    def resolved_instructions(
        self,
        parsed_input: DefaultAgentToolInput,
    ) -> str:
        """Return instructions for one spawned helper invocation."""
        if self.instructions is not None:
            return f"{self.instructions}\n\n" + default_agent_tool_details(
                name=parsed_input.name,
                description=parsed_input.description,
                task=parsed_input.task,
            )

        return default_agent_tool_instructions(
            name=parsed_input.name,
            description=parsed_input.description,
            task=parsed_input.task,
        )


class HandoffTool(ToolSpec[DelegatedTaskInput]):
    """Declarative tool schema for a predefined first-class handoff target."""

    def __init__(self, *, descriptor: AgentDescriptor) -> None:
        super().__init__(
            name=normalize_delegation_tool_name(descriptor.name),
            description=handoff_description(
                descriptor.name,
                descriptor.description,
            ),
            args_model=DelegatedTaskInput,
        )
        self.descriptor = descriptor


class DefaultHandoffTool(ToolSpec[DelegatedTaskInput]):
    """Declarative tool schema for the generic fresh-agent handoff path."""

    def __init__(self, *, config: DefaultHandoff) -> None:
        super().__init__(
            name=config.name,
            description=config.description,
            args_model=DelegatedTaskInput,
        )
        self.config = config


class LifecycleRunner(Protocol):
    """Structural runner protocol used by the lifecycle queue."""

    async def run(
        self,
        agent: Task,
        state: RunState,
        *,
        hooks: RunnerHooks | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult: ...

    def run_stream(
        self,
        agent: Task,
        state: RunState,
        *,
        hooks: RunnerHooks | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> RunStream: ...


@dataclass(slots=True)
class _QueuedRunInput:
    """One pending inbound run input waiting for runner execution.

    Each enqueued input gets its own ``Future`` so the caller can ``await``
    the result of its specific turn, even if several inputs are queued ahead.
    """

    run_input: RunInput
    """Public run input queued for the next runner invocation."""

    future: Future[RunResult] | None
    """Future resolved with the final run result for terminal inputs."""

    stream: RunStream | None
    """Live stream handle for streamed inputs."""

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
        self._bound_shim_manager: BoundShimManager | None = None

        # Guards `_pending_inputs` and `_is_running`. Held only for queue
        # bookkeeping — never during expensive operations like state copies
        # or runner invocations.
        self._lock = asyncio.Lock()
        self._shim_bind_lock = asyncio.Lock()

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

    @property
    def bound_shim_manager(self) -> BoundShimManager | None:
        """Return the already-bound shim manager, if any."""
        return self._bound_shim_manager

    async def ensure_shims_bound(self, *, agent: Task) -> BoundShimManager:
        """Bind the descriptor's shim definitions once for this agent instance."""
        if self._bound_shim_manager is not None:
            return self._bound_shim_manager

        async with self._shim_bind_lock:
            if self._bound_shim_manager is None:
                self._bound_shim_manager = await BoundShimManager.bind(
                    shims=self._descriptor.shims,
                    context=ShimBindingContext(
                        task=agent,
                    ),
                )
            return self._bound_shim_manager

    async def enqueue_input(
        self,
        *,
        agent: Task,
        runner: LifecycleRunner,
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
        await self.ensure_shims_bound(agent=agent)
        queued_input = _QueuedRunInput(
            run_input=run_input,
            future=get_running_loop().create_future(),
            stream=None,
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
        if queued_input.future is None:
            raise AssertionError("Terminal queue input unexpectedly has no future.")
        return await queued_input.future

    async def enqueue_input_stream(
        self,
        *,
        agent: Task,
        runner: LifecycleRunner,
        hooks: RunnerHooks | None,
        run_input: RunInput,
        cancellation_token: CancellationToken | None = None,
    ) -> RunStream:
        """Queue one streamed run input and return its live stream handle."""
        await self.ensure_shims_bound(agent=agent)
        internal_token = CancellationToken()
        stream = RunStream(on_close=internal_token.cancel)
        relay_task = cancellation_relay_task(
            source=cancellation_token,
            target=internal_token,
        )
        if relay_task is not None:
            stream.add_cleanup(cancel_task_callback(relay_task))

        queued_input = _QueuedRunInput(
            run_input=run_input,
            future=None,
            stream=stream,
            cancellation_token=internal_token,
        )
        should_drain = False

        async with self._lock:
            self._pending_inputs.append(queued_input)
            if not self._is_running:
                self._is_running = True
                should_drain = True

        if should_drain:
            drain_task = asyncio.create_task(
                self._drain_pending_inputs(
                    agent=agent,
                    runner=runner,
                    hooks=hooks,
                )
            )
            drain_task.add_done_callback(_consume_drain_task_result)

        return stream

    async def _drain_pending_inputs(
        self,
        *,
        agent: Task,
        runner: LifecycleRunner,
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

                if (
                    active_input.cancellation_token is not None
                    and active_input.cancellation_token.is_cancelled
                ):
                    exc = asyncio.CancelledError()
                    _fail_queued_input(active_input, exc)
                    active_input = None
                    continue

                # Execute one runner turn
                try:
                    if active_input.stream is None:
                        result = await runner.run(
                            agent=agent,
                            state=working_state,
                            hooks=hooks,
                            cancellation_token=active_input.cancellation_token,
                        )
                    else:
                        runner_stream = runner.run_stream(
                            agent=agent,
                            state=working_state,
                            hooks=hooks,
                            cancellation_token=active_input.cancellation_token,
                        )
                        async for event in runner_stream:
                            active_input.stream.emit(event)
                        result = await runner_stream.result()
                except Exception as exc:  # noqa: BLE001
                    # Runner failures are contained: only this input's
                    # future is failed. The persisted baseline is unchanged,
                    # so subsequent queued inputs start from the last good
                    # state.
                    _fail_queued_input(active_input, exc)
                else:
                    # Commit the working state only after success, ensuring
                    # the baseline always reflects a completed turn. Handoff
                    # may complete through a delegated child agent, so the
                    # runner can return a different final run state than the
                    # local working copy that started the turn.
                    self._run_state = copy_run_state(result.run_state) or working_state
                    _resolve_queued_input(active_input, result)
                finally:
                    active_input = None

        except BaseException as exc:
            # Catastrophic failure (cancellation, KeyboardInterrupt, etc.).
            # Fail any in-flight and remaining queued futures, then reset.
            if active_input is not None:
                _fail_queued_input(active_input, exc)

            async with self._lock:
                pending_inputs = list(self._pending_inputs)
                self._pending_inputs.clear()
                self._is_running = False

            for queued_input in pending_inputs:
                _fail_queued_input(queued_input, exc)
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
            shim_state=ShimState(),
        )

    # Case 3: continuation — fork from the current baseline
    next_state = copy_run_state(current_state)
    if next_state is None:
        raise AssertionError("Current run state copy unexpectedly returned None.")

    _append_run_input(next_state.continuation_history, run_input)
    return next_state


def _append_run_input(
    history: list[RunHistoryItem],
    run_input: str | list[RunHistoryItem],
) -> None:
    """Append one raw run input onto continuation history.

    Strings are appended directly. Lists are iterated and each item is
    shallow-copied to prevent cross-state aliasing of mutable containers.
    """
    if isinstance(run_input, str):
        history.append(run_input)
        return

    for item in run_input:
        history.append(copy_history_item(item))


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


def _resolve_queued_input(queued_input: _QueuedRunInput, result: RunResult) -> None:
    """Resolve one queued terminal or streaming input successfully."""
    if queued_input.stream is not None:
        queued_input.stream.finish(result)
        return

    if queued_input.future is None:
        raise AssertionError("Queued terminal input unexpectedly has no future.")

    _set_future_result(queued_input.future, result)


def _fail_queued_input(
    queued_input: _QueuedRunInput,
    exc: BaseException,
) -> None:
    """Fail one queued terminal or streaming input."""
    if queued_input.stream is not None:
        queued_input.stream.fail(exc)
        return

    if queued_input.future is None:
        raise AssertionError("Queued terminal input unexpectedly has no future.")

    _set_future_exception(queued_input.future, exc)


def _consume_drain_task_result(task: asyncio.Task[None]) -> None:
    """Consume a background drain task result to avoid unhandled warnings."""
    if task.cancelled():
        return
    _ = task.exception()
