"""Per-run model and tool loop, including cleanup and child delegation."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from functools import partial
from typing import Any, Literal, Protocol, cast, runtime_checkable

import structlog
from pydantic import BaseModel, JsonValue, ValidationError
from strenum import LowercaseStrEnum

from agentlane.messaging import AgentId
from agentlane.models import (
    MessageDict,
    MessageRole,
    Model,
    ModelBehaviorError,
    ModelContinuation,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    OutputSchema,
    PromptSpec,
    RunErrorDetails,
    Tool,
    ToolCall,
    ToolChoice,
    ToolExecutionContext,
    ToolExecutor,
    Tools,
    ToolSpec,
    retry_on_errors,
)
from agentlane.models.run import DefaultRunContext
from agentlane.runtime import CancellationToken
from agentlane.tracing import Span, generation_span

from ._agent_delivery import AgentDelivery
from ._cancellation import (
    cancellation_scope,
    raise_cleanup_errors,
)
from ._constants import (
    AGENT_DEPTH_LIMIT_REACHED,
    AGENT_THREAD_LIMIT_REACHED,
    CHANGED_TOOL_CALL_ARGUMENTS,
    DEFAULT_HANDOFF_AGENT_DESCRIPTION,
    DEFAULT_HANDOFF_AGENT_NAME,
    MISSING_MODEL_CHOICE,
)
from ._events import (
    RunEventEmitter,
    RunStateSnapshotBoundary,
)
from ._handoff import (
    DefaultAgentToolInput,
    DelegationKind,
    default_handoff_task_message,
    default_handoff_tool_result,
    delegated_agent_id,
    delegated_result_text,
    require_handoff_result,
)
from ._hooks import RunnerHooks
from ._lifecycle import (
    AgentDescriptor,
    AgentTool,
    DefaultAgentTool,
    DefaultHandoffTool,
    ExternalTool,
    HandoffTool,
    LifecycleRunner,
)
from ._render import normalize_message_content, render_request_messages
from ._run import (
    LiveRunStateView,
    RunHistoryItem,
    RunResult,
    RunState,
    copy_instructions,
    copy_run_state,
    copy_shim_state,
)
from ._snapshot import (
    AgentSnapshot,
    RunContinuation,
    RunPaused,
    ToolResultInput,
)
from ._stream_base import BaseRunStream
from ._task import Task
from ._tooling import InheritTools, RestrictTools, ToolConfig
from .shims import PreparedTurn, Shim
from .shims._manager import BoundShimManager

LOGGER = structlog.get_logger("agentlane.harness.runner")
"""Used for diagnostic events emitted by the per-run execution loop."""


class _AgentDelegationLimit(LowercaseStrEnum):
    """The resource that prevents reserving another child agent."""

    DEPTH = "depth"
    """Returned when a child would exceed the configured delegation depth."""

    THREAD = "thread"
    """Returned when all configured concurrent child slots are occupied."""


class _NextStep(Enum):
    """The operation selected after processing one model response."""

    HANDOFF = auto()
    """Selects control transfer to the handoff target from the current model response."""

    CONTINUE = auto()
    """Selects consumption of the automatic successor already started by the provider."""

    RUN_TOOLS = auto()
    """Selects execution of tool calls in the current model response."""

    WAIT_FOR_TOOLS = auto()
    """Selects waiting for outstanding asynchronous tool results before another turn."""

    FINISH = auto()
    """Selects final output after all model calls and pending tool results are handled."""


type _HandoffTarget = tuple[Tools, ToolCall, HandoffTool | DefaultHandoffTool]
type _Transition = (
    tuple[Literal[_NextStep.HANDOFF], _HandoffTarget]
    | tuple[
        Literal[
            _NextStep.CONTINUE,
            _NextStep.RUN_TOOLS,
            _NextStep.WAIT_FOR_TOOLS,
            _NextStep.FINISH,
        ],
        None,
    ]
)


@runtime_checkable
class RunnerTask(Protocol):
    """Structural protocol for tasks compatible with the default runner.

    The runner accesses these properties during request building and model
    calls. Any task that exposes them is compatible — no subclassing needed.
    """

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> Model[ModelResponse] | None: ...

    @property
    def model_args(self) -> dict[str, Any] | None: ...

    @property
    def schema(self) -> type[BaseModel] | OutputSchema[Any] | None: ...

    @property
    def tools(self) -> Tools | None: ...

    @property
    def base_tools(self) -> Tools | None: ...

    @property
    def instructions(self) -> str | PromptSpec[Any] | None: ...

    @property
    def shims(self) -> Sequence[Shim] | None: ...


@dataclass(slots=True)
class AgentDelegationState:
    """Process-local guard state for generic spawned agent calls."""

    max_depth: int
    max_threads: int
    child_turn_limits: dict[AgentId, int] = field(default_factory=dict[AgentId, int])
    """Active child runs and their inherited turn limits."""

    _live_agents: int = 0
    _depth_by_agent: dict[AgentId, int] = field(default_factory=dict[AgentId, int])
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def reserve_child(
        self,
        *,
        parent_id: AgentId,
        child_id: AgentId,
    ) -> _AgentDelegationLimit | None:
        """Reserve a live child slot and record its recursive depth."""
        async with self._lock:
            parent_depth = self._depth_by_agent.get(parent_id, 0)
            child_depth = parent_depth + 1
            if child_depth > self.max_depth:
                return _AgentDelegationLimit.DEPTH
            if self._live_agents >= self.max_threads:
                return _AgentDelegationLimit.THREAD

            # Reserve the depth and thread slot together before the child starts.
            self._live_agents += 1
            self._depth_by_agent[child_id] = child_depth

            return None

    async def release_child(self, *, child_id: AgentId) -> None:
        """Release a child slot after a delegated agent call finishes."""
        async with self._lock:
            if child_id in self._depth_by_agent:
                del self._depth_by_agent[child_id]
                if self._live_agents > 0:
                    self._live_agents -= 1


@dataclass(slots=True)
class _PendingToolCall:
    call: ToolCall
    task: asyncio.Task[MessageDict]
    delivered: bool = False


async def _ready_tool_messages(
    pending: dict[str, _PendingToolCall],
    required_ids: tuple[str, ...] = (),
    *,
    wait: bool = True,
) -> list[MessageDict]:
    """Join required calls, or wait for the next result without rerunning a call."""
    required = [pending[call_id].task for call_id in required_ids if call_id in pending]
    active = [entry.task for entry in pending.values() if not entry.delivered]

    if required:
        await asyncio.gather(*required)
    elif active and wait:
        await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)

    # Collect each completed result once; keep unfinished calls for later turns.
    results: list[MessageDict] = []
    for entry in pending.values():
        if not entry.delivered and entry.task.done():
            results.append(await entry.task)
            entry.delivered = True

    return results


class RunLoop:
    """Execute one run with isolated turn state and shared execution limits.

    The delegate runner is passed to child agents when a tool or handoff starts
    another run. Model and tool turns execute here without calling that runner.
    """

    def __init__(
        self,
        *,
        agent: Task,
        runner_task: RunnerTask,
        model: Model[ModelResponse],
        state: RunState | ToolResultInput,
        delegate_runner: LifecycleRunner,
        agent_delegation: AgentDelegationState,
        tool_executor: ToolExecutor,
        hooks: RunnerHooks,
        max_attempts: int = 1,
        max_turns: int = 128,
        is_retryable: Callable[[BaseException], bool] | None = None,
        cancellation_token: CancellationToken | None = None,
        emit: Callable[[ModelStreamEvent], None] | None = None,
        run_events: RunEventEmitter | None = None,
        control_stream: BaseRunStream[Any] | None = None,
    ) -> None:
        self.agent = agent
        self.runner_task = runner_task
        self.model = model
        self.external_input = state if isinstance(state, ToolResultInput) else None
        self.state = (
            state.continuation.to_run_state()
            if isinstance(state, ToolResultInput)
            else state
        )
        self.agent_delegation = agent_delegation
        self._is_child = agent.id in agent_delegation.child_turn_limits
        self.tool_executor = tool_executor
        self.hooks = hooks
        self.max_attempts = max_attempts
        self.max_turns = min(
            max_turns, agent_delegation.child_turn_limits.get(agent.id, max_turns)
        )
        if self.external_input is not None:
            self.max_turns = min(
                self.max_turns, self.external_input.continuation.max_turns
            )
        self.agent_delivery = AgentDelivery(
            agent,
            delegate_runner,
            cancellation_token,
            child_turn_limits=agent_delegation.child_turn_limits,
            max_turns=self.max_turns,
        )
        self.is_retryable = is_retryable
        self.cancellation_token = cancellation_token
        self.emit = emit
        self.run_events = run_events
        self.control_stream = control_stream
        self._external_tools_enabled = self.external_input is not None
        self._external_resume_applied = False

        # Keep tool budgets and pending results local to this run.
        self.transient_state = DefaultRunContext()
        self.tool_call_counts: dict[str, int] = {}
        self.tool_round_trips = 0
        self.pending_calls: dict[str, _PendingToolCall] = {}
        self.inherited_turn: PreparedTurn | None = None
        self.shims = _shim_manager(self.agent)

        # Build the harness retry policy once for models that do not own retries.
        self._retryable_run_once = retry_on_errors(
            max_retries=self.max_attempts,
            is_retryable=self.is_retryable,
        )(self._run_once)

    async def run(self) -> RunResult:
        """Execute turns and close model and tool resources on every exit path."""
        self._validate_external_setup()
        pause: RunPaused | None = None
        run_error: BaseException | None = None
        result: RunResult | None = None

        # Open the model session before binding controls or starting the run.
        model_scope = AsyncExitStack()
        try:
            async with cancellation_scope(self.cancellation_token):
                model = await model_scope.enter_async_context(self.model.open_run())

            if self.control_stream is not None and not self._external_tools_enabled:
                self.control_stream.bind_steering(model.steer)

            # Notify hooks and shims before the first prepared turn.
            await self.hooks.on_agent_start(self.agent, self.state)
            if self.shims is not None:
                async with cancellation_scope(self.cancellation_token):
                    await self.shims.on_run_start(self.state, self.transient_state)
            if self.run_events is not None:
                self.run_events.state_snapshot(
                    RunStateSnapshotBoundary.RUN_START, self.state
                )

            # Join background tools before model and shim cleanup on every exit path.
            with generation_span(disabled=self.model.tracing.is_disabled()) as span:
                try:
                    result = await self._run_turns(model, span)

                    return result
                except RunPaused as exc:
                    # Close the trace and resources before raising the pause.
                    pause = exc
                finally:
                    await self._cancel_pending_calls()
        except BaseException as exc:
            run_error = exc

            # Attach undelivered call IDs to the original failure for diagnosis.
            unresolved = [
                call_id
                for call_id, entry in self.pending_calls.items()
                if not entry.delivered
            ]
            if unresolved:
                exc.add_note("Unresolved async tool calls: " + ", ".join(unresolved))

            raise
        finally:
            await self._finish_run(model_scope, result, run_error)

        raise pause

    async def _run_turns(
        self, model: Model[ModelResponse], span: Span[Any]
    ) -> RunResult:
        while True:
            turn = await self._prepare_turn()
            response, completed = await self._call_model(model, turn, span)
            result = await self._finish_turn(turn, response, completed, span)
            if result is not None:
                return result

    def _validate_external_setup(self) -> None:
        """Validate saved state and restore used tool budgets before model I/O."""
        self._validate_external_catalog(self.runner_task.tools)
        if self.external_input is None:
            self._reject_unresolved_external_calls()
            return

        # Match the saved root before restoring tool usage.
        saved = self.external_input.continuation
        if self._is_child or saved.checkpoint.agent_id != self.agent.id:
            raise ValueError("External tool continuation belongs to a different root.")
        self.tool_call_counts = dict(saved.tool_call_counts)
        self.tool_round_trips = saved.tool_round_trips

    def _validate_external_catalog(self, tools: Tools | None) -> None:
        """Enable external tools only at a root without a live provider session."""
        if tools is not None and any(
            isinstance(tool, ExternalTool) for tool in tools.normalized_tools
        ):
            if self._is_child:
                raise ValueError("External tools do not support child suspension.")
            self._external_tools_enabled = True

        if self._external_tools_enabled:
            if self.model.requires_streaming:
                raise ValueError(
                    "External tools require a model without a live provider session."
                )
            if self.control_stream is not None:
                self.control_stream.bind_steering(None)

    def _reject_unresolved_external_calls(self) -> None:
        """Reject unfinished calls without a saved continuation and result."""
        if not self._external_tools_enabled:
            return

        # History alone cannot restore the saved tool budgets.
        pending: set[str] = set()
        for message in render_request_messages(None, self.state.history):
            for raw_call in cast(list[object], message.get("tool_calls") or []):
                call = ToolCall.model_validate(raw_call)
                pending.add(call.id)
            if message.get("role") == MessageRole.TOOL.value:
                pending.discard(str(message.get("tool_call_id")))
        if pending:
            raise ValueError("An unfinished external call requires ToolResultInput.")

    def _apply_external_result(self, tools: Tools | None) -> None:
        """Validate the current tool and append its external result once."""
        if self.external_input is None:
            raise RuntimeError("External result admission requires a typed input.")
        saved = self.external_input.continuation
        definition = next(
            (
                tool
                for tool in (tools.normalized_tools if tools is not None else ())
                if tool.name == saved.tool_call.function.name
            ),
            None,
        )
        if not isinstance(definition, ExternalTool):
            raise ValueError("The pending external tool is no longer registered.")
        if definition.schema != saved.args_schema:
            raise ValueError("The pending external tool contract has changed.")

        # Shims must preserve the pending call. Keep its validated arguments:
        # running argument validation again could generate different defaults.
        current = AgentSnapshot.capture(agent_id=self.agent.id, run_state=self.state)
        RunContinuation.from_json(saved.model_copy(update={"checkpoint": current}))
        self.state.history.append(
            _tool_result_message(
                tool_call=saved.tool_call,
                tool_name=definition.name,
                content=self.external_input.result,
            )
        )
        self._external_resume_applied = True

    async def _prepare_turn(self) -> PreparedTurn:
        self.state.turn_count += 1
        if self.state.turn_count > self.max_turns:
            raise RuntimeError(
                f"Runner exceeded the configured turn limit of {self.max_turns}."
            )

        # Build this turn from the agent's configured defaults.
        turn = PreparedTurn(
            run_state=self.state,
            tools=self.runner_task.tools,
            model_args=_model_args(self.runner_task),
            transient_state=self.transient_state,
        )

        # Let shims adjust the turn before applying tool restrictions.
        if self.shims is not None:
            async with cancellation_scope(self.cancellation_token):
                await self.shims.prepare_turn(turn)

        # An automatic successor keeps the previous turn's tool handlers.
        if self.inherited_turn is not None:
            turn.tools = self._inherited_tools(self.inherited_turn.tools, turn.tools)

        # Check tools and pending calls after shims can change them.
        self._validate_external_catalog(turn.tools)
        if self.external_input is not None and not self._external_resume_applied:
            self._apply_external_result(turn.tools)
        elif self.external_input is None:
            self._reject_unresolved_external_calls()

        # Resume uses the smaller of the saved and current limits.
        if turn.tools is not None and self.external_input is not None:
            saved = self.external_input.continuation
            limits = dict(turn.tools.tool_call_limits or {})
            for name, limit in (saved.tool_call_limits or {}).items():
                limits[name] = min(limits.get(name, limit), limit)
            turn.tools = replace(
                turn.tools,
                max_tool_round_trips=min(
                    turn.tools.max_tool_round_trips, saved.max_tool_round_trips
                ),
                tool_call_limits=limits or None,
            )

        # Remove exhausted tools before the next model request.
        if turn.tools is not None:
            turn.tools = _limit_tools(
                turn.tools, self.tool_call_counts, self.tool_round_trips
            )

        # Classify delegation events against the final tool catalog.
        if self.run_events is not None:
            self.run_events.register_delegation_tool_names(
                frozenset(
                    tool.name
                    for tool in (
                        turn.tools.normalized_tools if turn.tools is not None else ()
                    )
                    if isinstance(
                        tool,
                        (
                            AgentTool,
                            DefaultAgentTool,
                            HandoffTool,
                            DefaultHandoffTool,
                        ),
                    )
                )
            )
            self.run_events.state_snapshot(
                RunStateSnapshotBoundary.TURN_PREPARED, self.state
            )

        return turn

    def _inherited_tools(
        self, inherited: Tools | None, current: Tools | None
    ) -> Tools | None:
        if inherited is None or current is None:
            return None

        # The server already started this successor with the prior catalog.
        permitted = {tool.name for tool in current.normalized_tools}

        return replace(
            current,
            tools=tuple(
                tool for tool in inherited.normalized_tools if tool.name in permitted
            ),
        )

    async def _call_model(
        self,
        model: Model[ModelResponse],
        turn: PreparedTurn,
        span: Span[Any],
    ) -> tuple[ModelResponse, ModelStreamEvent]:
        # Render the current history, then let shims prepare the model input.
        messages = render_request_messages(
            turn.run_state.instructions, turn.run_state.history
        )
        if self.shims is not None:
            messages = await self.shims.transform_messages(turn, messages)

        # Use terminal calls only when the model does not need live events.
        if self.emit is None and not model.requires_streaming:
            response = await self._run_with_retry(
                model=model,
                messages=messages,
                tools=turn.tools,
                model_args=turn.model_args,
                parent_span=span,
            )

            return response, ModelStreamEvent(
                kind=ModelStreamEventKind.COMPLETED, response=response
            )

        # Consume live events even when the caller does not request streamed output.
        return await self._stream_model_call(
            on_async_call=partial(
                self._dispatch_async_call, tools=turn.tools, span=span
            ),
            model=model,
            messages=messages,
            tools=turn.tools,
            model_args=turn.model_args,
            emit=self.emit or (lambda event: None),
            parent_span=span,
        )

    async def _finish_turn(
        self,
        turn: PreparedTurn,
        response: ModelResponse,
        completed: ModelStreamEvent,
        span: Span[Any],
    ) -> RunResult | None:
        # Retain the response and the tool catalog used by an automatic successor.
        self.state.responses.append(response)
        self.inherited_turn = (
            turn
            if completed.continuation == ModelContinuation.AUTOMATIC.value
            else None
        )

        # Let shims observe the response before selecting the next loop action.
        if self.shims is not None:
            await self.shims.on_model_response(turn, response)

        # Validate model output before reading the message or dispatching tools.
        if not response.choices:
            raise _model_behavior_error(
                MISSING_MODEL_CHOICE,
                raw_response=response,
            )

        message = response.choices[0].message
        tool_calls = cast(list[ToolCall], list(message.tool_calls or []))
        normal_calls = self._normal_tool_calls(
            tool_calls, completed.required_tool_call_ids, response
        )

        # Provider-driven continuation or steering would bypass the pause checks.
        if self._external_tools_enabled and (
            completed.continuation is not None or completed.input_messages
        ):
            raise _model_behavior_error(
                "External tools do not support automatic continuation or steering.",
                raw_response=response,
            )

        # Resolve a handoff once, before committing an ordinary tool round.
        handoff: _HandoffTarget | None = None
        if tool_calls and turn.tools is not None:
            for call in tool_calls:
                definition = next(
                    (
                        tool
                        for tool in turn.tools.normalized_tools
                        if tool.name == (call.function.name or "")
                    ),
                    None,
                )
                if isinstance(definition, (HandoffTool, DefaultHandoffTool)):
                    if len(tool_calls) > 1:
                        raise ModelBehaviorError(
                            "Runner received a handoff together with additional tool calls. "
                            "First-class handoff must be the only tool call in the turn."
                        )

                    handoff = (turn.tools, call, definition)
                    break

        # Select the operation before its effects; pending tools prevent completion.
        transition = self._next_step(
            handoff=handoff,
            automatic_successor=completed.continuation
            == ModelContinuation.AUTOMATIC.value,
            has_tool_calls=bool(tool_calls),
        )

        # Handoff transfers history itself. Other paths commit the response before
        # any steering input that follows it.
        if transition[0] is not _NextStep.HANDOFF:
            self.state.history.append(response)
            self.state.history.extend(completed.input_messages or [])
            if tool_calls:
                self.tool_round_trips += 1

        match transition:
            case _NextStep.HANDOFF, tuple() as target:
                # Transfer control without committing a local tool round.
                handoff_tools, handoff_call, handoff_definition = target

                return await self._handoff(
                    handoff_tools, completed, response, handoff_call, handoff_definition
                )

            case _NextStep.CONTINUE, _:
                # The server already started the successor; do not submit it again.
                if normal_calls:
                    raise _model_behavior_error(
                        "Automatic successor skipped required tool results.",
                        raw_response=response,
                    )

            case _NextStep.RUN_TOOLS, _:
                # Complete this round before asking the model for another response.
                await self._finish_tool_round(
                    turn.tools, normal_calls, completed, response, span
                )

            case _NextStep.WAIT_FOR_TOOLS, _:
                # Deliver outstanding results before accepting a final answer.
                async with cancellation_scope(self.cancellation_token):
                    self.state.history.extend(
                        await _ready_tool_messages(
                            self.pending_calls,
                            completed.required_tool_call_ids,
                        )
                    )

            case _NextStep.FINISH, _:
                # Only a final assistant message can become the run result.
                if message.role != MessageRole.ASSISTANT.value:
                    raise _model_behavior_error(
                        "Runner expected the first model choice to be an assistant message.",
                        raw_response=response,
                    )

                return RunResult(
                    final_output=message.content,
                    responses=list(self.state.responses),
                    turn_count=self.state.turn_count,
                    run_state=copy_run_state(self.state),
                )

        return None

    def _next_step(
        self,
        *,
        handoff: _HandoffTarget | None,
        automatic_successor: bool,
        has_tool_calls: bool,
    ) -> _Transition:
        """Select the next operation without executing it or changing run state."""

        # Control transfer and an existing successor take precedence over new work.
        if handoff is not None:
            return _NextStep.HANDOFF, handoff
        if automatic_successor:
            return _NextStep.CONTINUE, None

        # Finish only after the current calls and outstanding results are handled.
        if has_tool_calls:
            return _NextStep.RUN_TOOLS, None
        if self._has_pending_results():
            return _NextStep.WAIT_FOR_TOOLS, None

        return _NextStep.FINISH, None

    def _normal_tool_calls(
        self,
        calls: list[ToolCall],
        required_ids: tuple[str, ...],
        response: ModelResponse,
    ) -> list[ToolCall]:
        # A repeated call ID must refer to the same work already in progress.
        for call in calls:
            pending = self.pending_calls.get(call.id)
            if pending is not None and pending.call != call:
                raise _model_behavior_error(
                    CHANGED_TOOL_CALL_ARGUMENTS, raw_response=response
                )

        # Exclude dispatched calls and verify that every required result is known.
        normal = [call for call in calls if call.id not in self.pending_calls]
        if set(required_ids) - set(self.pending_calls) - {call.id for call in normal}:
            raise _model_behavior_error(
                "Steering requires an unknown tool result.", raw_response=response
            )

        return normal

    async def _handoff(
        self,
        tools: Tools,
        completed: ModelStreamEvent,
        response: ModelResponse,
        call: ToolCall,
        definition: HandoffTool | DefaultHandoffTool,
    ) -> RunResult:
        # Complete steering and tool delivery before transferring control.
        if completed.continuation is not None:
            raise _model_behavior_error(
                "Cannot hand off while steering continuation is pending.",
                raw_response=response,
            )
        if self._has_pending_results():
            raise _model_behavior_error(
                "Cannot hand off while async tool results are pending.",
                raw_response=response,
            )

        return await self._execute_handoff(
            response=response,
            handoff_call=call,
            tool_definition=definition,
            tools=tools,
        )

    async def _finish_tool_round(
        self,
        tools: Tools | None,
        normal_calls: list[ToolCall],
        completed: ModelStreamEvent,
        response: ModelResponse,
        span: Span[Any],
    ) -> None:
        # Run remaining calls with budgets that include tools started during streaming.
        messages: list[MessageDict] = []
        if normal_calls:
            if self.pending_calls and tools is not None:
                tools = _limit_tools(
                    tools, self.tool_call_counts, self.tool_round_trips - 1
                )

            messages = await self._execute_tool_calls(
                tools=tools,
                tool_calls=normal_calls,
                response=response,
                parent_span=span,
            )

        for call in normal_calls:
            name = call.function.name or ""
            self.tool_call_counts[name] = self.tool_call_counts.get(name, 0) + 1

        # Wait for required results. With no outputs to send, wait for the next
        # tool result; otherwise collect only results that are already ready.
        async with cancellation_scope(self.cancellation_token):
            messages.extend(
                await _ready_tool_messages(
                    self.pending_calls,
                    completed.required_tool_call_ids,
                    wait=not messages,
                )
            )

        # Commit tool outputs together before reporting the end of the round.
        self.state.history.extend(messages)
        if self.run_events is not None:
            self.run_events.state_snapshot(
                RunStateSnapshotBoundary.TOOL_ROUND_END, self.state
            )

    def _dispatch_async_call(
        self,
        call: ToolCall,
        *,
        tools: Tools | None,
        span: Span[Any],
    ) -> None:
        if self._external_tools_enabled:
            raise _model_behavior_error(
                "External tool runs do not support asynchronous tool dispatch.",
                raw_response=None,
            )

        # Stream replay must not start the same tool call twice.
        previous = self.pending_calls.get(call.id)
        if previous is not None:
            if previous.call != call:
                raise _model_behavior_error(
                    CHANGED_TOOL_CALL_ARGUMENTS, raw_response=None
                )
            return

        # Resolve the handler from this turn so current permissions still apply.
        name = call.function.name or ""
        definition = (
            next((tool for tool in tools.executable_tools if tool.name == name), None)
            if tools
            else None
        )
        if (
            tools is None
            or tools.tool_choice == ToolChoice.NONE.value
            or definition is None
        ):
            raise _model_behavior_error(
                "Async call has no permitted local handler.", raw_response=None
            )

        # Check both budgets before reserving a call slot.
        limit = (tools.tool_call_limits or {}).get(name)
        if (
            limit is not None and self.tool_call_counts.get(name, 0) >= limit
        ) or self.tool_round_trips >= tools.max_tool_round_trips:
            raise _model_behavior_error(
                "Async call exceeds the tool budget.", raw_response=None
            )

        # Count the call before starting it; do not retry work after a timeout.
        self.tool_call_counts[name] = self.tool_call_counts.get(name, 0) + 1
        self.pending_calls[call.id] = _PendingToolCall(
            call,
            asyncio.create_task(
                self._execute_one_tool_call(
                    tools=tools,
                    tool_call=call,
                    tool_definition=definition.replace(retry_on_timeout=False),
                    parent_span=span,
                )
            ),
        )

    def _has_pending_results(self) -> bool:
        return any(not entry.delivered for entry in self.pending_calls.values())

    async def _cancel_pending_calls(self) -> None:
        for entry in self.pending_calls.values():
            if not entry.task.done():
                entry.task.cancel()

        # Wait for cancellation cleanup and observe errors from completed tasks.
        await asyncio.gather(
            *(entry.task for entry in self.pending_calls.values()),
            return_exceptions=True,
        )

    async def _finish_run(
        self,
        model_scope: AsyncExitStack,
        result: RunResult | None,
        run_error: BaseException | None,
    ) -> None:
        # Attempt every cleanup step even if an earlier step fails.
        cleanup_errors: list[BaseException] = []
        try:
            await model_scope.aclose()
        except BaseException as exc:
            cleanup_errors.append(exc)

        if self.shims is not None:
            try:
                await self.shims.on_run_end(result, self.transient_state)
            except BaseException as exc:
                cleanup_errors.append(exc)

        if self.run_events is not None:
            try:
                self.run_events.state_snapshot(
                    RunStateSnapshotBoundary.RUN_END, self.state
                )
            except BaseException as exc:
                cleanup_errors.append(exc)

        try:
            await self.hooks.on_agent_end(self.agent, result)
        except BaseException as exc:
            cleanup_errors.append(exc)

        # Keep the run failure primary; raise cleanup errors only after a successful
        # run.
        if cleanup_errors:
            if run_error is None:
                raise_cleanup_errors("Run cleanup failed.", cleanup_errors)

            LOGGER.warning(
                "Run cleanup failed after a run error",
                error_types=[type(error).__name__ for error in cleanup_errors],
            )

    async def _run_with_retry(
        self,
        *,
        model: Model[ModelResponse],
        messages: list[MessageDict],
        tools: Tools | None,
        model_args: dict[str, object] | None,
        parent_span: Span[Any] | None = None,
    ) -> ModelResponse:
        """Execute one model turn under the configured retry policy."""

        # Let the model own its retry policy when it exposes that capability.
        if model.handles_retries:
            return await self._run_once(
                model=model,
                messages=messages,
                tools=tools,
                model_args=model_args,
                parent_span=parent_span,
            )

        retry_result = await self._retryable_run_once(
            model=model,
            messages=messages,
            tools=tools,
            model_args=model_args,
            parent_span=parent_span,
        )

        return retry_result.result

    async def _run_once(
        self,
        *,
        model: Model[ModelResponse],
        messages: list[MessageDict],
        tools: Tools | None,
        model_args: dict[str, object] | None,
        parent_span: Span[Any] | None = None,
    ) -> ModelResponse:
        """Execute one single model attempt (no retry logic here)."""

        await self.hooks.on_llm_start(self.agent, messages)
        response = await model(
            messages,
            extra_call_args=model_args,
            schema=self.runner_task.schema,
            tools=tools,
            cancellation_token=self.cancellation_token,
            parent_span=parent_span,
        )
        await self.hooks.on_llm_end(self.agent, response)

        return response

    async def _stream_model_call(
        self,
        *,
        on_async_call: Callable[[ToolCall], None],
        model: Model[ModelResponse],
        messages: list[MessageDict],
        tools: Tools | None,
        model_args: dict[str, object] | None,
        emit: Callable[[ModelStreamEvent], None],
        parent_span: Span[Any] | None = None,
    ) -> tuple[ModelResponse, ModelStreamEvent]:
        """Execute one streaming model turn and return the completed response.

        Streaming runs intentionally avoid the outer retry wrapper used by
        terminal runs. Once live events have been emitted to the caller, a
        retry would replay another provider attempt on top of partial output.
        """
        completed_event: ModelStreamEvent | None = None

        await self.hooks.on_llm_start(self.agent, messages)
        async with (
            cancellation_scope(self.cancellation_token),
            AsyncExitStack() as scope,
        ):
            stream = model.stream_response(
                messages,
                extra_call_args=model_args,
                schema=self.runner_task.schema,
                tools=tools,
                cancellation_token=self.cancellation_token,
                parent_span=parent_span,
            )
            close = getattr(stream, "aclose", None)
            if callable(close):
                scope.push_async_callback(cast(Callable[[], Awaitable[None]], close))

            # Apply live inputs and dispatch tools while retaining the terminal
            # response.
            async for event in stream:
                if self._external_tools_enabled and event.input_messages:
                    raise _model_behavior_error(
                        "External tool runs do not support steering input.",
                        raw_response=event.response,
                    )
                if (
                    event.kind != ModelStreamEventKind.COMPLETED
                    and event.input_messages
                ):
                    # Queued steering precedes the tool outputs that resume the
                    # response.
                    offset = len(self.state.history)
                    while (
                        offset
                        and isinstance(self.state.history[offset - 1], dict)
                        and cast(MessageDict, self.state.history[offset - 1]).get(
                            "role"
                        )
                        == MessageRole.TOOL.value
                    ):
                        offset -= 1

                    self.state.history[offset:offset] = event.input_messages

                # Start a complete async call before waiting for more model output.
                if event.async_tool_call is not None:
                    on_async_call(event.async_tool_call)

                emit(event)
                if (
                    event.kind == ModelStreamEventKind.COMPLETED
                    and event.response is not None
                ):
                    completed_event = event

        # Validate the terminal contract before returning concrete values to the loop.
        if completed_event is None or completed_event.response is None:
            raise _model_behavior_error(
                "Streaming model call completed without a terminal response.",
                raw_response=None,
            )

        response = completed_event.response
        await self.hooks.on_llm_end(self.agent, response)

        return response, completed_event

    async def _execute_tool_calls(
        self,
        *,
        tools: Tools | None,
        tool_calls: list[ToolCall],
        response: ModelResponse,
        parent_span: Span[Any] | None = None,
    ) -> list[MessageDict]:
        """Execute one model-emitted tool batch and return tool messages."""
        if tools is None:
            raise _model_behavior_error(
                "Runner received tool calls, but the agent exposes no tools.",
                raw_response=response,
            )
        if tools.tool_choice == ToolChoice.NONE.value:
            raise _model_behavior_error(
                "Runner received tool calls even though tools were disabled for this turn.",
                raw_response=response,
            )

        # Validate the entire batch before any handler can have side effects.
        tool_definitions = {tool.name: tool for tool in tools.normalized_tools}
        for tool_call in tool_calls:
            if (tool_call.function.name or "") not in tool_definitions:
                raise _model_behavior_error(
                    "Runner received a tool call for an unknown tool.",
                    raw_response=response,
                )

        # History and round usage are already recorded; save the pending call.
        for call in tool_calls:
            definition = tool_definitions[call.function.name or ""]
            if isinstance(definition, ExternalTool):
                if (
                    not self._external_tools_enabled
                    or len(tool_calls) != 1
                    or self._has_pending_results()
                ):
                    raise _model_behavior_error(
                        "An external tool requires a sole call at an idle root boundary.",
                        raw_response=response,
                    )
                if (
                    type(self.state.turn_count) is not int
                    or type(self.state.revision) is not int
                ):
                    raise ValueError("Continuation counters must be integers.")
                arguments = _parse_external_tool_args(call, definition)
                self.tool_call_counts[definition.name] = (
                    self.tool_call_counts.get(definition.name, 0) + 1
                )
                raise RunPaused(
                    RunContinuation(
                        checkpoint=AgentSnapshot.capture(
                            agent_id=self.agent.id, run_state=self.state
                        ),
                        tool_call=call,
                        arguments=arguments,
                        args_schema=cast(dict[str, JsonValue], definition.schema),
                        tool_call_counts=self.tool_call_counts,
                        tool_round_trips=self.tool_round_trips,
                        max_turns=self.max_turns,
                        tool_call_limits=(
                            dict(tools.tool_call_limits)
                            if tools.tool_call_limits is not None
                            else None
                        ),
                        max_tool_round_trips=tools.max_tool_round_trips,
                    )
                )

        # Execute a tool call batch in parallel and return aggregated result
        if tools.parallel_tool_calls and len(tool_calls) > 1:
            return await asyncio.gather(
                *[
                    self._execute_one_tool_call(
                        tools=tools,
                        tool_call=tool_call,
                        tool_definition=tool_definitions[tool_call.function.name or ""],
                        parent_span=parent_span,
                    )
                    for tool_call in tool_calls
                ]
            )

        tool_messages: list[MessageDict] = []
        for tool_call in tool_calls:
            tool_messages.append(
                await self._execute_one_tool_call(
                    tools=tools,
                    tool_call=tool_call,
                    tool_definition=tool_definitions[tool_call.function.name or ""],
                    parent_span=parent_span,
                )
            )

        return tool_messages

    async def _execute_one_tool_call(
        self,
        *,
        tools: Tools,
        tool_call: ToolCall,
        tool_definition: ToolSpec[Any],
        parent_span: Span[Any] | None = None,
    ) -> MessageDict:
        """Execute one tool or delegated-agent call and return its tool message."""
        if isinstance(tool_definition, AgentTool):
            return await self._execute_agent_tool_call(
                tools=tools,
                tool_call=tool_call,
                tool_definition=tool_definition,
            )

        if isinstance(tool_definition, DefaultAgentTool):
            return await self._execute_default_agent_tool_call(
                tools=tools,
                tool_call=tool_call,
                tool_definition=tool_definition,
            )

        # Use the shared executor for local tools, hooks, cancellation, and tracing.
        if isinstance(tool_definition, Tool):
            tool_config = replace(tools, tools=(tool_definition,))
            context = ToolExecutionContext(
                run_id=str(self.agent.task_id),
                agent_name=self.runner_task.name,
                tool_call_id=tool_call.id,
                run_state=LiveRunStateView(self.state, self.agent.task_id),
            )
            tool_messages = await self.tool_executor.execute(
                tool_calls=[tool_call],
                tools=tool_config,
                cancellation_token=self.cancellation_token,
                on_tool_start=lambda started_call: self.hooks.on_tool_call_start(
                    self.agent,
                    started_call,
                ),
                on_tool_end=lambda ended_call, result: self.hooks.on_tool_call_end(
                    self.agent,
                    ended_call,
                    result,
                ),
                context={tool_call.id: context},
                tracing=self.model.tracing,
                parent_span=parent_span,
            )

            return tool_messages[0]

        raise _model_behavior_error(
            (
                "Runner received a non-executable tool call that "
                "it does not know how to execute."
            ),
            raw_response=None,
        )

    async def _execute_agent_tool_call(
        self,
        *,
        tools: Tools,
        tool_call: ToolCall,
        tool_definition: AgentTool,
    ) -> MessageDict:
        """Execute one agent-as-tool call through runtime messaging."""
        parsed_input = _parse_delegated_tool_args(
            tool_call=tool_call,
            args_model=tool_definition.args_type(),
        )

        await self.hooks.on_tool_call_start(self.agent, tool_call)
        delegated_result = await self._run_delegated_sub_agent(
            visible_tools=tools,
            tool_name=tool_definition.name,
            descriptor=tool_definition.descriptor,
            run_input=[parsed_input],
        )

        await self.hooks.on_tool_call_end(self.agent, tool_call, delegated_result)

        return _tool_result_message(
            tool_call=tool_call,
            tool_name=tool_definition.name,
            content=delegated_result,
        )

    async def _execute_default_agent_tool_call(
        self,
        *,
        tools: Tools,
        tool_call: ToolCall,
        tool_definition: DefaultAgentTool,
    ) -> MessageDict:
        """Execute one default spawned agent-as-tool call."""
        parsed_input = _parse_delegated_tool_args(
            tool_call=tool_call,
            args_model=tool_definition.args_type(),
        )

        await self.hooks.on_tool_call_start(self.agent, tool_call)
        delegated_result = await self._run_default_agent_tool(
            visible_tools=tools,
            tool_name=tool_definition.name,
            tool_definition=tool_definition,
            parsed_input=parsed_input,
        )

        await self.hooks.on_tool_call_end(self.agent, tool_call, delegated_result)

        return _tool_result_message(
            tool_call=tool_call,
            tool_name=tool_definition.name,
            content=delegated_result,
        )

    async def _execute_handoff(
        self,
        *,
        response: ModelResponse,
        handoff_call: ToolCall,
        tool_definition: HandoffTool | DefaultHandoffTool,
        tools: Tools,
    ) -> RunResult:
        """Transfer control to another agent and return its final result.

        Mirrors the run-loop modes: with ``emit`` the child run streams its
        model events through the same callback; without it the child runs to
        completion via runtime messaging.
        """
        parsed_input = _parse_delegated_tool_args(
            tool_call=handoff_call,
            args_model=tool_definition.args_type(),
        )

        # Resolve the child model and configuration before transferring state.
        if isinstance(tool_definition, HandoffTool):
            handoff_descriptor = _resolved_child_descriptor(
                runner_task=self.runner_task,
                parent_model=self.model,
                descriptor=tool_definition.descriptor,
            )
        else:
            config = tool_definition.config
            handoff_descriptor = AgentDescriptor(
                name=DEFAULT_HANDOFF_AGENT_NAME,
                description=DEFAULT_HANDOFF_AGENT_DESCRIPTION,
                model=config.model or self.model,
                instructions=config.instructions,
                model_args=(
                    dict(config.model_args)
                    if config.model_args is not None
                    else _model_args(self.runner_task)
                ),
                schema=config.schema,
                tools=config.tools,
            )

        # Pass inherited tool sources through the existing child shim policy.
        handoff_descriptor = replace(
            handoff_descriptor,
            shims=_child_shims(
                agent=self.agent,
                runner_task=self.runner_task,
                visible_tools=tools,
                policy=handoff_descriptor.tools,
                inherit_definitions=False,
            )
            + tuple(handoff_descriptor.shims or ()),
        )
        await self.hooks.on_tool_call_start(self.agent, handoff_call)
        if self.run_events is not None:
            self.run_events.handoff_start(
                task=self.agent,
                tool_call=handoff_call,
                target_name=handoff_descriptor.name,
            )

        transferred_state = replace(
            self.state,
            history=list(self.state.history),
            responses=list(self.state.responses),
        )

        # Handoff is a control-transfer primitive. The downstream agent should
        # see why control moved, so we preserve the parent assistant tool-call
        # turn, add a synthetic tool acknowledgement, and then add the optional
        # user-side delegation message.
        transferred_state.history.append(response)
        transferred_state.history.append(
            _tool_result_message(
                tool_call=handoff_call,
                tool_name=tool_definition.name,
                content=default_handoff_tool_result(tool_definition.name),
            )
        )
        transferred_state.history.append(
            parsed_input.task or default_handoff_task_message()
        )

        transferred_state.instructions = copy_instructions(
            handoff_descriptor.instructions
        )

        # Deliver through the runtime, forwarding events and controls for streamed runs.
        if self.emit is None:
            outcome = await self.agent_delivery.send(
                recipient=delegated_agent_id(
                    self.agent.id, tool_definition.name, kind=DelegationKind.HANDOFF
                ),
                descriptor=handoff_descriptor,
                run_input=transferred_state,
                parent_tools=self.runner_task.base_tools,
            )
            handoff_result = require_handoff_result(outcome)
        else:
            handoff_result = await self.agent_delivery.stream_handoff(
                recipient=delegated_agent_id(
                    self.agent.id, tool_definition.name, kind=DelegationKind.HANDOFF
                ),
                parent_tools=self.runner_task.base_tools,
                descriptor=handoff_descriptor,
                transferred_state=transferred_state,
                emit=self.emit,
                control_stream=self.control_stream,
            )

        # Adopt the child state so parent cleanup observes the final handoff outcome.
        if handoff_result.run_state is not None:
            child_state = handoff_result.run_state
            self.state.instructions = copy_instructions(child_state.instructions)
            self.state.history = list(child_state.history)
            self.state.responses = list(child_state.responses)
            self.state.shim_state = copy_shim_state(child_state.shim_state)
            self.state.turn_count = child_state.turn_count

        if self.run_events is not None:
            self.run_events.handoff_end(
                task=self.agent,
                tool_call=handoff_call,
                target_name=handoff_descriptor.name,
                result=handoff_result,
            )

        await self.hooks.on_tool_call_end(
            self.agent, handoff_call, handoff_result.final_output
        )

        return handoff_result

    async def _run_delegated_sub_agent(
        self,
        *,
        visible_tools: Tools,
        tool_name: str,
        descriptor: AgentDescriptor,
        run_input: list[RunHistoryItem],
    ) -> str:
        """Run one delegated sub-agent as a subroutine and return text output."""
        child_descriptor = _resolved_child_descriptor(
            runner_task=self.runner_task,
            parent_model=self.model,
            descriptor=descriptor,
        )
        child_descriptor = replace(
            child_descriptor,
            shims=_child_shims(
                agent=self.agent,
                runner_task=self.runner_task,
                visible_tools=visible_tools,
                policy=child_descriptor.tools,
                inherit_definitions=False,
            )
            + tuple(child_descriptor.shims or ()),
        )

        # Use runtime delivery so child registration and lifecycle stay centralized.
        outcome = await self.agent_delivery.send(
            recipient=delegated_agent_id(
                self.agent.id, tool_name, kind=DelegationKind.TOOL
            ),
            descriptor=child_descriptor,
            run_input=run_input,
            parent_tools=self.runner_task.base_tools,
        )

        return delegated_result_text(outcome)

    async def _run_default_agent_tool(
        self,
        *,
        visible_tools: Tools,
        tool_name: str,
        tool_definition: DefaultAgentTool,
        parsed_input: DefaultAgentToolInput,
    ) -> str:
        """Run one generic spawned helper agent as a subroutine."""

        # Reserve a child slot before building or running the delegated task.
        child_id = delegated_agent_id(
            self.agent.id, tool_name, kind=DelegationKind.TOOL
        )
        limit = await self.agent_delegation.reserve_child(
            parent_id=self.agent.id,
            child_id=child_id,
        )
        if limit == _AgentDelegationLimit.DEPTH:
            return AGENT_DEPTH_LIMIT_REACHED
        if limit == _AgentDelegationLimit.THREAD:
            return AGENT_THREAD_LIMIT_REACHED

        try:
            child_descriptor = AgentDescriptor(
                name=parsed_input.name,
                model=tool_definition.model or self.model,
                instructions=tool_definition.resolved_instructions(),
                model_args=(
                    dict(tool_definition.model_args)
                    if tool_definition.model_args is not None
                    else _model_args(self.runner_task)
                ),
                schema=tool_definition.output_schema,
                tools=tool_definition.tools,
                shims=_child_shims(
                    agent=self.agent,
                    runner_task=self.runner_task,
                    visible_tools=visible_tools,
                    policy=tool_definition.tools,
                    inherit_definitions=True,
                ),
            )
            outcome = await self.agent_delivery.send(
                recipient=child_id,
                descriptor=child_descriptor,
                run_input=[parsed_input.task],
                parent_tools=self.runner_task.base_tools,
            )

            return delegated_result_text(outcome)
        finally:
            # Release the slot after success, failure, or cancellation.
            await self.agent_delegation.release_child(child_id=child_id)


def _resolved_child_descriptor(
    *,
    runner_task: RunnerTask,
    parent_model: Model[ModelResponse],
    descriptor: AgentDescriptor,
) -> AgentDescriptor:
    """Fill child descriptor model defaults from the current agent."""
    model = descriptor.model or parent_model
    model_args = (
        dict(descriptor.model_args)
        if descriptor.model_args is not None
        else _model_args(runner_task)
    )

    return replace(
        descriptor,
        model=model,
        model_args=model_args,
    )


def _parse_external_tool_args(
    tool_call: ToolCall, definition: ExternalTool[Any]
) -> dict[str, JsonValue]:
    """Validate an external call against the registered argument model."""
    try:
        arguments = definition.args_type().model_validate_json(
            tool_call.function.arguments or "{}"
        )
    except ValidationError as exc:
        raise _model_behavior_error(
            "External tool arguments do not match the registered schema.",
            raw_response=None,
        ) from exc

    return cast(dict[str, JsonValue], arguments.model_dump(mode="json"))


def _parse_delegated_tool_args[ArgsT: BaseModel](
    *,
    tool_call: ToolCall,
    args_model: type[ArgsT],
) -> ArgsT:
    """Validate one delegated-agent tool call into structured task input."""
    raw_arguments = tool_call.function.arguments or "{}"
    try:
        parsed_arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ModelBehaviorError(
            f"Invalid arguments for delegated tool '{tool_call.function.name}': {exc}"
        ) from exc

    try:
        return args_model(**parsed_arguments)
    except ValidationError as exc:
        raise ModelBehaviorError(
            f"Invalid arguments for delegated tool '{tool_call.function.name}': {exc}"
        ) from exc


def _child_shims(
    *,
    agent: Task,
    runner_task: RunnerTask,
    visible_tools: Tools,
    policy: ToolConfig,
    inherit_definitions: bool,
) -> tuple[Shim, ...]:
    """Resolve inherited dynamic sources separately from child-local tools."""
    visible_names = frozenset(tool.name for tool in visible_tools.normalized_tools)
    names: frozenset[str]
    if isinstance(policy, InheritTools):
        names = visible_names
    elif isinstance(policy, RestrictTools):
        names = visible_names.intersection(policy.names)
    else:
        names = frozenset()

    # Use the bound manager to retain only the child-visible tool sources.
    manager = _shim_manager(agent)
    if manager is not None:
        return manager.child_shims(names, inherit_definitions=inherit_definitions)

    return tuple(runner_task.shims or ()) if inherit_definitions else ()


def _tool_result_message(
    *,
    tool_call: ToolCall,
    tool_name: str,
    content: object,
) -> MessageDict:
    """Return the canonical tool-result message for one completed call."""
    return {
        "role": MessageRole.TOOL.value,
        "tool_call_id": tool_call.id,
        "name": tool_name,
        "content": normalize_message_content(content),
    }


def _shim_manager(agent: Task) -> BoundShimManager | None:
    """Return the bound shim manager exposed by the task, if any."""
    manager = getattr(agent, "bound_shim_manager", None)
    if not isinstance(manager, BoundShimManager):
        return None

    return manager


def _model_args(runner_task: RunnerTask) -> dict[str, object] | None:
    """Return a defensive copy of the agent's model-call arguments."""
    model_args = runner_task.model_args
    if model_args is None:
        return None

    # Copy so per-call mutations (e.g. provider-side updates) never leak
    # back into the agent descriptor.
    return dict(model_args)


def _limit_tools(
    tools: Tools,
    tool_call_counts: dict[str, int],
    tool_round_trips: int,
) -> Tools:
    """Apply tool visibility limits using incremental counters."""
    if not tools.normalized_tools:
        return replace(tools, tool_choice=ToolChoice.NONE.value)
    if tools.tool_choice == ToolChoice.NONE.value:
        return tools

    if tool_round_trips >= tools.max_tool_round_trips:
        return replace(tools, tools=(), tool_choice=ToolChoice.NONE.value)

    if not tools.tool_call_limits:
        return tools

    active_tools = tuple(
        tool
        for tool in tools.normalized_tools
        if tool.name not in tools.tool_call_limits
        or tool_call_counts.get(tool.name, 0) < tools.tool_call_limits[tool.name]
    )
    if not active_tools:
        return replace(tools, tools=(), tool_choice=ToolChoice.NONE.value)
    if len(active_tools) == len(tools.normalized_tools):
        return tools

    # Strip exhausted tools from later requests.
    return replace(tools, tools=active_tools)


def _model_behavior_error(
    message: str,
    *,
    raw_response: ModelResponse | None,
) -> ModelBehaviorError:
    """Create a ``ModelBehaviorError`` annotated with raw response details."""
    error = ModelBehaviorError(message)
    error.run_data = RunErrorDetails(raw_response=raw_response)

    return error
