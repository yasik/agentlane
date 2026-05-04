"""Default runner for the harness agent loop.

The runner owns the generic loop for one agent run:

1. Build the next model request from instructions + run state.
2. Call the model (with optional retry).
3. Record the raw ``ModelResponse`` on the run state.
4. If tool calls were returned, execute them and continue the loop.
5. Otherwise extract the direct assistant answer and return ``RunResult``.

The lifecycle stays responsible for queueing and persistence. The runner
stays responsible for model-facing normalization, tool execution, and loop
control within one run.
"""

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from typing import Any, Literal, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ValidationError

from agentlane.messaging import AgentId
from agentlane.models import (
    MessageDict,
    Model,
    ModelBehaviorError,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    OutputSchema,
    PromptSpec,
    RunErrorDetails,
    Tool,
    ToolCall,
    ToolExecutor,
    Tools,
    ToolSpec,
    retry_on_errors,
)
from agentlane.models.run import DefaultRunContext
from agentlane.runtime import CancellationToken, RuntimeEngine

from ._cancellation import cancel_task_callback
from ._handoff import (
    DefaultAgentToolInput,
    DelegatedTaskInput,
    default_handoff_task_message,
    default_handoff_tool_result,
    delegated_agent_id,
    delegated_agent_type,
    delegated_result_text,
    require_handoff_result,
)
from ._hooks import RunnerHooks, coerce_runner_hooks
from ._lifecycle import (
    AgentDescriptor,
    AgentTool,
    DefaultAgentTool,
    DefaultHandoffTool,
    HandoffTool,
)
from ._run import (
    RunHistoryItem,
    RunInput,
    RunResult,
    RunState,
    copy_instructions,
    copy_run_state,
    copy_shim_state,
)
from ._stream import RunStream
from ._task import Task
from ._tooling import InheritTools, RestrictTools
from .shims import PreparedTurn, Shim
from .shims._manager import BoundShimManager
from .tools import HarnessToolsShim, base_harness_tools

_AGENT_DEPTH_LIMIT_REACHED = "Agent depth limit reached. Solve the task yourself."
_AGENT_THREAD_LIMIT_REACHED = "Agent thread limit reached. Solve the task yourself."
type _AgentDelegationLimit = Literal["depth", "thread"]


@runtime_checkable
class RunnerTask(Protocol):
    """Structural protocol for tasks compatible with the default runner.

    The runner accesses these properties during request building and model
    calls. Any task that exposes them is compatible — no subclassing needed.
    """

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


@runtime_checkable
class _StreamRunnableAgent(Protocol):
    """Structural protocol for local streamed harness delivery."""

    async def enqueue_input_stream(
        self,
        run_input: RunInput,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> RunStream: ...


def _agent_depth_map() -> dict[AgentId, int]:
    """Return a fresh spawned-agent depth map."""
    return {}


@dataclass(slots=True)
class _AgentDelegationState:
    """Process-local guard state for generic spawned agent calls."""

    max_depth: int
    max_threads: int
    _live_agents: int = 0
    _depth_by_agent: dict[AgentId, int] = field(default_factory=_agent_depth_map)
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
                return "depth"
            if self._live_agents >= self.max_threads:
                return "thread"

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


class Runner:
    """Default runner for harness agents.

    A single ``Runner`` instance can be shared across multiple agents
    safely. It holds configuration plus process-local execution guards, but
    never persists conversation state.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 1,
        max_turns: int = 128,
        agent_max_depth: int = 4,
        agent_max_threads: int = 16,
        is_retryable: Callable[[BaseException], bool] | None = None,
    ) -> None:
        """Initialize retry and loop limits for one reusable runner.

        Args:
            max_attempts: Total attempts per model call (1 = no retry).
            max_turns: Safety cap on conversation turns per run.
            agent_max_depth: Inclusive process-local recursive depth cap for
                generic spawned ``agent`` tool calls. A direct child has depth
                1.
            agent_max_threads: Process-local live-agent cap for generic
                spawned ``agent`` tool calls.
            is_retryable: Optional predicate for retry eligibility. Defaults
                to the standard HTTP-status-code check in ``retry_on_errors``.
        """
        if max_attempts < 1:
            raise ValueError("Runner.max_attempts must be at least 1.")
        if max_turns < 1:
            raise ValueError("Runner.max_turns must be at least 1.")
        if agent_max_depth < 1:
            raise ValueError("Runner.agent_max_depth must be at least 1.")
        if agent_max_threads < 1:
            raise ValueError("Runner.agent_max_threads must be at least 1.")

        self._max_attempts = max_attempts
        self._max_turns = max_turns
        self._is_retryable = is_retryable
        self._tool_executor = ToolExecutor()
        self._agent_delegation = _AgentDelegationState(
            max_depth=agent_max_depth,
            max_threads=agent_max_threads,
        )

        # Cache the retry-wrapped model call once at init so we don't
        # recreate the decorator closure on every turn.
        self._retryable_run_once = retry_on_errors(
            max_retries=self._max_attempts,
            is_retryable=self._is_retryable,
        )(self._run_once)

    async def run(
        self,
        agent: Task,
        state: RunState,
        *,
        hooks: RunnerHooks | Sequence[RunnerHooks] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        """Execute the generic harness loop for the provided run state.

        The lifecycle passes a private working copy of ``state``, so the
        runner mutates it freely (incrementing ``turn_count``, appending
        responses) without risking the persisted baseline.
        """
        resolved_hooks = coerce_runner_hooks(hooks)
        result: RunResult | None = None

        # Narrow the agent to the runner protocol once per run. All helper
        # functions receive the narrowed value instead of re-checking.
        runner_task = _narrow_runner_task(agent)
        shim_manager = _shim_manager(agent)
        transient_state = DefaultRunContext()

        # Incremental tool-usage counters — updated after each tool batch
        # so we never re-scan all prior responses.
        tool_call_counts: dict[str, int] = {}
        tool_round_trips = 0

        # Hook receives the same working copy — safe because the lifecycle
        # already isolated it before calling us.
        await resolved_hooks.on_agent_start(agent, state)
        if shim_manager is not None:
            await shim_manager.on_run_start(state, transient_state)
        try:
            while True:
                state.turn_count += 1
                _check_turn_limit(state.turn_count, self._max_turns)
                prepared_turn = PreparedTurn(
                    run_state=state,
                    tools=_visible_tools(
                        runner_task, tool_call_counts, tool_round_trips
                    ),
                    model_args=_model_args(runner_task),
                    transient_state=transient_state,
                )
                if shim_manager is not None:
                    await shim_manager.prepare_turn(prepared_turn)

                messages = _build_request(prepared_turn)
                if shim_manager is not None:
                    messages = await shim_manager.transform_messages(
                        prepared_turn,
                        messages,
                    )
                response = await self._run_with_retry(
                    agent=agent,
                    runner_task=runner_task,
                    messages=messages,
                    tools=prepared_turn.tools,
                    model_args=prepared_turn.model_args,
                    hooks=resolved_hooks,
                    cancellation_token=cancellation_token,
                )

                state.responses.append(response)
                if shim_manager is not None:
                    await shim_manager.on_model_response(prepared_turn, response)
                tool_calls = _extract_tool_calls(response)
                if tool_calls:
                    handoff_call = _extract_handoff_call(
                        tools=prepared_turn.tools,
                        tool_calls=tool_calls,
                    )
                    if handoff_call is not None:
                        return await self._execute_handoff(
                            agent=agent,
                            runner_task=runner_task,
                            state=state,
                            response=response,
                            handoff_call=handoff_call,
                            tools=prepared_turn.tools,
                            hooks=resolved_hooks,
                            cancellation_token=cancellation_token,
                        )

                    # Tool and sub-agent calls continue the same loop, so the
                    # raw assistant turn is committed before execution and fed
                    # back into the next model request together with the tool
                    # result messages.
                    state.history.append(response)
                    tool_messages = await self._execute_tool_calls(
                        agent=agent,
                        runner_task=runner_task,
                        tools=prepared_turn.tools,
                        tool_calls=tool_calls,
                        response=response,
                        hooks=resolved_hooks,
                        cancellation_token=cancellation_token,
                    )
                    state.history.extend(tool_messages)

                    # Update incremental counters from this batch
                    tool_round_trips += 1
                    for tc in tool_calls:
                        name = tc.function.name or ""
                        tool_call_counts[name] = tool_call_counts.get(name, 0) + 1

                    continue

                state.history.append(response)
                _validate_terminal_response(response)
                result = RunResult(
                    final_output=_extract_direct_answer(response),
                    responses=list(state.responses),
                    turn_count=state.turn_count,
                    run_state=copy_run_state(state),
                )
                return result
        finally:
            if shim_manager is not None:
                await shim_manager.on_run_end(result, transient_state)
            # Always fire the end hook — result is None if the loop raised.
            await resolved_hooks.on_agent_end(agent, result)

    def run_stream(
        self,
        agent: Task,
        state: RunState,
        *,
        hooks: RunnerHooks | Sequence[RunnerHooks] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> RunStream:
        """Execute the generic harness loop with live model streaming."""
        resolved_hooks = coerce_runner_hooks(hooks)
        stream = RunStream(
            on_close=(
                cancellation_token.cancel if cancellation_token is not None else None
            )
        )
        stream_task = asyncio.create_task(
            self._run_stream_task(
                agent=agent,
                state=state,
                stream=stream,
                hooks=resolved_hooks,
                cancellation_token=cancellation_token,
            )
        )
        stream.add_cleanup(cancel_task_callback(stream_task))
        return stream

    async def _run_stream_task(
        self,
        *,
        agent: Task,
        state: RunState,
        stream: RunStream,
        hooks: RunnerHooks,
        cancellation_token: CancellationToken | None,
    ) -> None:
        """Drive one streamed run and resolve the provided stream handle."""
        try:
            result = await self._run_stream_internal(
                agent=agent,
                state=state,
                emit=stream.emit,
                hooks=hooks,
                cancellation_token=cancellation_token,
            )
        except Exception as exc:
            stream.fail(exc)
        except BaseException as exc:
            stream.fail(exc)
            raise
        else:
            stream.finish(result)

    async def _run_stream_internal(
        self,
        *,
        agent: Task,
        state: RunState,
        emit: Callable[[ModelStreamEvent], None],
        hooks: RunnerHooks,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        """Execute the generic harness loop while forwarding model events."""
        result: RunResult | None = None
        runner_task = _narrow_runner_task(agent)
        shim_manager = _shim_manager(agent)
        transient_state = DefaultRunContext()
        tool_call_counts: dict[str, int] = {}
        tool_round_trips = 0

        await hooks.on_agent_start(agent, state)
        if shim_manager is not None:
            await shim_manager.on_run_start(state, transient_state)
        try:
            while True:
                state.turn_count += 1
                _check_turn_limit(state.turn_count, self._max_turns)
                prepared_turn = PreparedTurn(
                    run_state=state,
                    tools=_visible_tools(
                        runner_task, tool_call_counts, tool_round_trips
                    ),
                    model_args=_model_args(runner_task),
                    transient_state=transient_state,
                )
                if shim_manager is not None:
                    await shim_manager.prepare_turn(prepared_turn)

                messages = _build_request(prepared_turn)
                if shim_manager is not None:
                    messages = await shim_manager.transform_messages(
                        prepared_turn,
                        messages,
                    )
                response = await self._stream_model_call(
                    agent=agent,
                    runner_task=runner_task,
                    messages=messages,
                    tools=prepared_turn.tools,
                    model_args=prepared_turn.model_args,
                    emit=emit,
                    hooks=hooks,
                    cancellation_token=cancellation_token,
                )

                state.responses.append(response)
                if shim_manager is not None:
                    await shim_manager.on_model_response(prepared_turn, response)
                tool_calls = _extract_tool_calls(response)
                if tool_calls:
                    handoff_call = _extract_handoff_call(
                        tools=prepared_turn.tools,
                        tool_calls=tool_calls,
                    )
                    if handoff_call is not None:
                        return await self._execute_handoff_stream(
                            agent=agent,
                            runner_task=runner_task,
                            state=state,
                            response=response,
                            handoff_call=handoff_call,
                            tools=prepared_turn.tools,
                            emit=emit,
                            hooks=hooks,
                            cancellation_token=cancellation_token,
                        )

                    state.history.append(response)
                    tool_messages = await self._execute_tool_calls(
                        agent=agent,
                        runner_task=runner_task,
                        tools=prepared_turn.tools,
                        tool_calls=tool_calls,
                        response=response,
                        hooks=hooks,
                        cancellation_token=cancellation_token,
                    )
                    state.history.extend(tool_messages)

                    tool_round_trips += 1
                    for tc in tool_calls:
                        name = tc.function.name or ""
                        tool_call_counts[name] = tool_call_counts.get(name, 0) + 1

                    continue

                state.history.append(response)
                _validate_terminal_response(response)
                result = RunResult(
                    final_output=_extract_direct_answer(response),
                    responses=list(state.responses),
                    turn_count=state.turn_count,
                    run_state=copy_run_state(state),
                )
                return result
        finally:
            if shim_manager is not None:
                await shim_manager.on_run_end(result, transient_state)
            await hooks.on_agent_end(agent, result)

    async def _run_with_retry(
        self,
        *,
        agent: Task,
        runner_task: RunnerTask | None,
        messages: list[MessageDict],
        tools: Tools | None,
        model_args: dict[str, object] | None,
        hooks: RunnerHooks,
        cancellation_token: CancellationToken | None,
    ) -> ModelResponse:
        """Execute one model turn under the configured retry policy."""
        retry_result = await self._retryable_run_once(
            agent=agent,
            runner_task=runner_task,
            messages=messages,
            tools=tools,
            model_args=model_args,
            hooks=hooks,
            cancellation_token=cancellation_token,
        )
        return retry_result.result

    async def _run_once(
        self,
        *,
        agent: Task,
        runner_task: RunnerTask | None,
        messages: list[MessageDict],
        tools: Tools | None,
        model_args: dict[str, object] | None,
        hooks: RunnerHooks,
        cancellation_token: CancellationToken | None,
    ) -> ModelResponse:
        """Execute one single model attempt (no retry logic here)."""
        model = _require_model(runner_task)

        await hooks.on_llm_start(agent, messages)
        response = await model(
            messages,
            extra_call_args=model_args,
            schema=_schema(runner_task),
            tools=tools,
            cancellation_token=cancellation_token,
        )
        await hooks.on_llm_end(agent, response)

        return response

    async def _stream_model_call(
        self,
        *,
        agent: Task,
        runner_task: RunnerTask | None,
        messages: list[MessageDict],
        tools: Tools | None,
        model_args: dict[str, object] | None,
        emit: Callable[[ModelStreamEvent], None],
        hooks: RunnerHooks,
        cancellation_token: CancellationToken | None,
    ) -> ModelResponse:
        """Execute one streaming model turn and return the completed response.

        Streaming runs intentionally avoid the outer retry wrapper used by
        terminal runs. Once live events have been emitted to the caller, a
        retry would replay another provider attempt on top of partial output.
        """
        model = _require_model(runner_task)
        completed_response: ModelResponse | None = None

        await hooks.on_llm_start(agent, messages)
        async for event in model.stream_response(
            messages,
            extra_call_args=model_args,
            schema=_schema(runner_task),
            tools=tools,
            cancellation_token=cancellation_token,
        ):
            emit(event)
            if (
                event.kind == ModelStreamEventKind.COMPLETED
                and event.response is not None
            ):
                completed_response = event.response

        if completed_response is None:
            raise _model_behavior_error(
                "Streaming model call completed without a terminal response.",
                raw_response=None,
            )

        await hooks.on_llm_end(agent, completed_response)
        return completed_response

    async def _execute_tool_calls(
        self,
        *,
        agent: Task,
        runner_task: RunnerTask | None,
        tools: Tools | None,
        tool_calls: list[ToolCall],
        response: ModelResponse,
        hooks: RunnerHooks,
        cancellation_token: CancellationToken | None,
    ) -> list[MessageDict]:
        """Execute one model-emitted tool batch and return tool messages."""
        if tools is None:
            raise _model_behavior_error(
                "Runner received tool calls, but the agent exposes no tools.",
                raw_response=response,
            )
        if tools.tool_choice == "none":
            raise _model_behavior_error(
                "Runner received tool calls even though tools were disabled for this turn.",
                raw_response=response,
            )

        tool_definitions = {tool.name: tool for tool in tools.normalized_tools}
        for tool_call in tool_calls:
            if (tool_call.function.name or "") not in tool_definitions:
                raise _model_behavior_error(
                    "Runner received a tool call for an unknown tool.",
                    raw_response=response,
                )

        # Execute a tool call batch in parallel and return aggregated result
        if tools.parallel_tool_calls and len(tool_calls) > 1:
            return await asyncio.gather(
                *[
                    self._execute_one_tool_call(
                        agent=agent,
                        runner_task=runner_task,
                        tools=tools,
                        tool_call=tool_call,
                        tool_definition=tool_definitions[tool_call.function.name or ""],
                        hooks=hooks,
                        cancellation_token=cancellation_token,
                    )
                    for tool_call in tool_calls
                ]
            )

        tool_messages: list[MessageDict] = []
        for tool_call in tool_calls:
            tool_messages.append(
                await self._execute_one_tool_call(
                    agent=agent,
                    runner_task=runner_task,
                    tools=tools,
                    tool_call=tool_call,
                    tool_definition=tool_definitions[tool_call.function.name or ""],
                    hooks=hooks,
                    cancellation_token=cancellation_token,
                )
            )
        return tool_messages

    async def _execute_one_tool_call(
        self,
        *,
        agent: Task,
        runner_task: RunnerTask | None,
        tools: Tools,
        tool_call: ToolCall,
        tool_definition: ToolSpec[Any],
        hooks: RunnerHooks,
        cancellation_token: CancellationToken | None,
    ) -> MessageDict:
        """Execute one tool or delegated-agent call and return its tool message."""
        if isinstance(tool_definition, AgentTool):
            return await self._execute_agent_tool_call(
                agent=agent,
                runner_task=runner_task,
                tool_call=tool_call,
                tool_definition=tool_definition,
                hooks=hooks,
                cancellation_token=cancellation_token,
            )

        if isinstance(tool_definition, DefaultAgentTool):
            return await self._execute_default_agent_tool_call(
                agent=agent,
                runner_task=runner_task,
                tool_call=tool_call,
                tool_definition=tool_definition,
                hooks=hooks,
                cancellation_token=cancellation_token,
            )

        if isinstance(tool_definition, Tool):
            tool_config = replace(tools, tools=(tool_definition,))
            tool_messages = await self._tool_executor.execute(
                tool_calls=[tool_call],
                tools=tool_config,
                cancellation_token=cancellation_token,
                on_tool_start=lambda started_call: hooks.on_tool_call_start(
                    agent,
                    started_call,
                ),
                on_tool_end=lambda ended_call, result: hooks.on_tool_call_end(
                    agent,
                    ended_call,
                    result,
                ),
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
        agent: Task,
        runner_task: RunnerTask | None,
        tool_call: ToolCall,
        tool_definition: AgentTool,
        hooks: RunnerHooks,
        cancellation_token: CancellationToken | None,
    ) -> MessageDict:
        """Execute one agent-as-tool call through runtime messaging."""
        parsed_input = _parse_delegated_tool_args(
            tool_call=tool_call,
            args_model=tool_definition.args_type(),
        )
        await hooks.on_tool_call_start(agent, tool_call)
        delegated_result = await self._run_delegated_sub_agent(
            agent=agent,
            runner_task=runner_task,
            tool_name=tool_definition.name,
            descriptor=tool_definition.descriptor,
            run_input=_agent_tool_run_input(parsed_input),
            cancellation_token=cancellation_token,
        )

        await hooks.on_tool_call_end(agent, tool_call, delegated_result)
        return _tool_result_message(
            tool_call=tool_call,
            tool_name=tool_definition.name,
            content=delegated_result,
        )

    async def _execute_default_agent_tool_call(
        self,
        *,
        agent: Task,
        runner_task: RunnerTask | None,
        tool_call: ToolCall,
        tool_definition: DefaultAgentTool,
        hooks: RunnerHooks,
        cancellation_token: CancellationToken | None,
    ) -> MessageDict:
        """Execute one default spawned agent-as-tool call."""
        parsed_input = _parse_default_agent_tool_input(
            tool_call=tool_call,
            args_model=tool_definition.args_type(),
        )
        await hooks.on_tool_call_start(agent, tool_call)
        delegated_result = await self._run_default_agent_tool(
            agent=agent,
            runner_task=runner_task,
            tool_name=tool_definition.name,
            tool_definition=tool_definition,
            parsed_input=parsed_input,
            cancellation_token=cancellation_token,
        )

        await hooks.on_tool_call_end(agent, tool_call, delegated_result)
        return _tool_result_message(
            tool_call=tool_call,
            tool_name=tool_definition.name,
            content=delegated_result,
        )

    async def _execute_handoff(
        self,
        *,
        agent: Task,
        runner_task: RunnerTask | None,
        state: RunState,
        response: ModelResponse,
        handoff_call: ToolCall,
        tools: Tools | None,
        hooks: RunnerHooks,
        cancellation_token: CancellationToken | None,
    ) -> RunResult:
        """Transfer control to another agent and return its final result."""
        if tools is None:
            raise _model_behavior_error(
                "Runner received a handoff call, but the agent exposes no tools.",
                raw_response=response,
            )

        tool_definition = _require_handoff_tool_definition(
            tools=tools,
            tool_call=handoff_call,
            response=response,
        )
        parsed_input = _parse_delegated_task_input(
            tool_call=handoff_call,
            args_model=tool_definition.args_type(),
        )
        await hooks.on_tool_call_start(agent, handoff_call)

        transferred_state = replace(
            state,
            history=list(state.history),
            responses=list(state.responses),
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

        handoff_descriptor = _resolved_handoff_descriptor(
            runner_task=runner_task,
            tool_definition=tool_definition,
        )
        transferred_state.instructions = copy_instructions(
            handoff_descriptor.instructions
        )
        handoff_result = await self._deliver_handoff(
            agent=agent,
            runner_task=runner_task,
            tool_name=tool_definition.name,
            descriptor=handoff_descriptor,
            transferred_state=transferred_state,
            cancellation_token=cancellation_token,
        )

        if handoff_result.run_state is not None:
            _overwrite_run_state(state, handoff_result.run_state)

        await hooks.on_tool_call_end(agent, handoff_call, handoff_result.final_output)
        return handoff_result

    async def _execute_handoff_stream(
        self,
        *,
        agent: Task,
        runner_task: RunnerTask | None,
        state: RunState,
        response: ModelResponse,
        handoff_call: ToolCall,
        tools: Tools | None,
        emit: Callable[[ModelStreamEvent], None],
        hooks: RunnerHooks,
        cancellation_token: CancellationToken | None,
    ) -> RunResult:
        """Transfer control to another agent and continue streaming there."""
        if tools is None:
            raise _model_behavior_error(
                "Runner received a handoff call, but the agent exposes no tools.",
                raw_response=response,
            )

        tool_definition = _require_handoff_tool_definition(
            tools=tools,
            tool_call=handoff_call,
            response=response,
        )
        parsed_input = _parse_delegated_task_input(
            tool_call=handoff_call,
            args_model=tool_definition.args_type(),
        )
        await hooks.on_tool_call_start(agent, handoff_call)

        transferred_state = replace(
            state,
            history=list(state.history),
            responses=list(state.responses),
        )
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

        handoff_descriptor = _resolved_handoff_descriptor(
            runner_task=runner_task,
            tool_definition=tool_definition,
        )
        transferred_state.instructions = copy_instructions(
            handoff_descriptor.instructions
        )
        handoff_result = await self._deliver_handoff_stream(
            agent=agent,
            runner_task=runner_task,
            tool_name=tool_definition.name,
            descriptor=handoff_descriptor,
            transferred_state=transferred_state,
            emit=emit,
            cancellation_token=cancellation_token,
        )

        if handoff_result.run_state is not None:
            _overwrite_run_state(state, handoff_result.run_state)

        await hooks.on_tool_call_end(agent, handoff_call, handoff_result.final_output)
        return handoff_result

    async def _run_delegated_sub_agent(
        self,
        *,
        agent: Task,
        runner_task: RunnerTask | None,
        tool_name: str,
        descriptor: AgentDescriptor,
        run_input: list[RunHistoryItem],
        cancellation_token: CancellationToken | None,
    ) -> str:
        """Run one delegated sub-agent as a subroutine and return text output."""
        runtime = _require_runtime_engine(agent)
        child_descriptor = _resolved_child_descriptor(
            runner_task=runner_task,
            descriptor=descriptor,
        )
        runtime.register_factory(
            delegated_agent_type(agent.id, tool_name, kind="tool"),
            type(agent).create_factory(
                runner=self,
                descriptor=child_descriptor,
                parent_tools=_base_tools(runner_task),
            ),
        )
        outcome = await agent.send_message(
            run_input,
            recipient=delegated_agent_id(agent.id, tool_name, kind="tool"),
            cancellation_token=cancellation_token,
        )
        return delegated_result_text(outcome)

    async def _run_default_agent_tool(
        self,
        *,
        agent: Task,
        runner_task: RunnerTask | None,
        tool_name: str,
        tool_definition: DefaultAgentTool,
        parsed_input: DefaultAgentToolInput,
        cancellation_token: CancellationToken | None,
    ) -> str:
        """Run one generic spawned helper agent as a subroutine."""
        child_id = delegated_agent_id(agent.id, tool_name, kind="tool")
        limit = await self._agent_delegation.reserve_child(
            parent_id=agent.id,
            child_id=child_id,
        )
        if limit == "depth":
            return _AGENT_DEPTH_LIMIT_REACHED
        if limit == "thread":
            return _AGENT_THREAD_LIMIT_REACHED

        try:
            runtime = _require_runtime_engine(agent)
            child_descriptor = AgentDescriptor(
                name=parsed_input.name,
                model=tool_definition.model or _require_model(runner_task),
                instructions=tool_definition.resolved_instructions(),
                model_args=(
                    dict(tool_definition.model_args)
                    if tool_definition.model_args is not None
                    else _model_args(runner_task)
                ),
                schema=tool_definition.output_schema,
                tools=tool_definition.tools,
                shims=_default_agent_child_shims(tool_definition),
            )
            runtime.register_factory(
                delegated_agent_type(agent.id, tool_name, kind="tool"),
                type(agent).create_factory(
                    runner=self,
                    descriptor=child_descriptor,
                    parent_tools=_base_tools(runner_task),
                ),
            )
            outcome = await agent.send_message(
                _default_agent_tool_run_input(parsed_input),
                recipient=child_id,
                cancellation_token=cancellation_token,
            )
            return delegated_result_text(outcome)
        finally:
            await self._agent_delegation.release_child(child_id=child_id)

    async def _deliver_handoff(
        self,
        *,
        agent: Task,
        runner_task: RunnerTask | None,
        tool_name: str,
        descriptor: AgentDescriptor,
        transferred_state: RunState,
        cancellation_token: CancellationToken | None,
    ) -> RunResult:
        """Transfer the run to another agent and wait for its final result."""
        runtime = _require_runtime_engine(agent)
        runtime.register_factory(
            delegated_agent_type(agent.id, tool_name, kind="handoff"),
            type(agent).create_factory(
                runner=self,
                descriptor=descriptor,
                parent_tools=_base_tools(runner_task),
            ),
        )
        outcome = await agent.send_message(
            transferred_state,
            recipient=delegated_agent_id(agent.id, tool_name, kind="handoff"),
            cancellation_token=cancellation_token,
        )
        return require_handoff_result(outcome)

    async def _deliver_handoff_stream(
        self,
        *,
        agent: Task,
        runner_task: RunnerTask | None,
        tool_name: str,
        descriptor: AgentDescriptor,
        transferred_state: RunState,
        emit: Callable[[ModelStreamEvent], None],
        cancellation_token: CancellationToken | None,
    ) -> RunResult:
        """Transfer the run locally to another agent and stream its events."""
        runtime = _require_runtime_engine(agent)
        bound_agent = type(agent).bind(
            runtime,
            delegated_agent_id(agent.id, tool_name, kind="handoff"),
            runner=self,
            descriptor=descriptor,
            parent_tools=_base_tools(runner_task),
        )
        if not isinstance(bound_agent, _StreamRunnableAgent):
            raise RuntimeError(
                "Streamed handoff requires the child agent to support "
                "`enqueue_input_stream`."
            )
        child_agent = bound_agent
        child_stream = await child_agent.enqueue_input_stream(
            transferred_state,
            cancellation_token=cancellation_token,
        )
        async for event in child_stream:
            emit(event)
        return await child_stream.result()


def _extract_handoff_call(
    *,
    tools: Tools | None,
    tool_calls: list[ToolCall],
) -> ToolCall | None:
    """Return the handoff call when the model requested one."""
    if tools is None:
        return None

    handoff_calls = [
        tool_call
        for tool_call in tool_calls
        if isinstance(
            _tool_definition_by_name(tools, tool_call.function.name or ""),
            (HandoffTool, DefaultHandoffTool),
        )
    ]
    if not handoff_calls:
        return None

    if len(handoff_calls) > 1 or len(tool_calls) > 1:
        raise ModelBehaviorError(
            "Runner received a handoff together with additional tool calls. "
            "First-class handoff must be the only tool call in the turn."
        )

    return handoff_calls[0]


def _tool_definition_by_name(
    tools: Tools,
    tool_name: str,
) -> ToolSpec[Any] | None:
    """Return one visible tool definition by name."""
    for tool in tools.normalized_tools:
        if tool.name == tool_name:
            return tool
    return None


def _require_handoff_tool_definition(
    *,
    tools: Tools,
    tool_call: ToolCall,
    response: ModelResponse,
) -> HandoffTool | DefaultHandoffTool:
    """Return the handoff tool definition for the intercepted tool call."""
    tool_definition = _tool_definition_by_name(tools, tool_call.function.name or "")
    if isinstance(tool_definition, (HandoffTool, DefaultHandoffTool)):
        return tool_definition
    raise _model_behavior_error(
        "Runner expected the intercepted handoff call to resolve to a handoff tool.",
        raw_response=response,
    )


def _resolved_handoff_descriptor(
    *,
    runner_task: RunnerTask | None,
    tool_definition: HandoffTool | DefaultHandoffTool,
) -> AgentDescriptor:
    """Return the resolved descriptor for one intercepted handoff tool."""
    if isinstance(tool_definition, HandoffTool):
        return _resolved_child_descriptor(
            runner_task=runner_task,
            descriptor=tool_definition.descriptor,
        )

    config = tool_definition.config
    return AgentDescriptor(
        name="Delegated Agent",
        description="Fresh delegated handoff agent.",
        model=config.model or _require_model(runner_task),
        instructions=config.instructions,
        model_args=(
            dict(config.model_args)
            if config.model_args is not None
            else _model_args(runner_task)
        ),
        schema=config.schema,
        tools=config.tools,
    )


def _resolved_child_descriptor(
    *,
    runner_task: RunnerTask | None,
    descriptor: AgentDescriptor,
) -> AgentDescriptor:
    """Fill child descriptor model defaults from the current agent."""
    model = descriptor.model or _require_model(runner_task)
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


def _base_tools(runner_task: RunnerTask | None) -> Tools | None:
    """Return the explicit parent tool catalog before handoff merge."""
    if runner_task is None:
        return None
    return runner_task.base_tools


def _overwrite_run_state(target: RunState, source: RunState) -> None:
    """Replace one working run state in place with another completed state."""
    target.instructions = copy_instructions(source.instructions)
    target.history = list(source.history)
    target.responses = list(source.responses)
    target.shim_state = copy_shim_state(source.shim_state)
    target.turn_count = source.turn_count


def _parse_delegated_tool_args(
    *,
    tool_call: ToolCall,
    args_model: type[BaseModel],
) -> BaseModel:
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


def _parse_delegated_task_input(
    *,
    tool_call: ToolCall,
    args_model: type[DelegatedTaskInput],
) -> DelegatedTaskInput:
    """Validate one task-based delegation tool call."""
    parsed_input = _parse_delegated_tool_args(
        tool_call=tool_call,
        args_model=args_model,
    )
    if not isinstance(parsed_input, DelegatedTaskInput):
        raise AssertionError("Delegated task parsing returned the wrong model type.")
    return parsed_input


def _parse_default_agent_tool_input(
    *,
    tool_call: ToolCall,
    args_model: type[DefaultAgentToolInput],
) -> DefaultAgentToolInput:
    """Validate one generic spawned-agent tool call.

    This validates the model's generic spawned-agent arguments before the
    runner turns the task into the child agent's input.
    """
    parsed_input = _parse_delegated_tool_args(
        tool_call=tool_call,
        args_model=args_model,
    )
    if not isinstance(parsed_input, DefaultAgentToolInput):
        raise AssertionError(
            "Default agent tool parsing returned the wrong model type."
        )
    return parsed_input


def _agent_tool_run_input(parsed_input: BaseModel) -> list[RunHistoryItem]:
    """Build the downstream run input for a predefined agent-as-tool call.

    Agent-as-tool always forwards the validated structured payload that the
    model already produced for the tool schema. This keeps predefined and
    generic spawned helper agents aligned with normal tool-call mechanics.
    """
    return [parsed_input]


def _default_agent_tool_run_input(
    parsed_input: DefaultAgentToolInput,
) -> list[RunHistoryItem]:
    """Build the downstream run input for a generic spawned-agent call."""
    return [parsed_input.task]


def _default_agent_child_shims(tool_definition: DefaultAgentTool) -> tuple[Shim, ...]:
    """Return shim policy for one generic spawned helper."""
    if not isinstance(tool_definition.tools, (InheritTools, RestrictTools)):
        return ()

    return (HarnessToolsShim(base_harness_tools()),)


def _tool_result_message(
    *,
    tool_call: ToolCall,
    tool_name: str,
    content: object,
) -> MessageDict:
    """Return the canonical tool-result message for one completed call."""
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_name,
        "content": _normalize_content(content),
    }


def _require_runtime_engine(agent: Task) -> RuntimeEngine:
    """Return the runtime engine required for delegated agent routing."""
    engine = getattr(agent, "_engine", None)
    if isinstance(engine, RuntimeEngine):
        return engine
    raise RuntimeError(
        "Harness sub-agent delegation requires an engine that supports "
        "runtime factory registration."
    )


def _build_request(turn: PreparedTurn) -> list[MessageDict]:
    """Build the full model request from agent config and run state."""
    messages: list[MessageDict] = []

    messages.extend(_instruction_messages(turn.run_state.instructions))
    for item in turn.run_state.history:
        messages.extend(_history_item_to_messages(item))

    return messages


def _instruction_messages(
    instructions: str | PromptSpec[Any] | None,
) -> list[MessageDict]:
    """Render system instructions into canonical model messages."""
    if instructions is None:
        return []
    if isinstance(instructions, PromptSpec):
        return _prompt_messages("system", instructions)
    return [_normalize_message({"role": "system", "content": instructions})]


def _history_item_to_messages(item: RunHistoryItem) -> list[MessageDict]:
    """Render one persisted history item into canonical model messages.

    ``RunHistoryItem`` names the supported shapes explicitly. ``ModelResponse``
    objects are assistant turns, canonical message dicts pass through
    unchanged, and the remaining item kinds are treated as user-side input.
    """
    if isinstance(item, ModelResponse):
        return [_assistant_message_from_response(item)]
    message = _as_message_dict(item)
    if message is not None:
        return [_normalize_message(message)]
    return _user_item_to_messages(item)


def _user_item_to_messages(item: RunHistoryItem) -> list[MessageDict]:
    """Render one user-side continuation item into canonical model messages."""
    if isinstance(item, PromptSpec):
        return _prompt_messages("user", item)
    return [_normalize_message({"role": "user", "content": item})]


def _prompt_messages(
    role: Literal["system", "user"],
    prompt_spec: PromptSpec[Any],
) -> list[MessageDict]:
    """Render a ``PromptSpec`` and keep only messages matching ``role``."""
    messages = [
        _normalize_message(message)
        for message in prompt_spec.template.render_messages(prompt_spec.values)
        if message.get("role") == role
    ]
    if messages:
        return messages
    raise ValueError(f"PromptSpec must render at least one {role}-role message.")


def _normalize_message(message: Mapping[str, object]) -> MessageDict:
    """Copy one message dict and normalize its ``content`` field."""
    normalized_message = dict(message)
    if "content" in normalized_message:
        normalized_message["content"] = _normalize_content(
            normalized_message["content"]
        )
    return normalized_message


def _normalize_content(content: object) -> object:
    """Normalize arbitrary content into a model-ready value.

    Fast-path checks are ordered by expected frequency:
      1. ``str`` — most common, returned as-is.
      2. ``list`` — multi-part content arrays, passed through.
      3. ``BaseModel`` — Pydantic models serialized to JSON.
      4. dataclass — stdlib dataclasses serialized to JSON.
      5. Anything else — attempt ``json.dumps``, fall back to ``str()``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return cast(list[object], content)
    if isinstance(content, BaseModel):
        return content.model_dump_json()
    if is_dataclass(content) and not isinstance(content, type):
        return json.dumps(asdict(content))
    try:
        return json.dumps(content)
    except TypeError:
        return str(content)


def _as_message_dict(item: object) -> dict[str, object] | None:
    """Return the item when it already looks like one canonical message dict."""
    if not isinstance(item, dict):
        return None

    message = cast(dict[str, object], item)
    role = message.get("role")
    if isinstance(role, str):
        return message
    return None


def _narrow_runner_task(agent: Task) -> RunnerTask | None:
    """Narrow a ``Task`` to ``RunnerTask`` once per run."""
    if isinstance(agent, RunnerTask):
        return agent
    return None


def _shim_manager(agent: Task) -> BoundShimManager | None:
    """Return the bound shim manager exposed by the task, if any."""
    manager = getattr(agent, "bound_shim_manager", None)
    if not isinstance(manager, BoundShimManager):
        return None
    return manager


def _require_model(runner_task: RunnerTask | None) -> Model[ModelResponse]:
    """Return the agent's model client, or raise if unconfigured."""
    if runner_task is not None and runner_task.model is not None:
        return runner_task.model
    raise RuntimeError(
        "Runner requires the task to expose a configured 'model' client."
    )


def _model_args(runner_task: RunnerTask | None) -> dict[str, object] | None:
    """Return a defensive copy of the agent's model-call arguments."""
    if runner_task is None:
        return None
    model_args = runner_task.model_args
    if model_args is None:
        return None
    # Copy so per-call mutations (e.g. provider-side updates) never leak
    # back into the agent descriptor.
    return dict(model_args)


def _schema(
    runner_task: RunnerTask | None,
) -> type[BaseModel] | OutputSchema[Any] | None:
    """Return the structured-output schema exposed by the agent, if any."""
    return runner_task.schema if runner_task else None


def _tools(runner_task: RunnerTask | None) -> Tools | None:
    """Return the tool configuration exposed by the agent, if any."""
    return runner_task.tools if runner_task else None


def _visible_tools(
    runner_task: RunnerTask | None,
    tool_call_counts: dict[str, int],
    tool_round_trips: int,
) -> Tools | None:
    """Return the effective tools visible for the next model turn."""
    tools = _tools(runner_task)
    if tools is None:
        return None
    return _limit_tools(tools, tool_call_counts, tool_round_trips)


def _limit_tools(
    tools: Tools,
    tool_call_counts: dict[str, int],
    tool_round_trips: int,
) -> Tools | None:
    """Apply tool visibility limits using incremental counters."""
    if tools.tool_choice == "none":
        return tools

    if tool_round_trips >= tools.max_tool_round_trips:
        return None

    if not tools.tool_call_limits:
        return tools

    active_tools = tuple(
        tool
        for tool in tools.normalized_tools
        if tool.name not in tools.tool_call_limits
        or tool_call_counts.get(tool.name, 0) < tools.tool_call_limits[tool.name]
    )
    if not active_tools:
        return None
    if len(active_tools) == len(tools.normalized_tools):
        return tools

    # Strip exhausted tools from later requests.
    return replace(tools, tools=active_tools)


def _check_turn_limit(turn_count: int, max_turns: int) -> None:
    """Fail fast when the run exceeds the configured turn limit."""
    if turn_count > max_turns:
        raise RuntimeError(f"Runner exceeded the configured turn limit of {max_turns}.")


def _validate_terminal_response(response: ModelResponse) -> None:
    """Validate the response is a direct assistant answer.

    Tool-call responses are handled earlier in the loop. By the time this
    validator runs, the response must represent a terminal assistant answer.
    """
    _validate_response_choice(response)

    message = response.choices[0].message

    if message.role != "assistant":
        raise _model_behavior_error(
            "Runner expected the first model choice to be an assistant message.",
            raw_response=response,
        )


def _validate_response_choice(response: ModelResponse) -> None:
    """Ensure the response contains at least one choice."""
    if response.choices:
        return

    raise _model_behavior_error(
        "Runner expected the model response to contain at least one choice.",
        raw_response=response,
    )


def _assistant_message_from_response(response: ModelResponse) -> MessageDict:
    """Extract the first assistant message for continuation history."""
    _validate_response_choice(response)
    return response.choices[0].message.model_dump(mode="json", exclude_none=True)


def _extract_tool_calls(response: ModelResponse) -> list[ToolCall]:
    """Extract any tool calls from the first model choice."""
    _validate_response_choice(response)
    return cast(list[ToolCall], list(response.choices[0].message.tool_calls or []))


def _extract_direct_answer(response: ModelResponse) -> object:
    """Extract the text content from a validated terminal response."""
    _validate_response_choice(response)
    return response.choices[0].message.content


def _model_behavior_error(
    message: str,
    *,
    raw_response: ModelResponse | None,
) -> ModelBehaviorError:
    """Create a ``ModelBehaviorError`` annotated with raw response details."""
    error = ModelBehaviorError(message)
    error.run_data = RunErrorDetails(raw_response=raw_response)
    return error
