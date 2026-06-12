import asyncio
from pathlib import Path
from typing import Any

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    RunEvent,
    RunEventKind,
    RunHandoffEndEvent,
    RunHandoffStartEvent,
    RunLLMEndEvent,
    RunLLMStartEvent,
    RunModelStreamEvent,
    Runner,
    RunnerHooks,
    RunResult,
    RunState,
    RunStateSnapshotBoundary,
    RunStateSnapshotEvent,
    RunToolApprovalEvent,
    RunToolEndEvent,
    RunToolStartEvent,
    Task,
)
from agentlane.harness._handoff import normalize_delegation_tool_name
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.shims import PreparedTurn, Shim
from agentlane.harness.tools import (
    HarnessToolsShim,
    ToolApprovalBroker,
    ToolApprovalStatus,
    ToolPermissionDecision,
    ToolPermissionRequest,
    agent_tool,
    read_tool,
)
from agentlane.messaging import AgentId
from agentlane.models import (
    MessageDict,
    Model,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    ToolCall,
    Tools,
    get_content_or_none,
)
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine
from agentlane.tracing import Span

from .tools_test_utils import SequenceModel, echo_tool


def _make_assistant_response(
    content: str | None,
    *,
    tool_calls: list[ToolCall] | None = None,
) -> ModelResponse:
    """Build one canonical assistant response for run-event tests."""
    return ModelResponse.model_validate(
        {
            "id": "chatcmpl_run_events",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
                }
            ],
        }
    )


def _make_tool_call(
    *,
    tool_id: str,
    arguments: str,
    name: str = "echo",
) -> ToolCall:
    """Build one canonical tool call payload."""
    return ToolCall.model_validate(
        {
            "id": tool_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }
    )


class _StreamingSequenceModel(Model[ModelResponse]):
    def __init__(self, outcomes: list[ModelResponse]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[list[MessageDict]] = []

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: object | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        del messages, extra_call_args, schema, tools, cancellation_token, kwargs
        raise AssertionError("Run-event tests should use streamed model calls.")

    def stream_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: object | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
        **kwargs: object,
    ):
        del extra_call_args, schema, tools, cancellation_token, kwargs

        async def _stream():
            self.calls.append([dict(message) for message in messages])
            if not self._outcomes:
                raise AssertionError("Expected one queued model response.")

            response = self._outcomes.pop(0)
            content = get_content_or_none(response) or ""
            if content:
                yield ModelStreamEvent(
                    kind=ModelStreamEventKind.TEXT_DELTA,
                    text=content,
                )
            yield ModelStreamEvent(
                kind=ModelStreamEventKind.COMPLETED,
                response=response,
            )

        return _stream()


class _PreparedTurnShim(Shim):
    @property
    def name(self) -> str:
        return "prepared-turn"

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        turn.run_state.shim_state["prepared-turn"] = turn.run_state.turn_count


class _RecordingHooks(RunnerHooks):
    def __init__(self) -> None:
        self.events: list[str] = []

    async def on_agent_start(
        self,
        task: Task,
        state: RunState,
    ) -> None:
        del task, state
        self.events.append("agent_start")

    async def on_agent_end(
        self,
        task: Task,
        result: RunResult | None,
    ) -> None:
        del task, result
        self.events.append("agent_end")

    async def on_llm_start(
        self,
        task: Task,
        messages: list[MessageDict],
    ) -> None:
        del task, messages
        self.events.append("llm_start")

    async def on_llm_end(
        self,
        task: Task,
        response: ModelResponse,
    ) -> None:
        del task, response
        self.events.append("llm_end")

    async def on_tool_call_start(
        self,
        task: Task,
        tool_call: ToolCall,
    ) -> None:
        del task, tool_call
        self.events.append("tool_start")

    async def on_tool_call_end(
        self,
        task: Task,
        tool_call: ToolCall,
        result: object,
    ) -> None:
        del task, tool_call, result
        self.events.append("tool_end")


async def _collect_run_events(stream: Any) -> list[RunEvent]:
    events: list[RunEvent] = []
    async for event in stream:
        events.append(event)
    return events


async def _wait_until_no_pending_approvals(
    broker: ToolApprovalBroker,
) -> None:
    for _ in range(100):
        if broker.pending() == ():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Expected pending approvals to be discarded.")


def _state_snapshots(events: list[RunEvent]) -> list[RunStateSnapshotEvent]:
    return [event for event in events if isinstance(event, RunStateSnapshotEvent)]


def _model_stream_events(events: list[RunEvent]) -> list[RunModelStreamEvent]:
    return [event for event in events if isinstance(event, RunModelStreamEvent)]


def test_default_agent_run_events_orders_model_tool_and_state_events() -> None:
    async def scenario() -> None:
        executed: list[str] = []

        async def echo(text: str, cancellation_token: CancellationToken) -> str:
            """Echo text."""
            del cancellation_token
            executed.append(text)
            return f"tool:{text}"

        model = _StreamingSequenceModel(
            [
                _make_assistant_response(
                    content=None,
                    tool_calls=[
                        _make_tool_call(
                            tool_id="call_1",
                            arguments='{"text":"docs"}',
                        )
                    ],
                ),
                _make_assistant_response(content="docs are ready"),
            ]
        )
        hooks = _RecordingHooks()
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="ToolEvents",
                model=model,
                tools=Tools(tools=[echo]),
                shims=(_PreparedTurnShim(),),
            ),
            hooks=hooks,
        )

        stream = await agent.run_events("search docs")
        events = await _collect_run_events(stream)
        result = await stream.result()

        assert executed == ["docs"]
        assert result.final_output == "docs are ready"
        assert agent.run_state is not None
        assert agent.run_state.turn_count == 2
        assert hooks.events == [
            "agent_start",
            "llm_start",
            "llm_end",
            "tool_start",
            "tool_end",
            "llm_start",
            "llm_end",
            "agent_end",
        ]
        assert [event.kind for event in events] == [
            RunEventKind.AGENT_START,
            RunEventKind.STATE_SNAPSHOT,
            RunEventKind.STATE_SNAPSHOT,
            RunEventKind.LLM_START,
            RunEventKind.MODEL_STREAM,
            RunEventKind.LLM_END,
            RunEventKind.TOOL_START,
            RunEventKind.TOOL_END,
            RunEventKind.STATE_SNAPSHOT,
            RunEventKind.STATE_SNAPSHOT,
            RunEventKind.LLM_START,
            RunEventKind.MODEL_STREAM,
            RunEventKind.MODEL_STREAM,
            RunEventKind.LLM_END,
            RunEventKind.STATE_SNAPSHOT,
            RunEventKind.AGENT_END,
        ]

        llm_start_events = [
            event for event in events if isinstance(event, RunLLMStartEvent)
        ]
        llm_end_events = [
            event for event in events if isinstance(event, RunLLMEndEvent)
        ]
        tool_start_events = [
            event for event in events if isinstance(event, RunToolStartEvent)
        ]
        tool_end_events = [
            event for event in events if isinstance(event, RunToolEndEvent)
        ]
        assert len(llm_start_events) == 2
        assert len(llm_end_events) == 2
        assert llm_start_events[0].messages == [
            {"role": "user", "content": "search docs"}
        ]
        assert tool_start_events[0].tool_call.function.name == "echo"
        assert tool_end_events[0].result == "tool:docs"

        model_events = _model_stream_events(events)
        assert [event.event.kind for event in model_events] == [
            ModelStreamEventKind.COMPLETED,
            ModelStreamEventKind.TEXT_DELTA,
            ModelStreamEventKind.COMPLETED,
        ]
        assert model_events[1].event.text == "docs are ready"

        snapshots = _state_snapshots(events)
        assert [event.boundary for event in snapshots] == [
            RunStateSnapshotBoundary.RUN_START,
            RunStateSnapshotBoundary.TURN_PREPARED,
            RunStateSnapshotBoundary.TOOL_ROUND_END,
            RunStateSnapshotBoundary.TURN_PREPARED,
            RunStateSnapshotBoundary.RUN_END,
        ]
        assert snapshots[0].snapshot.turn_count == 0
        assert snapshots[0].snapshot.history_length == 1
        assert snapshots[0].snapshot.response_count == 0
        assert snapshots[0].snapshot.shim_state == {}
        assert snapshots[1].snapshot.shim_state == {"prepared-turn": 1}
        assert snapshots[2].snapshot.turn_count == 1
        assert snapshots[2].snapshot.history_length == 3
        assert snapshots[2].snapshot.response_count == 1
        assert snapshots[3].snapshot.shim_state == {"prepared-turn": 2}
        assert snapshots[4].snapshot.turn_count == 2
        assert snapshots[4].snapshot.history_length == 4
        assert snapshots[4].snapshot.response_count == 2

    asyncio.run(scenario())


_LINEAGE_EVENT_TYPES = (
    RunLLMStartEvent,
    RunLLMEndEvent,
    RunToolStartEvent,
    RunToolEndEvent,
)


def test_default_agent_run_events_stamp_root_lineage_on_single_agent_run() -> None:
    async def scenario() -> None:
        async def echo(text: str, cancellation_token: CancellationToken) -> str:
            """Echo text."""
            del cancellation_token
            return f"tool:{text}"

        model = _StreamingSequenceModel(
            [
                _make_assistant_response(
                    content=None,
                    tool_calls=[
                        _make_tool_call(tool_id="call_1", arguments='{"text":"x"}')
                    ],
                ),
                _make_assistant_response(content="done"),
            ]
        )
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Lineage",
                model=model,
                tools=Tools(tools=[echo]),
            )
        )

        stream = await agent.run_events("go")
        events = await _collect_run_events(stream)
        await stream.result()

        lineage_events = [
            event for event in events if isinstance(event, _LINEAGE_EVENT_TYPES)
        ]
        assert lineage_events, "expected task-carrying run events"
        # Single-agent runs are entirely root: no app-side root latching needed.
        assert all(event.is_root for event in lineage_events)
        assert all(event.parent_task_id is None for event in lineage_events)
        tool_events = [
            event
            for event in events
            if isinstance(event, (RunToolStartEvent, RunToolEndEvent))
        ]
        assert tool_events and all(
            event.is_delegation is False for event in tool_events
        )

    asyncio.run(scenario())


def test_default_agent_run_events_tag_agent_tool_calls_as_delegation() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        # The delegated child runs as a subroutine through get_response, so a
        # non-streaming SequenceModel is the correct child model here.
        child_model = SequenceModel([_make_assistant_response(content="child answer")])
        parent_model = _StreamingSequenceModel(
            [
                _make_assistant_response(
                    content=None,
                    tool_calls=[
                        _make_tool_call(
                            tool_id="call_echo",
                            name="echo",
                            arguments='{"text":"hi"}',
                        )
                    ],
                ),
                _make_assistant_response(
                    content=None,
                    tool_calls=[
                        _make_tool_call(
                            tool_id="call_agent",
                            name="agent",
                            arguments=(
                                '{"name":"Researcher","task":"Look into the case."}'
                            ),
                        )
                    ],
                ),
                _make_assistant_response(content="all done"),
            ]
        )
        agent = DefaultAgent(
            runtime=runtime,
            runner=runner,
            descriptor=AgentDescriptor(
                name="Manager",
                model=parent_model,
                tools=Tools(tools=[echo_tool("echo")]),
                shims=(HarnessToolsShim((agent_tool(model=child_model),)),),
            ),
        )

        stream = await agent.run_events("delegate then finish")
        events = await _collect_run_events(stream)
        result = await stream.result()

        assert result.final_output == "all done"
        tool_starts = {
            event.tool_call.function.name: event
            for event in events
            if isinstance(event, RunToolStartEvent)
        }
        tool_ends = {
            event.tool_call.function.name: event
            for event in events
            if isinstance(event, RunToolEndEvent)
        }
        # Delegation is tagged structurally by runner tool-definition type, so the
        # agent-as-tool call is identifiable without a tool-name registry, while
        # the plain echo tool is not delegation.
        assert tool_starts["agent"].is_delegation is True
        assert tool_ends["agent"].is_delegation is True
        assert tool_starts["echo"].is_delegation is False
        assert tool_ends["echo"].is_delegation is False
        # Every event in the parent stream is root: the delegated child runs as a
        # subroutine and surfaces only as the tagged tool call.
        assert all(
            event.is_root for event in events if isinstance(event, _LINEAGE_EVENT_TYPES)
        )
        # Snapshot-root-only contract: the child emits no state snapshots into
        # the parent stream, so the parent's snapshot turn_counts stay monotonic.
        snapshot_turns = [
            event.snapshot.turn_count for event in _state_snapshots(events)
        ]
        assert snapshot_turns == sorted(snapshot_turns)

    asyncio.run(scenario())


def test_runner_run_events_emits_handoff_events() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        child_descriptor = AgentDescriptor(
            name="Returns Specialist",
            model=_StreamingSequenceModel(
                [_make_assistant_response(content="handled by returns")]
            ),
            instructions="You handle returns.",
            tools=None,
        )
        parent_model = _StreamingSequenceModel(
            [
                _make_assistant_response(
                    content=None,
                    tool_calls=[
                        _make_tool_call(
                            tool_id="handoff_1",
                            name=normalize_delegation_tool_name(child_descriptor.name),
                            arguments='{"task":"Take over this return question."}',
                        )
                    ],
                )
            ]
        )
        agent = Agent.bind(
            runtime,
            AgentId.from_values("assistant-agent", "frontline-events"),
            runner,
            descriptor=AgentDescriptor(
                name="Frontline",
                model=parent_model,
                handoffs=(child_descriptor,),
            ),
        )
        state = RunState(
            instructions=None,
            history=["Can I return this order?"],
            responses=[],
        )

        stream = runner.run_events(agent, state)
        events = await _collect_run_events(stream)
        result = await stream.result()

        handoff_start = next(
            event for event in events if isinstance(event, RunHandoffStartEvent)
        )
        handoff_end = next(
            event for event in events if isinstance(event, RunHandoffEndEvent)
        )
        assert result.final_output == "handled by returns"
        assert handoff_start.target_name == "Returns Specialist"
        assert handoff_start.tool_call.id == "handoff_1"
        assert handoff_end.target_name == "Returns Specialist"
        assert handoff_end.result.final_output == "handled by returns"
        assert events.index(handoff_start) < events.index(handoff_end)

    asyncio.run(scenario())


def test_default_agent_run_events_can_merge_tool_approval_events(
    tmp_path: Path,
) -> None:
    class RequireApprovalPolicy:
        def check(
            self,
            request: ToolPermissionRequest,
        ) -> ToolPermissionDecision:
            del request
            return ToolPermissionDecision.require_approval()

    async def scenario() -> None:
        target = tmp_path / "notes.txt"
        target.write_text("approved notes", encoding="utf-8")
        broker = ToolApprovalBroker()
        read_definition = read_tool(
            cwd=tmp_path,
            permissions=RequireApprovalPolicy(),
            approval_callback=broker.callback,
        )
        model = _StreamingSequenceModel(
            [
                _make_assistant_response(
                    content=None,
                    tool_calls=[
                        _make_tool_call(
                            tool_id="approval_call_1",
                            name="read",
                            arguments='{"path":"notes.txt"}',
                        )
                    ],
                ),
                _make_assistant_response(content="read complete"),
            ]
        )
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="ApprovalEvents",
                model=model,
                tools=Tools(tools=[read_definition.tool]),
            )
        )

        stream = await agent.run_events(
            "read the note",
            approval_events=broker.events(),
        )
        events: list[RunEvent] = []
        async for event in stream:
            events.append(event)
            if (
                isinstance(event, RunToolApprovalEvent)
                and event.event.status == ToolApprovalStatus.PENDING
            ):
                assert await broker.resolve(
                    event.event.request_id,
                    ToolPermissionDecision.allow(),
                )

        result = await stream.result()
        approval_events = [
            event for event in events if isinstance(event, RunToolApprovalEvent)
        ]

        assert result.final_output == "read complete"
        assert [event.event.status for event in approval_events] == [
            ToolApprovalStatus.PENDING,
            ToolApprovalStatus.RESOLVED,
        ]
        assert approval_events[0].event.record.request.tool_name == "read"

    asyncio.run(scenario())


def test_default_agent_run_events_close_cancels_without_committing_state_or_approval(
    tmp_path: Path,
) -> None:
    class RequireApprovalPolicy:
        def check(
            self,
            request: ToolPermissionRequest,
        ) -> ToolPermissionDecision:
            del request
            return ToolPermissionDecision.require_approval()

    async def scenario() -> None:
        target = tmp_path / "notes.txt"
        target.write_text("approved notes", encoding="utf-8")
        broker = ToolApprovalBroker()
        read_definition = read_tool(
            cwd=tmp_path,
            permissions=RequireApprovalPolicy(),
            approval_callback=broker.callback,
        )
        model = _StreamingSequenceModel(
            [
                _make_assistant_response(
                    content=None,
                    tool_calls=[
                        _make_tool_call(
                            tool_id="approval_cancel_1",
                            name="read",
                            arguments='{"path":"notes.txt"}',
                        )
                    ],
                )
            ]
        )
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="ApprovalCancelEvents",
                model=model,
                tools=Tools(tools=[read_definition.tool]),
            )
        )

        stream = await agent.run_events(
            "read the note",
            approval_events=broker.events(),
        )
        async for event in stream:
            if (
                isinstance(event, RunToolApprovalEvent)
                and event.event.status == ToolApprovalStatus.PENDING
            ):
                assert len(broker.pending()) == 1
                await stream.aclose()
                break

        try:
            await stream.result()
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Expected cancelled run-events result.")

        await _wait_until_no_pending_approvals(broker)
        assert agent.run_state is None

    asyncio.run(scenario())
