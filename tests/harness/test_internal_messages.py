"""Internal context stays out of public records; ephemeral context ends with its run."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    AgentSnapshot,
    RunAgentEndEvent,
    RunLLMStartEvent,
    Runner,
    RunResult,
    RunState,
    RunStateSnapshotBoundary,
    RunStateSnapshotEvent,
)
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.compaction import render_request_messages
from agentlane.harness.shims import Shim
from agentlane.harness.tools import HarnessToolsShim, agent_tool
from agentlane.messaging import AgentId
from agentlane.models import (
    MessageDict,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    Tools,
    internal_message,
    is_ephemeral_message,
    is_internal_message,
    provider_messages,
)
from agentlane.models.run import RunContext
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine
from agentlane.tracing import Span

from .tools_test_utils import (
    SequenceModel,
    StreamingSequenceModel,
    echo_tool,
    make_assistant_response,
    make_tool_call,
)

_VIEW_NOTE = "VIEW NOTE: patient X, summary.md"
"""Ephemeral context text that must reach the model but never durable state."""


def test_internal_metadata_survives_copies_and_json_restore() -> None:
    ordinary = {"role": "user", "content": "Find my report", "name": "brain_artifacts"}
    context = internal_message(
        {"role": "user", "content": "Report inventory"}, origin="inventory"
    )
    state = RunState(instructions="Help", history=[ordinary, context], responses=[])
    before = render_request_messages(state.instructions, state.history)

    # Repeated saves must keep the same provider prefix and runtime identity.
    for _ in range(3):
        agent = DefaultAgent(
            descriptor=AgentDescriptor(name="Assistant"), run_state=state
        )
        copied = agent.run_state
        assert copied is not None
        assert copied is not state
        state = copied
        snapshot = AgentSnapshot.capture(
            agent_id=AgentId.from_values("assistant", "main"), run_state=state
        )
        state = AgentSnapshot.from_json(
            json.loads(snapshot.model_dump_json())
        ).to_run_state()
        assert is_internal_message(state.history[-1], origin="inventory")
        assert not is_internal_message(state.history[-1], origin="other")
        assert not is_internal_message(state.history[0])
        assert render_request_messages(state.instructions, state.history) == before
        assert provider_messages(before) == [
            {"role": "system", "content": "Help"},
            ordinary,
            {"role": "user", "content": "Report inventory"},
        ]


def test_default_serialization_omits_internal_messages_without_client_filtering() -> (
    None
):
    ordinary = {"role": "user", "content": "Find my report"}
    context = internal_message(
        {"role": "user", "content": "PRIVATE INVENTORY"}, origin="inventory"
    )
    state = RunState(instructions=None, history=[ordinary, context], responses=[])
    result = RunResult(final_output="Done", responses=[], turn_count=1, run_state=state)
    events = [
        RunLLMStartEvent(task_name="test", task_id="1", messages=[ordinary, context]),
        RunAgentEndEvent(task_name="test", task_id="1", result=result),
    ]

    for event in events:
        record = event.to_dict()
        assert "PRIVATE INVENTORY" not in json.dumps(record)
        assert "Find my report" in json.dumps(record)
        assert json.loads(event.dumps()) == record

    assert "PRIVATE INVENTORY" not in json.dumps(result.to_dict())
    assert state.history == [ordinary, context]

    snapshot = AgentSnapshot.capture(
        agent_id=AgentId.from_values("assistant", "main"), run_state=state
    )
    assert "PRIVATE INVENTORY" in snapshot.model_dump_json()


def test_old_snapshot_and_ordinary_messages_remain_public() -> None:
    state = RunState(instructions=None, history=["hello"], responses=[])
    snapshot = AgentSnapshot.capture(
        agent_id=AgentId.from_values("assistant", "main"), run_state=state
    )
    restored = AgentSnapshot.from_json(
        json.loads(snapshot.model_dump_json())
    ).to_run_state()
    messages = render_request_messages(restored.instructions, restored.history)

    event = RunLLMStartEvent(task_name="test", task_id="1", messages=messages)
    assert event.to_dict()["payload"]["messages"] == messages
    assert provider_messages(messages) == messages


def test_default_result_serialization_filters_nested_message_containers() -> None:
    context = internal_message(
        {"role": "user", "content": "private context"}, origin="test"
    )
    ordinary = {"role": "user", "content": "visible", "name": "brain_artifacts"}
    output = {"context": context, "nested": {"messages": [context, ordinary]}}
    result = RunResult(final_output=output, responses=[], turn_count=1)

    assert result.to_dict()["final_output"] == {
        "context": None,
        "nested": {"messages": [ordinary]},
    }
    assert result.final_output == output
    assert "context" in output


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": "PRIVATE"},
        {"type": "function_call_output", "call_id": "call-1", "output": "PRIVATE"},
    ],
)
def test_marked_message_shapes_are_hidden_by_default(message: MessageDict) -> None:
    context = internal_message(message, origin="test")
    assert context is not message
    assert "_agentlane" not in message
    assert is_internal_message(context)
    assert is_internal_message(context, origin="test")
    assert not is_internal_message(context, origin="other")
    assert provider_messages([context]) == [message]

    # Required output fields survive filtering, while message lists omit the item.
    result = RunResult(final_output=context, responses=[], turn_count=1)
    assert result.to_dict() == {
        "final_output": None,
        "responses": [],
        "turn_count": 1,
        "run_state": None,
    }
    event = RunLLMStartEvent(task_name="test", task_id="1", messages=[context])
    assert event.to_dict()["payload"]["messages"] == []
    assert "PRIVATE" not in event.dumps()
    assert result.final_output is context


@pytest.mark.parametrize(
    "metadata", [None, "internal", [], {}, {"visibility": "public"}]
)
def test_unmarked_metadata_stays_public(metadata: object) -> None:
    message = {"role": "user", "content": "visible", "_agentlane": metadata}
    assert not is_internal_message(message)
    result = RunResult(final_output=message, responses=[], turn_count=1)
    assert result.to_dict()["final_output"] == message


def _view_note(text: str = _VIEW_NOTE) -> MessageDict:
    return internal_message(
        {"role": "user", "content": text}, origin="ui_view", ephemeral=True
    )


def _contains(value: object, text: str) -> bool:
    return text in json.dumps(value, default=str)


def _tool_round() -> ModelResponse:
    return make_assistant_response(
        None,
        tool_calls=[
            make_tool_call(tool_id="call_1", name="echo", arguments='{"text":"ok"}')
        ],
    )


def _agent(model: SequenceModel, **kwargs: Any) -> DefaultAgent:
    return DefaultAgent(
        descriptor=AgentDescriptor(
            name="Assistant", model=model, tools=Tools(tools=[echo_tool("echo")])
        ),
        **kwargs,
    )


class _FailingModel(SequenceModel):
    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: Tools | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        del extra_call_args, schema, tools, cancellation_token, kwargs
        if not self._responses:
            self.calls.append([dict(message) for message in messages])
            raise RuntimeError("model failed")

        return await super().get_response(messages)


class _BlockingStreamModel(SequenceModel):
    """Answer terminal calls from the queue and block every stream until cancel."""

    async def stream_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: Tools | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
        **kwargs: object,
    ) -> AsyncIterator[ModelStreamEvent]:
        del extra_call_args, schema, tools, parent_span, kwargs
        self.calls.append([dict(message) for message in messages])
        assert cancellation_token is not None
        await cancellation_token.wait_cancelled()
        raise asyncio.CancelledError()
        yield  # pragma: no cover


class _SteeringModel(StreamingSequenceModel):
    """Deliver queued steering input with the first completed response."""

    def __init__(
        self, responses: list[ModelResponse], steering: list[MessageDict]
    ) -> None:
        super().__init__(responses)
        self._steering = steering

    async def stream_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: Tools | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
        **kwargs: object,
    ) -> AsyncIterator[ModelStreamEvent]:
        async for event in super().stream_response(
            messages,
            extra_call_args=extra_call_args,
            schema=schema,
            tools=tools,
            cancellation_token=cancellation_token,
            parent_span=parent_span,
            **kwargs,
        ):
            if event.kind == ModelStreamEventKind.COMPLETED and self._steering:
                event.input_messages = self._steering
                self._steering = []
            yield event


class _ViewContextShim(Shim):
    @property
    def name(self) -> str:
        return "ui-view"

    async def on_run_start(
        self, state: RunState, transient_state: RunContext[Any]
    ) -> None:
        del transient_state
        state.history.append(_view_note())


def test_ephemeral_marker_copies_internal_message_with_lifetime_flag() -> None:
    message = {"role": "user", "content": "view"}

    note = internal_message(message, origin="ui_view", ephemeral=True)
    persistent = internal_message(message, origin="ui_view")

    assert note is not message
    assert "_agentlane" not in message
    assert note["_agentlane"] == {
        "visibility": "internal",
        "origin": "ui_view",
        "ephemeral": True,
    }
    assert is_internal_message(note, origin="ui_view")
    assert is_ephemeral_message(note)
    assert is_ephemeral_message(note, origin="ui_view")
    assert not is_ephemeral_message(note, origin="other")
    assert provider_messages([note]) == [message]

    # Persistent context keeps the exact metadata that existing snapshots hold.
    assert persistent["_agentlane"] == {"visibility": "internal", "origin": "ui_view"}
    assert internal_message(message, origin="ui_view", ephemeral=False) == persistent
    assert not is_ephemeral_message(persistent)


@pytest.mark.parametrize(
    "metadata",
    [
        {"visibility": "public", "origin": "ui_view", "ephemeral": True},
        {"visibility": "internal", "origin": "ui_view", "ephemeral": "yes"},
        {"visibility": "internal", "origin": "ui_view", "ephemeral": 1},
        {"origin": "ui_view", "ephemeral": True},
    ],
)
def test_malformed_ephemeral_metadata_is_not_ephemeral(
    metadata: dict[str, object],
) -> None:
    message = {"role": "user", "content": "view", "_agentlane": metadata}

    assert not is_ephemeral_message(message)


@pytest.mark.asyncio
async def test_ephemeral_context_reaches_each_call_of_its_run_only() -> None:
    model = SequenceModel(
        [_tool_round(), make_assistant_response("done"), make_assistant_response("2")]
    )
    agent = _agent(model)
    inventory = internal_message(
        {"role": "user", "content": "INVENTORY"}, origin="inventory"
    )

    result = await agent.run(["question", _view_note(), inventory])

    # Every model call of the run receives the note right after the user input.
    expected_prefix = [
        {"role": "user", "content": "question"},
        {"role": "user", "content": _VIEW_NOTE},
        {"role": "user", "content": "INVENTORY"},
    ]
    assert len(model.calls) == 2
    assert all(call[:3] == expected_prefix for call in model.calls)

    # Committed state and snapshots keep only persistent context.
    snapshot = agent.snapshot()
    assert result.run_state is not None
    assert agent.run_state is not None
    assert snapshot is not None
    for durable in (result.run_state.history, agent.run_state.history):
        assert not _contains(durable, _VIEW_NOTE)
        assert inventory in durable
    assert not _contains(snapshot.to_json(), _VIEW_NOTE)
    assert _contains(snapshot.to_json(), "INVENTORY")

    # The next run continues from the committed state without the note.
    await agent.run("next")

    assert not _contains(model.calls[2], _VIEW_NOTE)
    assert model.calls[2][-1] == {"role": "user", "content": "next"}


@pytest.mark.asyncio
async def test_restored_agents_drop_ephemeral_context_from_saved_state(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "assistant.json"
    await _agent(
        SequenceModel([make_assistant_response("1")]), state_path=state_path
    ).run("first")

    # An agent restored from a state path sends the note but does not save it.
    model = SequenceModel([make_assistant_response("2")])
    restored = _agent(model, state_path=state_path)
    await restored.run(["second", _view_note()])

    assert model.calls[0][-1] == {"role": "user", "content": _VIEW_NOTE}
    assert _VIEW_NOTE not in state_path.read_text()
    assert "second" in state_path.read_text()

    # An agent restored from a snapshot sends only the note of its own run.
    snapshot = restored.snapshot()
    assert snapshot is not None
    model = SequenceModel([make_assistant_response("3")])
    from_snapshot = _agent(model, snapshot=snapshot)
    await from_snapshot.run(["third", _view_note("NEW NOTE")])

    assert not _contains(model.calls[0], _VIEW_NOTE)
    assert model.calls[0][-1] == {"role": "user", "content": "NEW NOTE"}
    assert not _contains(from_snapshot.snapshot(), "NEW NOTE")


@pytest.mark.asyncio
async def test_stream_entry_points_commit_state_without_ephemeral_context() -> None:
    stream_model = StreamingSequenceModel([make_assistant_response("streamed")])
    stream_agent = _agent(stream_model)

    model_stream = await stream_agent.run_stream(["question", _view_note()])
    async for _ in model_stream:
        pass
    stream_result = await model_stream.result()

    assert _contains(stream_model.calls[0], _VIEW_NOTE)
    assert stream_result.run_state is not None
    assert not _contains(stream_result.run_state.history, _VIEW_NOTE)
    assert not _contains(stream_agent.snapshot(), _VIEW_NOTE)

    event_model = SequenceModel([_tool_round(), make_assistant_response("done")])
    event_agent = _agent(event_model)

    event_stream = await event_agent.run_events(["question", _view_note()])
    events = [event async for event in event_stream]
    event_result = await event_stream.result()

    # Live model-start events keep the note; public records omit it.
    llm_starts = [event for event in events if isinstance(event, RunLLMStartEvent)]
    assert all(
        any(is_ephemeral_message(message) for message in event.messages)
        for event in llm_starts
    )
    assert all(_VIEW_NOTE not in event.dumps() for event in events)

    # Working history holds the note until the run ends; run_end shows the commit.
    assert event_result.run_state is not None
    committed_length = len(event_result.run_state.history)
    snapshots = {
        event.boundary: event.snapshot.history_length
        for event in events
        if isinstance(event, RunStateSnapshotEvent)
    }
    assert snapshots[RunStateSnapshotBoundary.RUN_START] == 2
    assert snapshots[RunStateSnapshotBoundary.RUN_END] == committed_length
    assert not _contains(event_result.run_state.history, _VIEW_NOTE)


@pytest.mark.asyncio
async def test_failed_and_closed_runs_keep_clean_committed_baseline() -> None:
    model = _FailingModel([make_assistant_response("first answer")])
    agent = _agent(model)
    await agent.run("first")
    baseline = agent.run_state

    with pytest.raises(RuntimeError, match="model failed"):
        await agent.run(["second", _view_note()])

    assert _contains(model.calls[-1], _VIEW_NOTE)
    assert agent.run_state == baseline
    assert not _contains(agent.snapshot(), _VIEW_NOTE)

    # Closing a stream cancels the run without a commit.
    stream_model = _BlockingStreamModel([make_assistant_response("first answer")])
    stream_agent = _agent(stream_model)
    await stream_agent.run("first")
    baseline = stream_agent.run_state

    stream = await stream_agent.run_stream(["second", _view_note()])
    while not stream_model.calls:
        await asyncio.sleep(0)
    await stream.aclose()
    with pytest.raises(asyncio.CancelledError):
        await stream.result()

    assert stream_agent.run_state == baseline
    assert not _contains(stream_agent.snapshot(), _VIEW_NOTE)


@pytest.mark.asyncio
async def test_ephemeral_steering_input_is_not_committed() -> None:
    model = _SteeringModel(
        [_tool_round(), make_assistant_response("done")], steering=[_view_note()]
    )
    agent = _agent(model)

    stream = await agent.run_stream("question")
    async for _ in stream:
        pass
    result = await stream.result()

    assert not _contains(model.calls[0], _VIEW_NOTE)
    assert _contains(model.calls[1], _VIEW_NOTE)
    assert result.run_state is not None
    assert not _contains(result.run_state.history, _VIEW_NOTE)
    assert not _contains(agent.snapshot(), _VIEW_NOTE)


@pytest.mark.asyncio
async def test_shim_run_start_context_is_sent_once_per_run_and_not_committed() -> None:
    model = SequenceModel(
        [_tool_round(), make_assistant_response("done"), make_assistant_response("2")]
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="Assistant",
            model=model,
            tools=Tools(tools=[echo_tool("echo")]),
            shims=(_ViewContextShim(),),
        )
    )

    result = await agent.run("question")
    await agent.run("next")

    note = {"role": "user", "content": _VIEW_NOTE}
    assert [call[:2] for call in model.calls[:2]] == [
        [{"role": "user", "content": "question"}, note]
    ] * 2
    assert model.calls[2].count(note) == 1
    assert model.calls[2][-1] == note
    assert result.run_state is not None
    assert not _contains(result.run_state.history, _VIEW_NOTE)
    assert not _contains(agent.snapshot(), _VIEW_NOTE)


@pytest.mark.asyncio
async def test_handoff_child_sees_ephemeral_context_until_run_ends() -> None:
    child_model = SequenceModel([make_assistant_response("specialist answer")])
    parent_model = SequenceModel(
        [
            make_assistant_response(
                None,
                tool_calls=[
                    make_tool_call(
                        tool_id="handoff_1",
                        name="specialist",
                        arguments='{"task":"Take over."}',
                    )
                ],
            )
        ]
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="Manager",
            model=parent_model,
            handoffs=(AgentDescriptor(name="Specialist", model=child_model),),
        )
    )

    result = await agent.run(["question", _view_note()])

    assert result.final_output == "specialist answer"
    assert _contains(parent_model.calls[0], _VIEW_NOTE)
    assert _contains(child_model.calls[0], _VIEW_NOTE)
    assert result.run_state is not None
    assert not _contains(result.run_state.history, _VIEW_NOTE)
    assert not _contains(agent.snapshot(), _VIEW_NOTE)


@pytest.mark.asyncio
async def test_agent_tool_child_does_not_receive_ephemeral_context() -> None:
    child_model = SequenceModel([make_assistant_response("researched")])
    parent_model = SequenceModel(
        [
            make_assistant_response(
                None,
                tool_calls=[
                    make_tool_call(
                        tool_id="call_1",
                        name="agent",
                        arguments='{"name":"Researcher","task":"Research it."}',
                    )
                ],
            ),
            make_assistant_response("done"),
        ]
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="Manager",
            model=parent_model,
            shims=(HarnessToolsShim((agent_tool(model=child_model),)),),
        )
    )

    await agent.run(["question", _view_note()])

    assert all(_contains(call, _VIEW_NOTE) for call in parent_model.calls)
    assert child_model.calls[0][-1] == {"role": "user", "content": "Research it."}
    assert not _contains(child_model.calls, _VIEW_NOTE)


def test_snapshot_capture_excludes_ephemeral_and_keeps_persistent_context() -> None:
    inventory = internal_message(
        {"role": "user", "content": "INVENTORY"}, origin="inventory"
    )
    state = RunState(
        instructions=None,
        history=["question", _view_note(), inventory],
        responses=[],
    )

    snapshot = AgentSnapshot.capture(
        agent_id=AgentId.from_values("assistant", "main"), run_state=state
    )

    assert snapshot.to_run_state().history == [
        {"role": "user", "content": "question"},
        inventory,
    ]
    assert len(state.history) == 3


@pytest.mark.asyncio
async def test_constructor_baseline_drops_ephemeral_context_but_run_input_keeps_it() -> (
    None
):
    state = RunState(
        instructions=None, history=["question", _view_note()], responses=[]
    )

    # A baseline has no active run, so its ephemeral context is already stale.
    model = SequenceModel([make_assistant_response("answer")])
    agent = _agent(model, run_state=state)
    assert agent.run_state is not None
    assert agent.run_state.history == ["question"]
    await agent.run("next")
    assert not _contains(model.calls[0], _VIEW_NOTE)

    # A state passed as run input starts a run, so the run keeps the context.
    model = SequenceModel([make_assistant_response("answer")])
    resumed = _agent(model)
    result = await resumed.run(state)
    assert model.calls[0][-1] == {"role": "user", "content": _VIEW_NOTE}
    assert result.run_state is not None
    assert not _contains(result.run_state.history, _VIEW_NOTE)
    assert len(state.history) == 2


@pytest.mark.asyncio
async def test_direct_runner_result_state_excludes_ephemeral_context() -> None:
    runner = Runner()
    model = SequenceModel([make_assistant_response("answer")])
    agent = Agent(
        SingleThreadedRuntimeEngine(),
        runner,
        bind_id=AgentId.from_values("assistant", "direct"),
        descriptor=AgentDescriptor(name="Assistant", model=model),
    )
    state = RunState(
        instructions=None, history=["question", _view_note()], responses=[]
    )

    result = await runner.run(agent, state)

    assert _contains(model.calls[0], _VIEW_NOTE)
    assert result.run_state is not None
    assert result.run_state.history[0] == "question"
    assert not _contains(result.run_state.history, _VIEW_NOTE)
