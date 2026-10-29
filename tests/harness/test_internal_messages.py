"""Internal context stays durable but does not enter public conversation records."""

import json

import pytest

from agentlane.harness import (
    AgentDescriptor,
    AgentSnapshot,
    RunAgentEndEvent,
    RunLLMStartEvent,
    RunResult,
    RunState,
)
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.compaction import render_request_messages
from agentlane.messaging import AgentId
from agentlane.models import (
    MessageDict,
    internal_message,
    is_internal_message,
    provider_messages,
)


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
