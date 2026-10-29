"""Internal context stays durable but does not enter public conversation records."""

import json
from typing import cast

from pydantic import JsonValue

from agentlane.harness import (
    AgentSnapshot,
    RunAgentEndEvent,
    RunLLMStartEvent,
    RunResult,
    RunState,
    public_message_projection,
)
from agentlane.harness._render import render_request_messages
from agentlane.harness._run import copy_run_state
from agentlane.messaging import AgentId
from agentlane.models import internal_message, is_internal_message, provider_messages


def test_internal_metadata_survives_copies_and_json_restore() -> None:
    ordinary = {"role": "user", "content": "Find my report", "name": "brain_artifacts"}
    context = internal_message(
        {"role": "user", "content": "Report inventory"}, origin="inventory"
    )
    state = RunState(instructions="Help", history=[ordinary, context], responses=[])
    before = render_request_messages(state.instructions, state.history)

    # Repeated saves must keep the same provider prefix and runtime identity.
    for _ in range(3):
        state = copy_run_state(state)
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


def test_public_projection_covers_requests_results_and_nested_history() -> None:
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
        native = cast(JsonValue, event.to_dict())
        public = public_message_projection(native)
        assert "PRIVATE INVENTORY" in json.dumps(native)
        assert "PRIVATE INVENTORY" not in json.dumps(public)
        assert "Find my report" in json.dumps(public)

    assert public_message_projection(cast(JsonValue, context)) is None
    assert public_message_projection({"nested": cast(JsonValue, context)}) == {}
    assert state.history == [ordinary, context]


def test_old_snapshot_and_ordinary_messages_remain_public() -> None:
    state = RunState(instructions=None, history=["hello"], responses=[])
    snapshot = AgentSnapshot.capture(
        agent_id=AgentId.from_values("assistant", "main"), run_state=state
    )
    restored = AgentSnapshot.from_json(
        json.loads(snapshot.model_dump_json())
    ).to_run_state()
    messages = render_request_messages(restored.instructions, restored.history)

    assert public_message_projection(cast(JsonValue, messages)) == messages
    assert provider_messages(messages) == messages
