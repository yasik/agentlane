"""Public serializer checks against real AgentLane execution."""

import asyncio
import json
from contextlib import suppress
from typing import Any, cast

import pytest
from pydantic import JsonValue

from agentlane.harness import RunEventKind, RunEventRecord
from agentlane.harness.agents import DefaultAgent
from agentlane.models import PromptSpec

from ..tools_test_utils import SequenceModel


@pytest.mark.asyncio
async def test_serialize_real_agent_preserves_tool_result_and_final_state(
    serializable_agent: DefaultAgent,
) -> None:
    """Every live event is JSON-ready, including prompts in the final run state."""
    stream = await serializable_agent.run_events("Use the echo tool")
    delivered: list[tuple[str, dict[str, JsonValue]]] = []
    try:
        async for event in stream:
            record = event.to_dict()
            event_type, payload = record["type"], record["payload"]
            delivered.append(
                (event_type, json.loads(json.dumps(payload, allow_nan=False)))
            )

        result = await stream.result()
    finally:
        await stream.aclose()
        with suppress(Exception, asyncio.CancelledError):
            await stream.result()

    tools = [
        payload for kind, payload in delivered if kind == RunEventKind.TOOL_END.value
    ]
    assert len(tools) == 1
    assert tools[0]["result"] == "Full tool result.\n" * 1000
    assert result.final_output == "Complete"

    assert delivered[-1][0] == RunEventKind.AGENT_END.value
    serialized_result = cast(dict[str, Any], delivered[-1][1]["result"])
    state = serialized_result["run_state"]
    assert state["instructions"]["messages"] == [
        {"role": "system", "content": "Explain serialization"}
    ]
    assert state["instructions"]["values"] == {"topic": "serialization"}
    assert serialized_result["final_output"] == result.final_output
    assert serialized_result["responses"] == [
        response.model_dump(mode="json") for response in result.responses
    ]


@pytest.mark.asyncio
async def test_serialize_shim_transformed_request_preserves_stored_prompt(
    transformed_request_agent: DefaultAgent,
    transformed_request_model: SequenceModel,
) -> None:
    """The request event captures transformed messages, not a re-render of state."""
    stream = await transformed_request_agent.run_events("Explain this")
    records: list[RunEventRecord] = []
    try:
        async for event in stream:
            record = event.to_dict()
            assert json.loads(event.dumps()) == record
            records.append(record)

        result = await stream.result()
    finally:
        await stream.aclose()
        with suppress(Exception, asyncio.CancelledError):
            await stream.result()

    requests = [
        record["payload"]["messages"]
        for record in records
        if record["type"] == RunEventKind.LLM_START.value
    ]
    expected_messages = [
        {"role": "system", "content": "Explain the transformed request"},
        {"role": "user", "content": "Explain this"},
    ]
    assert requests == [expected_messages]
    assert transformed_request_model.calls == [expected_messages]

    assert records[-1]["type"] == RunEventKind.AGENT_END.value
    serialized_result = cast(dict[str, Any], records[-1]["payload"]["result"])
    assert serialized_result["run_state"]["instructions"] == {
        "messages": [{"role": "system", "content": "Explain stored prompt"}],
        "values": {"topic": "stored prompt", "unused": "retained"},
        "response_format": None,
    }
    assert result.run_state is not None
    assert isinstance(result.run_state.instructions, PromptSpec)
    assert result.run_state.instructions.values == {
        "topic": "stored prompt",
        "unused": "retained",
    }
