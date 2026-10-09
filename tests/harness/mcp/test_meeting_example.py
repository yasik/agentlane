"""Exercise the meeting example through DefaultAgent and a real MCP subprocess."""

import asyncio
import json
import runpy
from pathlib import Path
from typing import Any

import pytest

from ..tools_test_utils import SequenceModel, make_assistant_response, make_tool_call

EXAMPLE_PATH = Path(__file__).parents[3] / "examples/harness/mcp_meeting_assistant"


@pytest.fixture(name="example")
def fixture_example() -> dict[str, Any]:
    return runpy.run_path(str(EXAMPLE_PATH / "main.py"))


def source_answer() -> str:
    meetings = json.loads((EXAMPLE_PATH / "meetings.json").read_text())
    meeting = next(item for item in meetings if item["id"] == "atlas-launch")
    return json.dumps(
        {
            field: meeting[field]
            for field in ("id", "decision", "owner", "due_date", "quote")
        }
    )


def scripted_model(answer: str) -> SequenceModel:
    return SequenceModel(
        [
            make_assistant_response(
                None,
                tool_calls=[
                    make_tool_call(
                        tool_id="search-call",
                        name="meetings__search_meetings",
                        arguments='{"query":"Atlas"}',
                    )
                ],
            ),
            make_assistant_response(
                None,
                tool_calls=[
                    make_tool_call(
                        tool_id="read-call",
                        name="meetings__get_meeting",
                        arguments='{"meeting_id":"atlas-launch"}',
                    )
                ],
            ),
            make_assistant_response(answer),
        ]
    )


@pytest.mark.asyncio
async def test_meeting_example_discovers_calls_and_verifies_real_mcp(
    example: dict[str, Any],
) -> None:
    initial_tasks = set(asyncio.all_tasks())
    model = scripted_model(source_answer())
    report = await example["run_example"](model)

    assert report["verified"] is True
    assert report["manager_closed"] is True
    assert report["model_turns"] == 3
    assert [call["tool"] for call in report["tool_calls"]] == [
        "meetings__search_meetings",
        "meetings__get_meeting",
    ]
    initial_tools = model.call_tools[0]
    assert initial_tools is not None
    assert {tool.name for tool in initial_tools.normalized_tools} == {
        "meetings__search_meetings",
        "meetings__get_meeting",
    }
    assert all(tool.strict is False for tool in initial_tools.normalized_tools)
    second_tools = model.call_tools[1]
    assert second_tools is not None
    assert [tool.name for tool in second_tools.normalized_tools] == [
        "meetings__get_meeting"
    ]
    assert model.call_tools[2] is None
    assert "Maya Chen" not in json.dumps(model.calls[0])
    search_result = next(
        message for message in model.calls[1] if message.get("role") == "tool"
    )
    assert "atlas-launch" in str(search_result["content"])
    assert "Maya Chen" not in str(search_result["content"])
    read_result = [
        message for message in model.calls[2] if message.get("role") == "tool"
    ][-1]
    assert "Maya Chen" in str(read_result["content"])
    assert "2026-10-09" in str(read_result["content"])
    await asyncio.sleep(0)
    assert not [task for task in asyncio.all_tasks() - initial_tasks if not task.done()]
    print(json.dumps({"mode": "deterministic-test-model", **report}, indent=2))


@pytest.mark.asyncio
async def test_meeting_example_rejects_correct_prose_without_mcp_calls(
    example: dict[str, Any],
) -> None:
    model = SequenceModel([make_assistant_response(source_answer())])
    with pytest.raises(RuntimeError, match="MCP search followed by an MCP read"):
        await example["run_example"](model)


@pytest.mark.asyncio
async def test_meeting_example_rejects_answer_with_wrong_facts(
    example: dict[str, Any],
) -> None:
    answer = json.loads(source_answer())
    # Keep the correct name in the quote; it cannot compensate for a wrong owner.
    answer["owner"] = "Wrong Person"
    model = scripted_model(json.dumps(answer))
    with pytest.raises(RuntimeError, match="does not contain all source facts"):
        await example["run_example"](model)
