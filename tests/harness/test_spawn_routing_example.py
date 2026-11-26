"""Exercise the routing demo's settings through its public agent setup."""

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

from agentlane.harness import SpawnModelConfig

from .tools_test_utils import SequenceModel, make_assistant_response, make_tool_call


@pytest.fixture(name="example")
def fixture_example() -> dict[str, Any]:
    path = Path(__file__).parents[2] / "examples/harness/spawn_model_routing/main.py"

    return runpy.run_path(str(path))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_can_choose", "mandatory_route", "default_route", "suggestion", "expected"),
    [
        (True, None, "quick", "reasoning", "reasoning"),
        (True, "quick", "quick", "reasoning", "quick"),
        (False, None, "quick", None, "quick"),
        (False, None, "reasoning", None, "reasoning"),
        (True, None, "quick", None, "quick"),
    ],
)
async def test_routing_example_applies_explicit_settings(
    example: dict[str, Any],
    model_can_choose: bool,
    mandatory_route: str | None,
    default_route: str,
    suggestion: str | None,
    expected: str,
) -> None:
    children = {
        name: SequenceModel([make_assistant_response("323")])
        for name in ("quick", "reasoning")
    }
    provider = example["RoutingProvider"](
        {name: SpawnModelConfig(client, None) for name, client in children.items()},
        model_can_choose=model_can_choose,
        mandatory_route=mandatory_route,
        default_route=default_route,
    )
    task = {"name": "Calculator", "task": "Multiply 17 by 19."}
    arguments = (
        {"input": task, "spawn_parameters": {"work_class": suggestion}}
        if model_can_choose
        else task
    )
    parent = SequenceModel(
        [
            make_assistant_response(
                None,
                tool_calls=[
                    make_tool_call(
                        tool_id="delegate",
                        name="agent",
                        arguments=json.dumps(arguments),
                    )
                ],
            ),
            make_assistant_response("323"),
        ]
    )

    result = await example["create_agent"](parent, provider).run(task["task"])

    assert result.final_output == "323"
    assert len(parent.calls) == 2
    assert [name for name, client in children.items() if client.calls] == [expected]
    assert len(children[expected].calls) == 1
    tools = parent.call_tools[0]
    assert tools is not None
    schema = tools.normalized_tools[0].args_type().model_json_schema()
    assert ("spawn_parameters" in schema["properties"]) is model_can_choose
