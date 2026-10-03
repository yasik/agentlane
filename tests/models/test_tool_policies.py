"""Schema and retry policies for external tool sources."""

import asyncio

import pytest

from agentlane.models import Tool, ToolCall, ToolExecutor, Tools, as_tool


def test_explicit_non_strict_schema_survives_wrapping() -> None:
    """An external schema keeps optional fields and open nested objects."""
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "options": {"type": "object", "additionalProperties": True},
        },
        "required": ["query"],
    }

    @as_tool(parameters_schema=schema, strict=False, retry_on_timeout=False)
    def search(query: str, options: dict[str, object] | None = None) -> str:
        del options
        return query

    copied = search.replace(name="remote__search").with_handler(lambda inner: inner)

    assert copied.strict is False
    assert copied.retry_on_timeout is False
    assert copied.schema["parameters"] == schema
    assert copied.schema["strict"] is False
    assert Tools(tools=(copied,)).as_args()["tools"][0]["function"] == copied.schema


def test_non_strict_inferred_schema_keeps_optional_arguments() -> None:
    """Opting out of strictness does not make defaulted parameters required."""

    def search(query: str, limit: int = 3) -> str:
        return f"{query}:{limit}"

    tool = Tool.from_function(search, strict=False)
    assert tool.schema["parameters"]["required"] == ["query"]


@pytest.mark.parametrize("retry_on_timeout, expected_calls", [(False, 1), (True, 4)])
@pytest.mark.asyncio
async def test_outer_timeout_respects_per_tool_retry_policy(
    retry_on_timeout: bool, expected_calls: int
) -> None:
    """A side effect is dispatched once when the tool disables timeout retries."""
    calls = 0

    async def side_effect() -> str:
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()
        return "unreachable"

    tool = Tool.from_function(side_effect, retry_on_timeout=retry_on_timeout)
    result = await ToolExecutor().execute(
        tool_calls=[
            ToolCall.model_validate(
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "side_effect", "arguments": "{}"},
                }
            )
        ],
        tools=Tools(tools=(tool,), tool_call_timeout=0.01, tool_call_max_retries=3),
    )

    assert calls == expected_calls
    assert "timed out" in result[0]["content"]
