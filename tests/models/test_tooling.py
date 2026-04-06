"""Tests for the native Tool primitive and tool execution."""

import asyncio
import json
from collections.abc import Callable
from typing import Annotated, cast

import pytest
from pydantic import BaseModel

from agentlane.models import (
    ModelBehaviorError,
    Tool,
    ToolCall,
    ToolExecutor,
    Tools,
    as_tool,
)
from agentlane.runtime import CancellationToken


class EchoArgs(BaseModel):
    """Arguments for the test echo tool."""

    text: str


class EchoResult(BaseModel):
    """Structured return value for the test echo tool."""

    echoed: str


async def _echo_handler(
    args: EchoArgs,
    cancellation_token: CancellationToken,
) -> EchoResult:
    """Return a structured echo result."""
    del cancellation_token
    return EchoResult(echoed=args.text)


def _make_tool_call(arguments: str) -> ToolCall:
    """Build one tool call payload for tests."""
    return ToolCall.model_validate(
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "echo",
                "arguments": arguments,
            },
        }
    )


def test_tool_runs_and_formats_pydantic_result() -> None:
    """Tool.run should execute the handler and stringify Pydantic output."""
    tool = Tool(
        name="echo",
        description="Echo text",
        args_model=EchoArgs,
        handler=_echo_handler,
    )

    result = asyncio.run(
        tool.run(
            EchoArgs(text="hello"),
            CancellationToken(),
        )
    )

    assert result == EchoResult(echoed="hello")
    assert json.loads(tool.return_value_as_string(result)) == {"echoed": "hello"}


def test_tool_from_function_infers_name_description_and_schema() -> None:
    """Tool.from_function should infer the schema from a typed callable."""

    async def search_docs(query: str, limit: int = 3) -> str:
        """Search the docs for the requested topic."""
        return f"{query}:{limit}"

    tool = Tool.from_function(search_docs)

    args_model = tool.args_type()
    result = asyncio.run(
        tool.run(
            args_model(query="harness"),
            CancellationToken(),
        )
    )

    assert tool.name == "search_docs"
    assert tool.description == "Search the docs for the requested topic."
    assert result == "harness:3"
    assert args_model(query="harness").model_dump()["limit"] == 3
    assert tool.schema["parameters"]["required"] == ["query", "limit"]
    assert tool.schema["parameters"]["properties"]["limit"]["default"] == 3


def test_tool_from_function_supports_annotated_descriptions_and_cancellation() -> None:
    """`cancellation_token` should be injected and not exposed in the schema."""
    received_tokens: list[CancellationToken] = []

    async def lookup_city(
        city: Annotated[str, "City to search for"],
        cancellation_token: CancellationToken,
    ) -> str:
        """Look up one city."""
        received_tokens.append(cancellation_token)
        return city.upper()

    tool = Tool.from_function(lookup_city)
    token = CancellationToken()
    args_model = tool.args_type()
    result = asyncio.run(tool.run(args_model(city="berlin"), token))

    assert result == "BERLIN"
    assert received_tokens == [token]
    assert "cancellation_token" not in tool.schema["parameters"]["properties"]
    assert tool.schema["parameters"]["properties"]["city"]["description"] == (
        "City to search for"
    )


def test_tool_from_function_raises_for_missing_parameter_annotations() -> None:
    """Visible callable parameters must be annotated for schema inference."""

    def invalid_tool(query: str, limit: int) -> str:
        return f"{query}:{limit}"

    invalid_tool.__annotations__.pop("query")

    with pytest.raises(TypeError, match="Missing: query"):
        Tool.from_function(cast(Callable[..., object], invalid_tool))


def test_as_tool_decorates_typed_callable() -> None:
    """as_tool should return a native Tool from a typed function declaration."""

    @as_tool
    async def lookup_order(
        order_id: str,
        cancellation_token: CancellationToken,
    ) -> str:
        """Look up one order."""
        del cancellation_token
        return f"order:{order_id}"

    args_model = lookup_order.args_type()
    result = asyncio.run(
        lookup_order.run(
            args_model(order_id="A-123"),
            CancellationToken(),
        )
    )

    assert isinstance(lookup_order, Tool)
    assert lookup_order.name == "lookup_order"
    assert lookup_order.description == "Look up one order."
    assert result == "order:A-123"


def test_as_tool_supports_configured_overrides() -> None:
    """as_tool(...) should support explicit native tool overrides."""

    @as_tool(name="help_search", description="Search the help center.")
    async def search_help_center(question: str) -> str:
        return f"result:{question}"

    args_model = search_help_center.args_type()
    result = asyncio.run(
        search_help_center.run(
            args_model(question="returns"),
            CancellationToken(),
        )
    )

    assert isinstance(search_help_center, Tool)
    assert search_help_center.name == "help_search"
    assert search_help_center.description == "Search the help center."
    assert result == "result:returns"


def test_tools_accept_plain_typed_callables() -> None:
    """Tools should normalize plain typed callables into native Tool values."""

    async def search_help_center(question: str) -> str:
        """Search the help center for one policy answer."""
        return f"result:{question}"

    tools = Tools(tools=[search_help_center])

    assert len(tools.normalized_tools) == 1
    normalized_tool = tools.normalized_tools[0]
    assert isinstance(normalized_tool, Tool)
    assert normalized_tool.name == "search_help_center"
    assert (
        normalized_tool.description == "Search the help center for one policy answer."
    )
    assert tools.as_args()["tools"] == [
        {
            "type": "function",
            "function": normalized_tool.schema,
        }
    ]


def test_tool_executor_returns_chat_completion_tool_message() -> None:
    """The default tool adapter should emit a chat-completions tool message."""
    tool = Tool(
        name="echo",
        description="Echo text",
        args_model=EchoArgs,
        handler=_echo_handler,
    )
    executor = ToolExecutor()

    messages = asyncio.run(
        executor.execute(
            tool_calls=[_make_tool_call('{"text": "hello"}')],
            tools=Tools(tools=[tool]),
        )
    )

    assert messages == [
        {
            "tool_call_id": "call_1",
            "role": "tool",
            "name": "echo",
            "content": '{"echoed":"hello"}',
        }
    ]


def test_tool_executor_raises_for_unregistered_tool() -> None:
    """Unknown tool calls should fail fast with ModelBehaviorError."""
    executor = ToolExecutor()

    with pytest.raises(ModelBehaviorError, match="Tool 'echo' is not registered"):
        asyncio.run(
            executor.execute(
                tool_calls=[_make_tool_call('{"text": "hello"}')],
                tools=Tools(tools=[]),
            )
        )
