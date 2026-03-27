"""Tests for the native Tool primitive and tool execution."""

import asyncio
import json

import pytest
from pydantic import BaseModel

from agentlane.models import ModelBehaviorError, Tool, ToolCall, ToolExecutor, Tools
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
