"""Tests for the imported LiteLLM client wrapper."""

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock

import litellm
import pytest
from agentlane_litellm import Client, Factory
from pydantic import BaseModel

from agentlane.models import Config, ModelResponse, Tool, ToolCall, Tools
from agentlane.runtime import CancellationToken


class EchoArgs(BaseModel):
    """Arguments for the native tool used in LiteLLM client tests."""

    text: str


class StructuredResponse(BaseModel):
    """Structured response schema used in adapter tests."""

    message: str


async def _echo_handler(
    args: EchoArgs,
    cancellation_token: CancellationToken,
) -> str:
    """Return the provided text."""
    del cancellation_token
    return args.text


def _make_model_response(
    content: str | None,
    *,
    tool_calls: list[ToolCall] | None = None,
) -> ModelResponse:
    """Create one canonical model response object."""
    return ModelResponse.model_validate(
        {
            "id": "chatcmpl_123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
    )


def _make_tool_call(arguments: str) -> ToolCall:
    """Create one canonical tool call for adapter tests."""
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


def test_litellm_client_forwards_native_tool_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool metadata should be forwarded using the native AgentLane schema."""
    client = Client(Config(api_key="test-key", model="gpt-4o"))
    tool: Tool[EchoArgs, str] = Tool(
        name="echo",
        description="Echo text",
        args_model=EchoArgs,
        handler=_echo_handler,
    )
    completion_mock = AsyncMock(return_value=_make_model_response("hello"))
    monkeypatch.setattr(litellm, "acompletion", completion_mock)

    response = asyncio.run(
        client.get_response(
            messages=[{"role": "user", "content": "hello"}],
            tools=Tools(
                tools=[tool],
                parallel_tool_calls=True,
            ),
        )
    )

    assert response.choices[0].message.content == "hello"

    await_kwargs = cast(dict[str, Any], cast(Any, completion_mock.await_args).kwargs)
    assert await_kwargs["model"] == "gpt-4o"
    assert await_kwargs["tools"][0]["function"]["name"] == "echo"
    assert await_kwargs["parallel_tool_calls"] is True
    assert await_kwargs["drop_params"] is True


def test_litellm_factory_forwards_default_model_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory kwargs outside Config should become default model-call args."""
    factory = Factory(Config(api_key="test-key", model="gpt-4o"))
    client = factory.get_model_client(
        temperature=0.2,
        max_tokens=128,
    )
    completion_mock = AsyncMock(return_value=_make_model_response("hello"))
    monkeypatch.setattr(litellm, "acompletion", completion_mock)

    response = asyncio.run(
        client.get_response(
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.choices[0].message.content == "hello"

    await_kwargs = cast(dict[str, Any], cast(Any, completion_mock.await_args).kwargs)
    assert await_kwargs["temperature"] == 0.2
    assert await_kwargs["max_tokens"] == 128


def test_litellm_client_returns_raw_tool_calls_without_executing_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider clients should return raw tool-call responses unchanged."""
    client = Client(Config(api_key="test-key", model="gpt-4o"))
    tool: Tool[EchoArgs, str] = Tool(
        name="echo",
        description="Echo text",
        args_model=EchoArgs,
        handler=_echo_handler,
    )
    completion_mock = AsyncMock(
        return_value=_make_model_response(
            None,
            tool_calls=[_make_tool_call('{"text":"hello"}')],
        )
    )
    monkeypatch.setattr(litellm, "acompletion", completion_mock)

    response = asyncio.run(
        client.get_response(
            messages=[{"role": "user", "content": "hello"}],
            schema=StructuredResponse,
            tools=Tools(tools=[tool]),
        )
    )

    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls is not None
    assert len(cast(list[object], response.choices[0].message.tool_calls)) == 1
