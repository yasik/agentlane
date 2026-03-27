"""Tests for the imported LiteLLM client wrapper."""

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock

import litellm
import pytest
from agentlane_litellm import Client
from pydantic import BaseModel

from agentlane.models import Config, ModelResponse, Tool, Tools
from agentlane.runtime import CancellationToken


class EchoArgs(BaseModel):
    """Arguments for the native tool used in LiteLLM client tests."""

    text: str


async def _echo_handler(
    args: EchoArgs,
    cancellation_token: CancellationToken,
) -> str:
    """Return the provided text."""
    del cancellation_token
    return args.text


def _make_model_response(content: str) -> ModelResponse:
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
                    },
                    "finish_reason": "stop",
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


def test_litellm_client_forwards_native_tool_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool metadata should be forwarded using the native AgentLane schema."""
    client = Client(Config(api_key="test-key", model="gpt-4o"))
    tool = Tool(
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
