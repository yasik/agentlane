"""Tests for the imported OpenAI Responses client."""

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from agentlane_openai import ResponsesClient
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)
from pydantic import BaseModel

from agentlane.models import Config
from agentlane.models import Tool as NativeTool
from agentlane.models import Tools
from agentlane.runtime import CancellationToken


class EchoArgs(BaseModel):
    """Arguments for the native tool used in client tests."""

    text: str


class StructuredResponse(BaseModel):
    """Schema returned by the mocked response."""

    message: str


async def _echo_handler(
    args: EchoArgs,
    cancellation_token: CancellationToken,
) -> str:
    """Return a simple string result for tool forwarding checks."""
    del cancellation_token
    return args.text


def _make_response(content: str) -> MagicMock:
    """Create one mocked OpenAI Responses API response."""
    response = MagicMock(spec=OpenAIResponse)
    response.id = "resp_123"
    response.model = "gpt-4o"
    response.created_at = 1234567890.0
    response.output = [
        ResponseOutputMessage(
            id="msg_123",
            role="assistant",
            status="completed",
            type="message",
            content=[
                ResponseOutputText(
                    type="output_text",
                    text=content,
                    annotations=[],
                )
            ],
        )
    ]
    response.usage = ResponseUsage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )
    response.status = "completed"
    response.model_dump.return_value = {}
    return response


def test_responses_client_get_response_converts_and_forwards_configuration() -> None:
    """The client should convert Responses API output and forward tools/schema config."""
    client = ResponsesClient(
        Config(
            api_key="test-key",
            model="gpt-4o",
            enforce_structured_output=True,
        )
    )
    tool: NativeTool[EchoArgs, str] = NativeTool(
        name="echo",
        description="Echo text",
        args_model=EchoArgs,
        handler=_echo_handler,
    )
    create_mock = AsyncMock(return_value=_make_response('{"message": "hello"}'))
    openai_client = cast(Any, client)._openai_client
    openai_client.responses.create = create_mock

    response = asyncio.run(
        client.get_response(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Return JSON."},
            ],
            schema=StructuredResponse,
            tools=Tools(
                tools=[tool],
                parallel_tool_calls=True,
            ),
        )
    )

    assert response.choices[0].message.content == '{"message": "hello"}'
    assert response.choices[0].finish_reason == "stop"

    await_kwargs = cast(dict[str, Any], cast(Any, create_mock.await_args).kwargs)
    assert await_kwargs["model"] == "gpt-4o"
    assert await_kwargs["input"][0]["role"] == "developer"
    assert await_kwargs["input"][1]["role"] == "user"
    assert await_kwargs["tools"][0]["name"] == "echo"
    assert await_kwargs["parallel_tool_calls"] is True
    assert await_kwargs["text"]["format"]["type"] == "json_schema"
    assert await_kwargs["text"]["format"]["schema"]["properties"] == {
        "message": {"title": "Message", "type": "string"}
    }
