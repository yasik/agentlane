"""Tests for the imported LiteLLM client wrapper."""

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock

import litellm
import pytest
from agentlane_litellm import Client, Factory
from litellm.types.utils import ModelResponseStream
from pydantic import BaseModel
from structlog.testing import capture_logs

from agentlane.models import (
    Config,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    ModelTracing,
    Tool,
    ToolCall,
    ToolExecutionContext,
    Tools,
)
from agentlane.runtime import CancellationToken
from agentlane.tracing import (
    DefaultTraceProvider,
    TracingProcessor,
    get_trace_provider,
    set_trace_provider,
    trace,
)


class EchoArgs(BaseModel):
    """Arguments for the native tool used in LiteLLM client tests."""

    text: str


class StructuredResponse(BaseModel):
    """Structured response schema used in adapter tests."""

    message: str


async def _echo_handler(
    args: EchoArgs,
    cancellation_token: CancellationToken,
    context: ToolExecutionContext,
) -> str:
    """Return the provided text."""
    del context
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


def _make_stream_chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_arguments: str | None = None,
    finish_reason: str | None = None,
) -> ModelResponseStream:
    """Create one LiteLLM stream chunk for adapter tests."""
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    if tool_arguments is not None:
        delta["tool_calls"] = [
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": tool_arguments,
                },
            }
        ]

    payload: dict[str, Any] = {
        "id": "chatcmpl_123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if finish_reason is not None:
        payload["usage"] = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
    return ModelResponseStream.model_validate(payload)


class _FakeAsyncStream:
    """Minimal async stream with LiteLLM-compatible close semantics."""

    def __init__(self, chunks: list[ModelResponseStream]) -> None:
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self) -> "_FakeAsyncStream":
        return self

    async def __anext__(self) -> ModelResponseStream:
        try:
            return next(self._chunks)
        except StopIteration as error:
            raise StopAsyncIteration from error

    async def aclose(self) -> None:
        self.closed = True


class _CollectingTracingProcessor(TracingProcessor):
    """Capture finished spans for assertions."""

    def __init__(self) -> None:
        self.spans: list[Any] = []

    def on_trace_start(self, trace: Any) -> None:
        del trace

    def on_trace_end(self, trace: Any) -> None:
        del trace

    def on_span_start(self, span: Any) -> None:
        del span

    def on_span_end(self, span: Any) -> None:
        self.spans.append(span)

    def shutdown(self) -> None:
        return None

    def force_flush(self) -> None:
        return None


async def _collect_stream_events(client: Client) -> list[ModelStreamEvent]:
    """Collect one client stream into a list of events."""
    events: list[ModelStreamEvent] = []
    async for event in client.stream_response(
        messages=[{"role": "user", "content": "hello"}],
    ):
        events.append(event)
    return events


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


def test_litellm_client_preserves_non_strict_external_tool_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External tool strictness and optional arguments reach the provider."""
    schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "options": {"type": "object", "additionalProperties": True},
        },
        "required": ["text"],
    }
    tool: Tool[EchoArgs, str] = Tool(
        name="remote__echo",
        description="Remote echo",
        args_model=EchoArgs,
        handler=_echo_handler,
        parameters_schema=schema,
        strict=False,
    )
    client = Client(Config(api_key="test-key", model="gpt-4o"))
    completion_mock = AsyncMock(return_value=_make_model_response("done"))
    monkeypatch.setattr(litellm, "acompletion", completion_mock)

    asyncio.run(
        client.get_response(
            messages=[{"role": "user", "content": "echo"}], tools=Tools(tools=(tool,))
        )
    )

    kwargs = cast(Any, completion_mock.await_args).kwargs
    assert kwargs["tools"][0]["function"]["strict"] is False
    assert kwargs["tools"][0]["function"]["parameters"] == schema


def test_litellm_client_redacts_credentials_from_request_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request logs must redact credentials without changing provider arguments."""
    api_key = "canary-api-key"
    authorization = "Bearer canary-authorization"
    header_api_key = "canary-header-api-key"
    refresh_token = "canary-refresh-token"  # noqa: S105
    aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"  # noqa: S105
    aws_secret_access_key = "canary-aws-secret-access-key"  # noqa: S105
    google_api_key = "canary-google-api-key"
    vertex_credentials = '{"type":"service_account","private_key":"canary-private-key"}'
    client = Client(
        Config(
            api_key=api_key,
            model="gpt-4o",
            default_headers={
                "Authorization": authorization,
                "X-API-Key": header_api_key,
                "X-Trace-ID": "visible-trace",
            },
        )
    )
    completion_mock = AsyncMock(return_value=_make_model_response("hello"))
    monkeypatch.setattr(litellm, "acompletion", completion_mock)

    with capture_logs() as logs:
        asyncio.run(
            client.get_response(
                messages=[{"role": "user", "content": "hello"}],
                extra_call_args={
                    "provider_options": {
                        "refresh_token": refresh_token,
                        "region": "eu",
                        "aws_access_key_id": aws_access_key_id,
                        "aws_secret_access_key": aws_secret_access_key,
                        "vertex_credentials": vertex_credentials,
                        "headers": {"X-Goog-Api-Key": google_api_key},
                        "input_cost_per_token": 0.0001,
                    }
                },
            )
        )

    rendered_logs = repr(logs)
    assert api_key not in rendered_logs
    assert authorization not in rendered_logs
    assert header_api_key not in rendered_logs
    assert refresh_token not in rendered_logs
    assert aws_access_key_id not in rendered_logs
    assert aws_secret_access_key not in rendered_logs
    assert google_api_key not in rendered_logs
    assert vertex_credentials not in rendered_logs
    started = next(log for log in logs if log["event"] == "LLM call started")
    logged_args = cast(dict[str, Any], started["call_args"])
    logged_headers = cast(dict[str, Any], logged_args["default_headers"])
    logged_options = cast(dict[str, Any], logged_args["provider_options"])
    assert logged_args["api_key"] == "[redacted]"
    assert logged_headers["Authorization"] == "[redacted]"
    assert logged_headers["X-API-Key"] == "[redacted]"
    assert logged_headers["X-Trace-ID"] == "visible-trace"
    assert logged_options == {
        "refresh_token": "[redacted]",
        "region": "eu",
        "aws_access_key_id": "[redacted]",
        "aws_secret_access_key": "[redacted]",
        "vertex_credentials": "[redacted]",
        "headers": {"X-Goog-Api-Key": "[redacted]"},
        "input_cost_per_token": 0.0001,
    }

    await_kwargs = cast(dict[str, Any], cast(Any, completion_mock.await_args).kwargs)
    assert await_kwargs["api_key"] == api_key
    assert await_kwargs["default_headers"]["Authorization"] == authorization
    assert await_kwargs["default_headers"]["X-API-Key"] == header_api_key
    assert await_kwargs["provider_options"]["refresh_token"] == refresh_token
    assert await_kwargs["provider_options"]["aws_access_key_id"] == aws_access_key_id
    assert (
        await_kwargs["provider_options"]["aws_secret_access_key"]
        == aws_secret_access_key
    )
    assert await_kwargs["provider_options"]["vertex_credentials"] == vertex_credentials
    assert (
        await_kwargs["provider_options"]["headers"]["X-Goog-Api-Key"] == google_api_key
    )


def test_litellm_client_redacts_credentials_from_stream_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming request logs must use the same credential redaction."""
    api_key = "canary-stream-api-key"
    client = Client(
        Config(
            api_key=api_key,
            model="gpt-4o",
            default_headers={"Proxy-Authorization": "Bearer canary-proxy"},
        )
    )
    completion_mock = AsyncMock(
        return_value=_FakeAsyncStream(
            [
                _make_stream_chunk(content="hello"),
                _make_stream_chunk(finish_reason="stop"),
            ]
        )
    )
    monkeypatch.setattr(litellm, "acompletion", completion_mock)

    with capture_logs() as logs:
        events = asyncio.run(_collect_stream_events(client))

    assert events[-1].kind == ModelStreamEventKind.COMPLETED
    assert api_key not in repr(logs)
    assert "Bearer canary-proxy" not in repr(logs)
    started = next(log for log in logs if log["event"] == "LLM stream started")
    logged_args = cast(dict[str, Any], started["call_args"])
    logged_headers = cast(dict[str, Any], logged_args["default_headers"])
    assert logged_args["api_key"] == "[redacted]"
    assert logged_headers["Proxy-Authorization"] == "[redacted]"

    await_kwargs = cast(dict[str, Any], cast(Any, completion_mock.await_args).kwargs)
    assert await_kwargs["api_key"] == api_key
    assert await_kwargs["default_headers"]["Proxy-Authorization"] == (
        "Bearer canary-proxy"
    )


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


def test_litellm_client_stream_response_emits_normalized_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming should expose chunk deltas and one final completed event."""
    client = Client(Config(api_key="test-key", model="gpt-4o"))
    stream = _FakeAsyncStream(
        [
            _make_stream_chunk(content="Hel", reasoning_content="thinking"),
            _make_stream_chunk(
                tool_arguments='{"text":"hello"}', finish_reason="tool_calls"
            ),
        ]
    )
    completion_mock = AsyncMock(return_value=stream)
    monkeypatch.setattr(litellm, "acompletion", completion_mock)

    events = asyncio.run(_collect_stream_events(client))

    assert [event.kind for event in events] == [
        ModelStreamEventKind.TEXT_DELTA,
        ModelStreamEventKind.REASONING,
        ModelStreamEventKind.TOOL_CALL_ARGUMENTS_DELTA,
        ModelStreamEventKind.COMPLETED,
    ]
    assert events[0].text == "Hel"
    assert events[1].reasoning == "thinking"
    assert events[2].arguments_delta == '{"text":"hello"}'
    assert events[3].response is not None
    assert events[3].response.choices[0].finish_reason == "tool_calls"
    assert events[3].response.choices[0].message.tool_calls is not None
    assert stream.closed is True

    await_kwargs = cast(dict[str, Any], cast(Any, completion_mock.await_args).kwargs)
    assert await_kwargs["stream"] is True
    assert await_kwargs["drop_params"] is True


def test_litellm_client_stream_response_traces_serialized_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming should retain serialized events on the generation span."""
    original_provider = get_trace_provider()
    provider = DefaultTraceProvider()
    processor = _CollectingTracingProcessor()
    provider.register_processor(processor)
    set_trace_provider(provider)

    client = Client(
        Config(
            api_key="test-key",
            model="gpt-4o",
            tracing=ModelTracing.ENABLED,
        )
    )
    stream = _FakeAsyncStream(
        [
            _make_stream_chunk(content="Hel"),
            _make_stream_chunk(content="lo", finish_reason="stop"),
        ]
    )
    completion_mock = AsyncMock(return_value=stream)
    monkeypatch.setattr(litellm, "acompletion", completion_mock)

    try:
        with trace("streaming-test"):
            events = asyncio.run(_collect_stream_events(client))
    finally:
        set_trace_provider(original_provider)

    assert events[-1].kind == ModelStreamEventKind.COMPLETED
    generation_spans = [
        span for span in processor.spans if span.span_data.type == "generation"
    ]
    assert len(generation_spans) == 1
    assert generation_spans[0].span_data.events is not None
    assert generation_spans[0].span_data.events[0]["kind"] == "text_delta"
    assert generation_spans[0].span_data.events[-1]["kind"] == "completed"
