"""Tests for the imported OpenAI Responses client."""

import asyncio
from copy import deepcopy
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from agentlane_openai import ResponsesClient, ResponsesFactory
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)
from pydantic import BaseModel

from agentlane.models import (
    Config,
    ModelStreamEvent,
    ModelStreamEventKind,
    ModelTracing,
)
from agentlane.models import Tool as NativeTool
from agentlane.models import (
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
    """Arguments for the native tool used in client tests."""

    text: str


class StructuredResponse(BaseModel):
    """Schema returned by the mocked response."""

    message: str


async def _echo_handler(
    args: EchoArgs,
    cancellation_token: CancellationToken,
    context: ToolExecutionContext,
) -> str:
    """Return a simple string result for tool forwarding checks."""
    del context
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


def _make_tool_call_response() -> MagicMock:
    """Create one mocked Responses API tool-call response."""
    response = MagicMock(spec=OpenAIResponse)
    response.id = "resp_tool_123"
    response.model = "gpt-4o"
    response.created_at = 1234567890.0
    response.output = [
        ResponseFunctionToolCall(
            arguments='{"text":"hello"}',
            call_id="call_1",
            name="echo",
            type="function_call",
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


class _FakeAsyncStream:
    """Minimal async stream used for provider streaming tests."""

    def __init__(self, events: list[object]) -> None:
        self._events = iter(events)
        self.closed = False

    def __aiter__(self) -> "_FakeAsyncStream":
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._events)
        except StopIteration as error:
            raise StopAsyncIteration from error

    async def close(self) -> None:
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


async def _collect_stream_events(client: ResponsesClient) -> list[ModelStreamEvent]:
    """Collect one streamed client response into a list of events."""
    events: list[ModelStreamEvent] = []
    async for event in client.stream_response(
        messages=[{"role": "user", "content": "hello"}],
    ):
        events.append(event)
    return events


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


@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("text_type", ["text", "input_text", "output_text"])
@pytest.mark.asyncio
async def test_responses_client_full_history_preserves_request_content(
    provider: str, streaming: bool, text_type: str
) -> None:
    """Both request modes must retain instructions, assistant text, and tool order."""
    client = ResponsesClient(
        Config(
            api_key="test-key",
            model="azure/gpt-4o" if provider == "azure" else "gpt-4o",
            base_url=(
                "https://example.openai.azure.com/" if provider == "azure" else None
            ),
            tracing=ModelTracing.DISABLED,
        )
    )

    # A complete tool round-trip exposes lost instructions and shifted history.
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Be helpful."},
        {"role": "developer", "content": "Use the supplied history."},
        {
            "role": "developer",
            "content": [
                {"type": "text", "text": "Keep instructions. "},
                {"type": "text", "text": "Keep prior answers."},
            ],
        },
        {"role": "user", "content": "Remember code 123."},
        {"role": "assistant", "content": "I remember code 123."},
        {"role": "user", "content": "Check the code."},
        {
            "role": "assistant",
            "phase": "commentary",
            "content": [
                {"type": text_type, "text": "I will "},
                {"type": text_type, "text": ""},
                {"type": text_type, "text": "check code 123."},
            ],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"code":123}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Code 123 is valid."},
        {"role": "user", "content": "What did you find?"},
    ]
    original_messages = deepcopy(messages)

    # Inspect the SDK request without sending a live request to either provider.
    response = _make_response("Code 123 is valid.")
    stream = _FakeAsyncStream([MagicMock(type="response.completed", response=response)])
    create_mock = AsyncMock(return_value=stream if streaming else response)
    openai_client = cast(Any, client)._openai_client
    openai_client.responses.create = create_mock

    if streaming:
        events = [event async for event in client.stream_response(messages)]
        assert events[-1].kind == ModelStreamEventKind.COMPLETED
        assert stream.closed
    else:
        await client.get_response(messages)

    # Assert the full ordered payload, not just the presence of selected text.
    create_mock.assert_awaited_once()
    payload = cast(dict[str, Any], cast(Any, create_mock.await_args).kwargs)
    assert payload["input"] == [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "Be helpful."}],
        },
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "Use the supplied history."}],
        },
        {
            "type": "message",
            "role": "developer",
            "content": [
                {"type": "input_text", "text": "Keep instructions. "},
                {"type": "input_text", "text": "Keep prior answers."},
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Remember code 123."}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "I remember code 123.",
                    "annotations": [],
                }
            ],
            "status": "completed",
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Check the code."}],
        },
        {
            "type": "message",
            "role": "assistant",
            "phase": "commentary",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "I will ", "annotations": []},
                {"type": "output_text", "text": "check code 123.", "annotations": []},
            ],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"code":123}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "Code 123 is valid.",
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What did you find?"}],
        },
    ]

    assert messages == original_messages
    assert "previous_response_id" not in payload

    await openai_client.close()


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        [],
        [{"type": "text", "text": ""}],
        [{"type": "input_text", "text": ""}],
        [{"type": "output_text", "text": "", "annotations": []}],
        [{"type": "text"}],
        [{"type": "text", "text": None}],
    ],
)
def test_responses_client_empty_assistant_content_preserves_tool_calls(
    content: object,
) -> None:
    """Tool-only assistant turns must not create empty message items."""
    client = ResponsesClient(Config(api_key="test-key", model="gpt-4o"))

    result = cast(Any, client)._messages_to_input(
        [
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    },
                ],
            }
        ]
    )

    assert result == [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{}",
        }
    ]


def test_responses_client_assistant_parts_preserve_refusals_and_annotations() -> None:
    """Assistant history must retain refusal parts and existing text metadata."""
    client = ResponsesClient(Config(api_key="test-key", model="gpt-4o"))

    # Mix converted and native parts to detect metadata loss during replay.
    annotation = {
        "type": "url_citation",
        "start_index": 0,
        "end_index": 6,
        "title": "Source",
        "url": "https://example.com",
    }
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "phase": "final_answer",
            "id": "msg_prior",
            "status": "incomplete",
            "content": [
                {"type": "text", "text": "Allowed text."},
                {"type": "refusal", "refusal": "I cannot provide that."},
                {"type": "output_text", "text": "Source", "annotations": [annotation]},
            ],
        },
        {"role": "assistant", "content": None, "refusal": "I cannot help with that."},
    ]
    original_messages = deepcopy(messages)

    result = cast(Any, client)._messages_to_input(messages)

    assert result == [
        {
            "type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "id": "msg_prior",
            "status": "incomplete",
            "content": [
                {"type": "output_text", "text": "Allowed text.", "annotations": []},
                {"type": "refusal", "refusal": "I cannot provide that."},
                {"type": "output_text", "text": "Source", "annotations": [annotation]},
            ],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "refusal", "refusal": "I cannot help with that."}],
            "status": "completed",
        },
    ]

    assert messages == original_messages


def test_responses_factory_forwards_default_model_args() -> None:
    """Factory kwargs outside Config should become default Responses API args."""
    factory = ResponsesFactory(Config(api_key="test-key", model="gpt-4o"))
    client = factory.get_model_client(
        temperature=0.2,
        prompt_cache_retention="24h",
        verbosity="low",
    )
    create_mock = AsyncMock(return_value=_make_response("hello"))
    openai_client = cast(Any, client)._openai_client
    openai_client.responses.create = create_mock

    response = asyncio.run(
        client.get_response(
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.choices[0].message.content == "hello"

    await_kwargs = cast(dict[str, Any], cast(Any, create_mock.await_args).kwargs)
    assert await_kwargs["temperature"] == 0.2
    assert await_kwargs["prompt_cache_retention"] == "24h"
    assert await_kwargs["text"]["verbosity"] == "low"


def test_responses_client_returns_raw_tool_calls_without_executing_them() -> None:
    """Provider clients should return raw tool-call responses unchanged."""
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
    create_mock = AsyncMock(return_value=_make_tool_call_response())
    openai_client = cast(Any, client)._openai_client
    openai_client.responses.create = create_mock

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


def test_responses_client_stream_response_emits_semantic_events() -> None:
    """Streaming should expose text, reasoning, provider, and completed events."""
    client = ResponsesClient(Config(api_key="test-key", model="gpt-4o"))
    stream = _FakeAsyncStream(
        [
            MagicMock(
                type="response.output_text.delta",
                delta="Hel",
                output_index=0,
            ),
            MagicMock(
                type="response.reasoning_text.delta",
                delta="thinking",
                output_index=0,
            ),
            MagicMock(
                type="response.output_item.added",
                output_index=1,
                item=MagicMock(type="reasoning"),
            ),
            MagicMock(
                type="response.completed",
                response=_make_response("Hello"),
            ),
        ]
    )
    create_mock = AsyncMock(return_value=stream)
    openai_client = cast(Any, client)._openai_client
    openai_client.responses.create = create_mock

    events = asyncio.run(_collect_stream_events(client))

    assert [event.kind for event in events] == [
        ModelStreamEventKind.TEXT_DELTA,
        ModelStreamEventKind.REASONING,
        ModelStreamEventKind.PROVIDER,
        ModelStreamEventKind.COMPLETED,
    ]
    assert events[0].text == "Hel"
    assert events[1].reasoning == "thinking"
    assert events[2].provider_event_type == "response.output_item.added"
    assert events[3].response is not None
    assert events[3].response.choices[0].message.content == "Hello"
    assert stream.closed is True

    await_kwargs = cast(dict[str, Any], cast(Any, create_mock.await_args).kwargs)
    assert await_kwargs["stream"] is True
    assert await_kwargs["model"] == "gpt-4o"


def test_responses_client_stream_response_traces_serialized_events() -> None:
    """Streaming should retain serialized events on the generation span."""
    original_provider = get_trace_provider()
    provider = DefaultTraceProvider()
    processor = _CollectingTracingProcessor()
    provider.register_processor(processor)
    set_trace_provider(provider)

    client = ResponsesClient(
        Config(
            api_key="test-key",
            model="gpt-4o",
            tracing=ModelTracing.ENABLED,
        )
    )
    stream = _FakeAsyncStream(
        [
            MagicMock(
                type="response.output_text.delta",
                delta="Hel",
                output_index=0,
            ),
            MagicMock(
                type="response.completed",
                response=_make_response("Hello"),
            ),
        ]
    )
    openai_client = cast(Any, client)._openai_client
    openai_client.responses.create = AsyncMock(return_value=stream)

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
