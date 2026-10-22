"""Tests for the imported OpenAI Responses client."""

import asyncio
import gc
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest
from agentlane_openai import ResponseError, ResponsesClient, ResponsesFactory
from openai import RateLimitError
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseCustomToolCall,
    ResponseFailedEvent,
    ResponseOutputItemAddedEvent,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_item import ImageGenerationCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)
from pydantic import BaseModel, TypeAdapter

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    AgentSnapshot,
    ExternalTool,
    RunContinuation,
    Runner,
    RunPaused,
    RunState,
    ToolResultInput,
)
from agentlane.harness.compaction import render_request_messages
from agentlane.messaging import AgentId
from agentlane.models import (
    Config,
    MessageDict,
    ModelStreamEvent,
    ModelStreamEventKind,
    ModelTracing,
)
from agentlane.models import Tool as NativeTool
from agentlane.models import (
    ToolExecutionContext,
    Tools,
    extract_retry_after,
)
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine
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
        input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
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
        input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
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


def test_native_options_do_not_mutate_caller_values() -> None:
    reasoning = {"summary": "auto"}
    client = ResponsesClient(
        Config(api_key="test-key", model="future-model"),
        temperature=0.2,
        reasoning_effort="high",
        reasoning=reasoning,
        service_tier="priority",
    )
    create = AsyncMock(return_value=_make_response("ok"))
    cast(Any, client)._openai_client.responses.create = create

    # Merge adapter aliases with native options without changing the caller's values.
    asyncio.run(client.get_response([{"role": "user", "content": "hello"}]))

    assert reasoning == {"summary": "auto"}
    assert create.call_args.kwargs["reasoning"] == {"summary": "auto", "effort": "high"}
    assert create.call_args.kwargs["model"] == "future-model"
    assert create.call_args.kwargs["service_tier"] == "priority"


def test_quota_failure_keeps_sdk_error_without_retry() -> None:
    client = ResponsesClient(Config(api_key="test-key", model="test"))
    error = RateLimitError(
        "quota exhausted",
        response=httpx2.Response(
            429, request=httpx2.Request("POST", "https://example.test")
        ),
        body={"code": "insufficient_quota", "message": "quota exhausted"},
    )
    create = AsyncMock(side_effect=error)
    cast(Any, client)._openai_client.responses.create = create

    async def run() -> None:
        runner = Runner(max_attempts=2)
        agent = Agent(
            SingleThreadedRuntimeEngine(),
            runner,
            descriptor=AgentDescriptor(name="quota", model=client),
        )
        await runner.run(
            agent, RunState(instructions=None, history=["hello"], responses=[])
        )

    # Neither the adapter nor the harness may retry a quota failure.
    with pytest.raises(RateLimitError) as caught:
        asyncio.run(run())

    assert caught.value is error
    assert create.await_count == 1


@pytest.mark.parametrize("delay", ["30", "Thu, 01 Jan 2099 00:00:00 GMT"])
def test_sdk_retry_after_is_read_from_response(delay: str) -> None:
    error = RateLimitError(
        "rate limit",
        response=httpx2.Response(
            429,
            headers={"Retry-After": delay},
            request=httpx2.Request("POST", "https://example.test"),
        ),
        body={"code": "rate_limit_exceeded"},
    )
    seconds = extract_retry_after(error)

    assert seconds is not None and seconds >= 30


def test_failed_stream_retains_native_terminal_response() -> None:
    client = ResponsesClient(Config(api_key="test-key", model="test"))
    response = _make_response("partial")
    response.status = "failed"
    response.error = {"code": "server_error", "message": "failed"}
    stream = _FakeAsyncStream(
        [ResponseFailedEvent.model_construct(type="response.failed", response=response)]
    )
    create = AsyncMock(return_value=stream)
    cast(Any, client)._openai_client.responses.create = create
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(_collect_stream_events(client))

    assert getattr(caught.value, "response", None) is response
    assert stream.closed
    assert create.await_count == 1


def test_native_history_replays_once_after_snapshot() -> None:
    native = OpenAIResponse.model_construct(
        id="resp_native",
        created_at=1,
        model="test",
        object="response",
        status="completed",
        error=None,
        incomplete_details=None,
        usage=None,
        service_tier="priority",
        output=[
            ResponseReasoningItem(
                id="reason_1", type="reasoning", summary=[], encrypted_content="opaque"
            ),
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                phase="commentary",
                content=[
                    ResponseOutputText(
                        type="output_text", text="Working", annotations=[]
                    )
                ],
            ),
            ResponseFunctionToolCall(
                id="fc_1",
                call_id="call_1",
                name="echo",
                arguments='{"text":"ok"}',
                type="function_call",
            ),
        ],
    )
    client = ResponsesClient(Config(api_key="test-key", model="test"))
    create = AsyncMock(return_value=native)
    cast(Any, client)._openai_client.responses.create = create
    response = asyncio.run(client.get_response([{"role": "user", "content": "hello"}]))

    # Round-trip the result through persistence before replaying native output.
    snapshot = AgentSnapshot.capture(
        agent_id=AgentId.from_values("test", "main"),
        run_state=RunState(instructions=None, history=[response], responses=[response]),
    )

    # Decode in a fresh interpreter to check JSON portability without SDK object state.
    code = "from agentlane.harness import AgentSnapshot; import sys,json; s=AgentSnapshot.from_json(json.load(sys.stdin)); print(json.dumps(s.to_json()))"
    decoded = subprocess.run(  # noqa: S603 - fixed interpreter and inline test script
        [sys.executable, "-c", code],
        input=snapshot.model_dump_json(),
        text=True,
        check=True,
        capture_output=True,
    )
    restored = AgentSnapshot.from_json(json.loads(decoded.stdout)).to_run_state()

    # Resume with one tool result and verify native items appear exactly once.
    messages = render_request_messages(
        None,
        [
            *restored.history,
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ],
    )
    asyncio.run(client.get_response(messages))
    sent = create.call_args.kwargs["input"]

    assert sent[:-1] == [
        item.model_dump(mode="json", exclude_unset=True) for item in native.output
    ]
    assert sent[-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "ok",
    }
    extras = restored.responses[0].model_extra

    assert extras is not None
    assert extras["openai_response"]["service_tier"] == "priority"


def test_native_input_and_chat_media_are_preserved() -> None:
    native = [{"type": "configuration_update", "reasoning": {"effort": "high"}}]
    messages: list[MessageDict] = [
        {"role": "user", "openai_responses_input": native},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.test/image.png",
                        "detail": "high",
                    },
                },
                {"type": "input_file", "file_id": "file_1"},
            ],
        },
    ]
    client = ResponsesClient(Config(api_key="test-key", model="test"))
    create = AsyncMock(return_value=_make_response("ok"))
    cast(Any, client)._openai_client.responses.create = create

    # Mix native input with chat media in one request to check both paths.
    asyncio.run(client.get_response(messages))

    assert create.call_args.kwargs["input"] == [
        *native,
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": "https://example.test/image.png",
                    "detail": "high",
                },
                {"type": "input_file", "file_id": "file_1"},
            ],
        },
    ]


def test_responses_client_get_response_converts_and_forwards_configuration() -> None:
    """The client should convert Responses API output and forward tools/schema config."""
    client = ResponsesClient(
        Config(
            api_key="test-key",
            model="gpt-4o",
            enforce_structured_output=True,
        ),
        text={"verbosity": "low"},
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
    assert await_kwargs["tools"][0]["strict"] is True
    assert await_kwargs["text"]["format"]["type"] == "json_schema"
    assert await_kwargs["text"]["verbosity"] == "low"
    assert await_kwargs["text"]["format"]["schema"]["properties"] == {
        "message": {"title": "Message", "type": "string"}
    }


@pytest.mark.parametrize("repair", ["schema", "refusal"])
def test_response_repair_preserves_request_and_appends_feedback(repair: str) -> None:
    """Repair uses the existing request options and leaves caller history unchanged."""
    client = ResponsesClient(
        Config(
            api_key="test-key",
            model="azure/test" if repair == "refusal" else "test",
            base_url=(
                "https://example.openai.azure.com/" if repair == "refusal" else None
            ),
            enforce_structured_output=repair == "schema",
            schema_validation_retries=1,
        ),
        refusal_context="Use the supplied clinical context.",
    )
    first = _make_response("not valid JSON")
    if repair == "refusal":
        first.model_dump.return_value = {
            "content_filters": [
                {
                    "blocked": True,
                    "source_type": "completion",
                    "content_filter_results": {},
                }
            ]
        }
    responses = iter([first, _make_response('{"message":"ok"}')])
    requests: list[dict[str, Any]] = []

    # Capture each request before the next repair can extend the input list.
    async def create(**kwargs: Any) -> OpenAIResponse:
        requests.append(deepcopy(kwargs))
        return next(responses)

    cast(Any, client)._openai_client.responses.create = create
    messages: list[MessageDict] = [{"role": "user", "content": "hello"}]
    response = asyncio.run(
        client.get_response(
            messages,
            schema=StructuredResponse if repair == "schema" else None,
            extra_call_args={"max_tokens": 100, "reasoning": {"effort": "low"}},
        )
    )

    assert response.choices[0].message.content == '{"message":"ok"}'
    assert len(requests) == 2
    assert len(requests[0]["input"]) == 1
    assert requests[1]["input"][:-1] == requests[0]["input"]
    feedback = requests[1]["input"][-1]
    assert feedback["role"] == "user"
    assert feedback["content"][0]["type"] == "input_text"
    if repair == "schema":
        assert "valid JSON" in feedback["content"][0]["text"]
    else:
        assert feedback["content"][0]["text"] == "Use the supplied clinical context."

    assert {key: value for key, value in requests[0].items() if key != "input"} == {
        key: value for key, value in requests[1].items() if key != "input"
    }
    assert requests[1]["max_output_tokens"] == 100
    assert requests[1]["reasoning"] == {"effort": "low"}
    assert messages == [{"role": "user", "content": "hello"}]


def test_responses_client_preserves_non_strict_external_tool_schema() -> None:
    """External tool schemas keep optional and open nested objects unchanged."""
    schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "options": {"type": "object", "additionalProperties": {"type": "string"}},
        },
        "required": ["text"],
    }
    original = deepcopy(schema)
    tool: NativeTool[EchoArgs, str] = NativeTool(
        name="remote__echo",
        description="Remote echo",
        args_model=EchoArgs,
        handler=_echo_handler,
        parameters_schema=schema,
        strict=False,
        retry_on_timeout=False,
    )
    client = ResponsesClient(Config(api_key="test-key", model="gpt-4o"))
    create_mock = AsyncMock(return_value=_make_response("done"))
    cast(Any, client)._openai_client.responses.create = create_mock

    asyncio.run(
        client.get_response(
            messages=[{"role": "user", "content": "echo"}],
            tools=Tools(tools=(tool,)),
            extra_call_args={
                "tools": [{"type": "web_search", "search_context_size": "low"}],
                "tool_choice": {"type": "web_search"},
            },
        )
    )

    kwargs = cast(Any, create_mock.await_args).kwargs

    assert kwargs["tools"][0] == {"type": "web_search", "search_context_size": "low"}
    assert kwargs["tools"][1]["strict"] is False
    assert kwargs["tools"][1]["parameters"] == original == schema
    assert kwargs["tool_choice"] == {"type": "web_search"}


@pytest.mark.parametrize("has_handler", [False, True])
def test_native_function_declarations_cannot_override_prepared_tools(
    has_handler: bool,
) -> None:
    tool: NativeTool[EchoArgs, str] = NativeTool(
        name="echo", description="Echo", args_model=EchoArgs, handler=_echo_handler
    )
    client = ResponsesClient(
        Config(api_key="test-key", model="test"),
        tools=[{"type": "function", "name": "echo", "parameters": {}}],
    )
    create = AsyncMock(return_value=_make_response("done"))
    cast(Any, client)._openai_client.responses.create = create

    async def run() -> None:
        runner = Runner()
        agent = Agent(
            SingleThreadedRuntimeEngine(),
            runner,
            descriptor=AgentDescriptor(
                name="prepared",
                model=client,
                tools=Tools(
                    tools=[tool], tool_call_limits=None if has_handler else {"echo": 0}
                ),
            ),
        )
        await runner.run(
            agent, RunState(instructions=None, history=["hello"], responses=[])
        )

    # Reject native function declarations even when the harness exhausted that tool.
    with pytest.raises(ValueError, match="native tool"):
        asyncio.run(run())

    create.assert_not_awaited()


def test_native_artifact_is_preserved_and_custom_execution_fails_clearly() -> None:
    native = OpenAIResponse.model_construct(
        id="resp_artifact",
        created_at=1,
        model="test",
        object="response",
        status="completed",
        error=None,
        usage=None,
        output=[
            ImageGenerationCall(
                id="image_1",
                type="image_generation_call",
                status="completed",
                result="encoded-image",
            )
        ],
    )
    client = ResponsesClient(Config(api_key="test-key", model="test"))
    create = AsyncMock(return_value=native)
    cast(Any, client)._openai_client.responses.create = create
    response = asyncio.run(client.get_response([{"role": "user", "content": "draw"}]))

    assert response.choices[0].message.content is None
    assert response.model_extra is not None
    assert (
        response.model_extra["openai_response"]["output"][0]["result"]
        == "encoded-image"
    )

    # Hosted output is replayable, but an unsupported local call must fail clearly.
    native.output = [
        ResponseCustomToolCall(
            type="custom_tool_call", call_id="call_custom", name="raw", input="unparsed"
        )
    ]
    with pytest.raises(ResponseError, match="Unsupported.*custom_tool_call") as caught:
        asyncio.run(client.get_response([{"role": "user", "content": "execute"}]))

    assert caught.value.response is native


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
    stream = _FakeAsyncStream(
        [
            ResponseCompletedEvent.model_construct(
                type="response.completed", response=response
            )
        ]
    )
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

    create = AsyncMock(return_value=_make_response("answer"))
    cast(Any, client)._openai_client.responses.create = create

    asyncio.run(
        client.get_response(
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
    )
    assert create.await_args is not None
    result = create.await_args.kwargs["input"]

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

    create = AsyncMock(return_value=_make_response("answer"))
    cast(Any, client)._openai_client.responses.create = create

    asyncio.run(client.get_response(messages))
    assert create.await_args is not None
    result = create.await_args.kwargs["input"]

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
            ResponseTextDeltaEvent.model_construct(
                type="response.output_text.delta",
                delta="Hel",
                output_index=0,
            ),
            ResponseReasoningTextDeltaEvent.model_construct(
                type="response.reasoning_text.delta",
                delta="thinking",
                output_index=0,
            ),
            ResponseOutputItemAddedEvent.model_construct(
                type="response.output_item.added",
                output_index=1,
                item=ResponseReasoningItem.model_construct(
                    type="reasoning", id="rs_1", summary=[]
                ),
            ),
            ResponseCompletedEvent.model_construct(
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


def test_http_stream_releases_iterators_before_event_loop_shutdown() -> None:
    """Completed streams must not leave SDK transport generators for loop shutdown."""
    errors: list[dict[str, Any]] = []
    terminal: dict[str, Any] = {
        "type": "response.completed",
        "sequence_number": 1,
        "response": {
            "id": "resp_cleanup",
            "object": "response",
            "created_at": 1,
            "model": "test",
            "status": "completed",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
    }
    event_bytes = ("data: " + json.dumps(terminal) + "\n\n").encode()
    done_bytes = b"data: [DONE]\n\n"

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            await reader.readexactly(length)

            # Keep the SSE terminator separate so a terminal-only consumer stops early.
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: "
                + str(len(event_bytes) + len(done_bytes)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + event_bytes
            )
            await writer.drain()
            await asyncio.sleep(0.01)
            writer.write(done_bytes)
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def run() -> None:
        asyncio.get_running_loop().set_exception_handler(
            lambda _loop, context: errors.append(context)
        )
        async with await asyncio.start_server(serve, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            client = ResponsesClient(
                Config(
                    api_key="test", model="test", base_url=f"http://127.0.0.1:{port}"
                )
            )
            try:
                for _ in range(10):
                    events = [
                        event
                        async for event in client.stream_response(
                            [{"role": "user", "content": "hello"}]
                        )
                    ]
                    assert len(events) == 1
                    assert events[0].response is not None
                    assert events[0].response.id == "resp_cleanup"
            finally:
                await cast(Any, client)._openai_client.close()

            # Exercise finalization while the owning loop can still report failures.
            gc.collect()
            await asyncio.sleep(0.1)

    asyncio.run(run())

    assert errors == []


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
            ResponseTextDeltaEvent.model_construct(
                type="response.output_text.delta",
                delta="Hel",
                output_index=0,
            ),
            ResponseCompletedEvent.model_construct(
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
    assert generation_spans[0].span_data.usage["total_tokens"] == 15


def _assert_cold_reasoning_serialization(
    mode: str, provider_name: str, tracing_name: str
) -> None:
    """Check first-use serialization without warming the SDK schema in pytest."""
    assert not ResponseReasoningItem.__pydantic_complete__
    tracing_mode = ModelTracing[tracing_name]
    client = ResponsesClient(
        Config(
            api_key="test-key",
            model=f"{provider_name}/gpt-4o",
            base_url=(
                "https://example.openai.azure.com/"
                if provider_name == "azure"
                else None
            ),
            tracing=tracing_mode,
        )
    )
    original_provider = get_trace_provider()
    provider = DefaultTraceProvider()
    processor = _CollectingTracingProcessor()
    provider.register_processor(processor)
    set_trace_provider(provider)

    try:
        # Parse as the SDK does. Direct model validation would hide the defect.
        # Later iterations also check empty summaries and reuse of the built schema.
        for index, summary in enumerate(
            [[{"type": "summary_text", "text": "Check the synthetic input."}], [], []]
        ):
            payload = {
                "id": "rs_test",
                "type": "reasoning",
                "summary": summary,
                "content": [{"type": "reasoning_text", "text": "Synthetic reasoning."}],
                "encrypted_content": "synthetic-encrypted-content",
                "status": "completed",
            }
            reasoning = TypeAdapter(ResponseReasoningItem).validate_python(payload)
            if index == 0:
                assert not ResponseReasoningItem.__pydantic_complete__

            tool_response = _make_tool_call_response()
            response = OpenAIResponse.model_construct(
                id="resp_reasoning",
                created_at=1234567890.0,
                model="gpt-4o",
                object="response",
                status="completed",
                output=[reasoning, *tool_response.output],
                usage=tool_response.usage,
            )
            stream = _FakeAsyncStream(
                [
                    ResponseCompletedEvent.model_construct(
                        type="response.completed", response=response
                    )
                ]
            )
            cast(Any, client)._openai_client.responses.create = AsyncMock(
                return_value=stream if mode == "stream" else response
            )

            with trace("cold-reasoning-test"):
                if mode == "stream":
                    events = asyncio.run(_collect_stream_events(client))
                    assert [event.kind for event in events] == [
                        ModelStreamEventKind.COMPLETED
                    ]
                    assert stream.closed
                    result = events[0].response
                    assert result is not None
                else:
                    result = asyncio.run(
                        client.get_response(
                            messages=[{"role": "user", "content": "Run the tool."}]
                        )
                    )

            assert isinstance(
                cast(Any, result).reasoning_content, ResponseReasoningItem
            )
            assert cast(Any, result).reasoning_content is reasoning
            assert result.choices[0].message.content is None
            assert result.choices[0].finish_reason == "tool_calls"
            assert result.model_dump(mode="json")["reasoning_content"] == payload
            assert json.loads(result.model_dump_json())["reasoning_content"] == payload

            event = ModelStreamEvent(
                kind=ModelStreamEventKind.COMPLETED, response=result
            )
            assert event.to_trace_dict()["response"]["reasoning_content"] == payload

            if tracing_mode == ModelTracing.ENABLED:
                spans = [
                    span
                    for span in processor.spans
                    if span.span_data.type == "generation"
                ]
                assert len(spans) == index + 1
                assert spans[-1].span_data.output[0]["output"][0] == payload
                if mode == "stream":
                    assert spans[-1].span_data.events == [
                        {
                            "kind": "completed",
                            "provider_event_type": "response.completed",
                            "response": result.model_dump(mode="json"),
                        }
                    ]
    finally:
        set_trace_provider(original_provider)


@pytest.mark.parametrize("mode", ["response", "stream"])
@pytest.mark.parametrize("provider_name", ["openai", "azure"])
@pytest.mark.parametrize("tracing_name", ["ENABLED", "DISABLED"])
def test_responses_reasoning_serializes_in_fresh_process(
    mode: str, provider_name: str, tracing_name: str
) -> None:
    """A cold SDK reasoning item must survive conversion and payload tracing."""
    # A separate interpreter prevents unrelated tests from initializing SDK models.
    result = subprocess.run(  # noqa: S603 - fixed interpreter and local test inputs
        [
            sys.executable,
            "-c",
            "import runpy, sys; "
            "runpy.run_path(sys.argv[1])[sys.argv[2]](*sys.argv[3:])",
            str(Path(__file__).resolve()),
            _assert_cold_reasoning_serialization.__name__,
            mode,
            provider_name,
            tracing_name,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_responses_resume_keeps_original_call_and_one_result(
    streaming: bool,
) -> None:
    root = AgentId.from_values("test", "root")
    native_output: list[dict[str, Any]] = [
        {
            "id": "rs_original",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "saved-reasoning",
        },
        {
            "id": "fc_original",
            "type": "function_call",
            "call_id": "call_original",
            "name": "external",
            "arguments": '{"text":"Choose a location."}',
            "status": "completed",
        },
    ]
    tools = Tools(
        tools=[
            ExternalTool(
                name="external", description="Ask the host.", args_model=EchoArgs
            )
        ]
    )
    saved = None
    for output in (native_output, []):
        native = OpenAIResponse.model_validate(
            {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "test",
                "status": "completed",
                "output": output,
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        transport = _FakeAsyncStream(
            [
                ResponseCompletedEvent(
                    type="response.completed", response=native, sequence_number=1
                )
            ]
        )
        create = AsyncMock(return_value=transport if streaming else native)
        client = ResponsesClient(Config(api_key="test-key", model="test"))
        cast(Any, client)._openai_client.responses.create = create
        runner = Runner()
        agent = Agent(
            SingleThreadedRuntimeEngine(),
            runner,
            bind_id=root,
            descriptor=AgentDescriptor(name="test", model=client, tools=tools),
        )

        state = (
            ToolResultInput(continuation=saved, result="Prague")
            if saved
            else RunState(instructions=None, history=["Start."], responses=[])
        )
        try:
            # Each iteration has a new runner, model client, and runtime.
            stream = runner.run_events(agent, state) if streaming else None
            try:
                operation = stream.result() if stream else runner.run(agent, state)
                if saved is None:
                    with pytest.raises(RunPaused) as paused:
                        await operation
                    saved = RunContinuation.from_json(
                        json.loads(json.dumps(paused.value.continuation.to_json()))
                    )
                else:
                    await operation
            finally:
                if stream:
                    await stream.aclose()
                    assert transport.closed
            create.assert_awaited_once()
            assert create.await_args is not None
            if not output:
                request = create.await_args.kwargs
                assert request["input"] == [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Start."}],
                    },
                    *native_output,
                    {
                        "type": "function_call_output",
                        "call_id": "call_original",
                        "output": "Prague",
                    },
                ]
                assert "previous_response_id" not in request
        finally:
            await cast(Any, client)._openai_client.close()
