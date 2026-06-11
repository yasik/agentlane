"""Tracing behavior for tool execution.

Tool-call ``function_span`` recording follows the per-call ``tracing`` mode
passed to ``ToolExecutor.execute``. The harness runner forwards the active
model's mode (``Model.tracing``) so tool spans trace consistently with the
model that triggered them.
"""

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import BaseModel

from agentlane.models import (
    Model,
    ModelTracing,
    Tool,
    ToolCall,
    ToolExecutionContext,
    ToolExecutor,
    Tools,
)
from agentlane.runtime import CancellationToken
from agentlane.tracing import (
    DefaultTraceProvider,
    Span,
    TracingProcessor,
    set_trace_provider,
    trace,
)


class _EchoArgs(BaseModel):
    """Arguments for the echo tool used in tracing tests."""

    text: str


class _EchoResult(BaseModel):
    """Structured return value for the echo tool."""

    echoed: str


async def _echo_handler(
    args: _EchoArgs,
    cancellation_token: CancellationToken,
    context: ToolExecutionContext,
) -> _EchoResult:
    """Return a structured echo result."""
    del cancellation_token, context
    return _EchoResult(echoed=args.text)


def _echo_tool() -> Tool[_EchoArgs, _EchoResult]:
    """Build the echo tool used across tracing tests."""
    return Tool(
        name="echo",
        description="Echo text",
        args_model=_EchoArgs,
        handler=_echo_handler,
    )


def _echo_call() -> ToolCall:
    """Build one echo tool call payload."""
    return ToolCall.model_validate(
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "echo", "arguments": '{"text": "hello"}'},
        }
    )


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


@pytest.fixture(name="collecting_processor")
def fixture_collecting_processor() -> Iterator[_CollectingTracingProcessor]:
    """Install a collecting processor and restore a clean provider afterward."""
    processor = _CollectingTracingProcessor()
    provider = DefaultTraceProvider()
    provider.register_processor(processor)
    set_trace_provider(provider)
    yield processor
    set_trace_provider(DefaultTraceProvider())


def _execute_echo(tracing: ModelTracing) -> None:
    """Execute one echo tool call inside an active trace with the given mode."""

    async def _run() -> None:
        with trace("tool_tracing_test"):
            await ToolExecutor().execute(
                tool_calls=[_echo_call()],
                tools=Tools(tools=[_echo_tool()]),
                tracing=tracing,
            )

    asyncio.run(_run())


def test_model_tracing_defaults_to_disabled() -> None:
    """The base ``Model`` exposes ``tracing``, defaulting to ``DISABLED``."""

    class _BareModel(Model[Any]):
        async def get_response(
            self,
            messages: list[dict[str, Any]],
            extra_call_args: dict[str, Any] | None = None,
            schema: Any = None,
            tools: Tools | None = None,
            cancellation_token: Any = None,
            parent_span: Span[Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            raise NotImplementedError

    assert _BareModel().tracing is ModelTracing.DISABLED


def test_execute_records_function_span_when_tracing_enabled(
    collecting_processor: _CollectingTracingProcessor,
) -> None:
    """A per-call ENABLED mode should record a function span for the tool call."""
    _execute_echo(ModelTracing.ENABLED)

    function_spans = [
        span for span in collecting_processor.spans if span.span_data.type == "function"
    ]
    assert len(function_spans) == 1
    assert function_spans[0].span_data.name == "echo"
    assert function_spans[0].span_data.input == '{"text": "hello"}'


def test_execute_records_no_function_span_when_tracing_disabled(
    collecting_processor: _CollectingTracingProcessor,
) -> None:
    """The default DISABLED mode should record no function span."""
    _execute_echo(ModelTracing.DISABLED)

    function_spans = [
        span for span in collecting_processor.spans if span.span_data.type == "function"
    ]
    assert function_spans == []
