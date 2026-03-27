"""Span and trace creation functions using global trace provider."""

from collections.abc import Mapping, Sequence
from typing import Any

import structlog

from ._setup import get_trace_provider
from ._span import Span
from ._span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
)
from ._trace import Trace

LOGGER = structlog.get_logger(log_tag="tracing.create")


def trace(
    workflow_name: str,
    trace_id: str | None = None,
    group_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    disabled: bool = False,
) -> Trace:
    """Create a new trace. The trace will not be started automatically; you should either use
    it as a context manager (`with trace(...):`) or call `trace.start()` + `trace.finish()`
    manually.

    In addition to the workflow name and optional grouping identifier, you can provide
    an arbitrary metadata dictionary to attach additional user-defined information to
    the trace.

    Args:
        workflow_name: The name of the logical app or workflow.
        trace_id: The ID of the trace. Optional. If not provided, we will generate an ID.
        group_id: Optional grouping identifier to link multiple traces.
        metadata: Optional dictionary of additional metadata to attach to the trace.
        disabled: If True, we will return a Trace but the Trace will not be recorded.

    Returns:
        The newly created trace object.
    """
    current_trace = get_trace_provider().get_current_trace()
    if current_trace:
        LOGGER.warning(
            "Trace already exists. Creating a new trace, but this is probably a mistake."
        )

    return get_trace_provider().create_trace(
        name=workflow_name,
        trace_id=trace_id,
        group_id=group_id,
        metadata=metadata,
        disabled=disabled,
    )


def get_current_trace() -> Trace | None:
    """Returns the currently active trace, if present."""
    return get_trace_provider().get_current_trace()


def get_current_span() -> Span[Any] | None:
    """Returns the currently active span, if present."""
    return get_trace_provider().get_current_span()


def agent_span(
    name: str,
    handoffs: list[str] | None = None,
    tools: list[str] | None = None,
    output_type: str | None = None,
    span_id: str | None = None,
    parent: Trace | Span[Any] | None = None,
    disabled: bool = False,
) -> Span[AgentSpanData]:
    """Create a new agent span. The span will not be started automatically, you should either do
    `with agent_span() ...` or call `span.start()` + `span.finish()` manually.

    Args:
        name: The name of the agent.
        handoffs: Optional list of agent names to which this agent could hand off control.
        tools: Optional list of tool names available to this agent.
        output_type: Optional name of the output type produced by the agent.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.

    Returns:
        The newly created agent span.
    """
    return get_trace_provider().create_span(
        span_data=AgentSpanData(
            name=name, handoffs=handoffs, tools=tools, output_type=output_type
        ),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )


def function_span(
    name: str,
    inputs: str | None = None,
    output: str | None = None,
    span_id: str | None = None,
    parent: Trace | Span[Any] | None = None,
    disabled: bool = False,
) -> Span[FunctionSpanData]:
    """Create a new function span. The span will not be started automatically, you should either do
    `with function_span() ...` or call `span.start()` + `span.finish()` manually.

    Args:
        name: The name of the function.
        input: The input to the function.
        output: The output of the function.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.

    Returns:
        The newly created function span.
    """
    return get_trace_provider().create_span(
        span_data=FunctionSpanData(name=name, input=inputs, output=output),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )


def generation_span(
    inputs: Sequence[Mapping[str, Any]] | None = None,
    output: Sequence[Mapping[str, Any]] | None = None,
    model: str | None = None,
    model_config: Mapping[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    span_id: str | None = None,
    parent: Trace | Span[Any] | None = None,
    disabled: bool = False,
) -> Span[GenerationSpanData]:
    """Create a new generation span. The span will not be started automatically, you should either
    do `with generation_span() ...` or call `span.start()` + `span.finish()` manually.

    This span captures the details of a model generation, including the
    input message sequence, any generated outputs, the model name and
    configuration, and usage data. If you only need to capture a model
    response identifier, use `response_span()` instead.

    Args:
        input: The sequence of input messages sent to the model.
        output: The sequence of output messages received from the model.
        model: The model identifier used for the generation.
        model_config: The model configuration (hyperparameters) used.
        usage: A dictionary of usage information (input tokens, output tokens, etc.).
        span_id: The ID of the span. Optional. If not provided, we will generate an ID.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.

    Returns:
        The newly created generation span.
    """
    return get_trace_provider().create_span(
        span_data=GenerationSpanData(
            input=inputs,
            output=output,
            model=model,
            model_config=model_config,
            usage=usage,
        ),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )


def custom_span(
    name: str,
    data: dict[str, Any] | None = None,
    span_id: str | None = None,
    parent: Trace | Span[Any] | None = None,
    disabled: bool = False,
) -> Span[CustomSpanData]:
    """Create a new custom span, to which you can add your own metadata. The span will not be
    started automatically, you should either do `with custom_span() ...` or call
    `span.start()` + `span.finish()` manually.

    Args:
        name: The name of the custom span.
        data: Arbitrary structured data to associate with the span.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.

    Returns:
        The newly created custom span.
    """
    return get_trace_provider().create_span(
        span_data=CustomSpanData(name=name, data=data or {}),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )
