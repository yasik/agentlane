"""Tests for the core tracing context API."""

from agentlane.tracing import (
    function_span,
    generation_span,
    get_current_span,
    get_current_trace,
    trace,
)


def test_trace_start_and_finish_bind_current_trace() -> None:
    """Starting and finishing a trace should manage current-trace scope."""
    current_trace = trace(workflow_name="workflow")

    assert get_current_trace() is None

    current_trace.start(mark_as_current=True)
    assert get_current_trace() is current_trace

    current_trace.finish(reset_current=True)
    assert get_current_trace() is None


def test_function_span_context_manager_sets_and_resets_current_span() -> None:
    """Span context managers should update the current span for the active scope."""
    assert get_current_span() is None

    with trace("workflow") as current_trace:
        with function_span(name="lookup", inputs="query=value") as span:
            assert get_current_trace() is current_trace
            assert get_current_span() is span
            assert span.trace_id == current_trace.trace_id
            assert span.span_data.name == "lookup"
            assert span.span_data.input == "query=value"

    assert get_current_span() is None
    assert get_current_trace() is None


def test_generation_span_preserves_explicit_parent_and_trace_ids() -> None:
    """Child spans should preserve the explicit parent relationship."""
    with trace("workflow") as current_trace:
        with function_span(name="parent") as parent:
            child = generation_span(model="gpt-4o", parent=parent)

    assert child.parent_id == parent.span_id
    assert child.trace_id == parent.trace_id
    assert child.trace_id == current_trace.trace_id
    assert child.span_data.model == "gpt-4o"
