# pylint: disable=E0402

"""Public API for emitting metrics from spans."""

from typing import Any

import structlog

from .._scope import get_current_span
from .._setup import get_trace_provider
from .._span import Span
from ._collector import get_collector
from ._types import MetricRecord, MetricValue

LOGGER = structlog.get_logger(log_tag="tracing.metrics")


def emit_metric(
    name: str,
    value: MetricValue,
    span: Span[Any] | None = None,
) -> None:
    """Emit a metric from the current or specified span.

    This is the primary API for recording custom metrics. Metrics are
    collected at the span level and aggregated at the trace level when
    the trace ends.

    Args:
        name: The metric name (should be snake_case).
        value: The numeric value to record.
        span: Optional span to associate with (uses current span if not provided).

    Example:
        >>> from diadia_tracing import emit_metric
        >>>
        >>> # Inside a span context
        >>> with agent_span("my_agent"):
        ...     # Do work...
        ...     emit_metric("items_processed", 42)
        ...     emit_metric("success_rate", 0.95)

    Note:
        If no span or trace is active, the metric is silently dropped
        (with a debug log).
    """
    # Resolve span
    if span is None:
        span = get_current_span()

    if span is None:
        LOGGER.debug(f"No active span for metric '{name}', dropping")
        return

    # Get trace ID
    trace_id = span.trace_id
    if trace_id == "no-op":
        LOGGER.debug(f"No-op span for metric '{name}', dropping")
        return

    # Create record
    record = MetricRecord(
        name=name,
        value=value,
        span_id=span.span_id,
        trace_id=trace_id,
        timestamp=get_trace_provider().time_iso(),
    )

    # Record in collector
    collector = get_collector(trace_id)
    collector.record(record)


def emit_metrics(
    metrics: dict[str, MetricValue],
    span: Span[Any] | None = None,
) -> None:
    """Emit multiple metrics at once.

    Convenience function for recording multiple metrics in a single call.

    Args:
        metrics: Dictionary mapping metric names to values.
        span: Optional span to associate with.

    Example:
        >>> emit_metrics({
        ...     "searches_executed": 5,
        ...     "citations_found": 12,
        ...     "avg_relevance": 0.85,
        ... })
    """
    for name, value in metrics.items():
        emit_metric(name, value, span=span)
