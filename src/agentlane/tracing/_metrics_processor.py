"""Processor that collects span metrics and integrates with trace lifecycle."""

from collections.abc import Callable
from typing import Any

import structlog

from ._metrics import AggregatedMetric, peek_collector
from ._processor_interface import TracingProcessor

LOGGER = structlog.get_logger(log_tag="tracing.metrics_processor")

MetricsCallback = Callable[[str, dict[str, AggregatedMetric]], None]
"""Type alias for metrics callback: (trace_id, metrics_dict) -> None."""


class MetricsProcessor(TracingProcessor):
    """Processor that handles metric collection and trace-level aggregation.

    This processor:
    - Creates collectors when traces start (lazily on first emit)
    - Computes aggregations when traces end
    - Optionally invokes a callback with aggregated metrics

    The callback allows integration with other processors (e.g., Braintrust)
    without tight coupling.

    Example:
        >>> def on_metrics(trace_id: str, metrics: dict[str, AggregatedMetric]):
        ...     print(f"Trace {trace_id} metrics: {metrics}")
        >>>
        >>> processor = MetricsProcessor(on_trace_metrics=on_metrics)
        >>> provider.register_processor(processor)
    """

    def __init__(
        self,
        on_trace_metrics: MetricsCallback | None = None,
    ) -> None:
        """Initialize the metrics processor.

        Args:
            on_trace_metrics: Optional callback invoked with aggregated metrics
                when a trace ends. Receives (trace_id, metrics_dict).
        """
        self._on_trace_metrics = on_trace_metrics
        self._trace_metrics: dict[str, dict[str, AggregatedMetric]] = {}

    def on_trace_start(self, trace: Any) -> None:
        """Initialize collector for the trace.

        Collector is created lazily on first emit, no action needed here.
        """

    def on_trace_end(self, trace: Any) -> None:
        """Compute aggregations and invoke callback."""
        trace_id = trace.trace_id
        if trace_id == "no-op":
            return

        # Peek at collector (don't remove - cleanup happens in processor registry)
        collector = peek_collector(trace_id)
        if collector is None:
            return

        # Compute aggregations
        aggregated = collector.aggregate()

        if not aggregated:
            return

        # Store for retrieval
        self._trace_metrics[trace_id] = aggregated

        # Invoke callback if provided
        if self._on_trace_metrics is not None:
            try:
                self._on_trace_metrics(trace_id, aggregated)
            except Exception as e:
                LOGGER.warning(f"Error in metrics callback: {e}")

    def on_span_start(self, span: Any) -> None:
        """No action needed on span start."""

    def on_span_end(self, span: Any) -> None:
        """No action needed on span end (metrics already recorded via emit)."""

    def get_trace_metrics(
        self,
        trace_id: str,
    ) -> dict[str, AggregatedMetric] | None:
        """Get aggregated metrics for a completed trace.

        Args:
            trace_id: The trace ID.

        Returns:
            Aggregated metrics or None if not found.
        """
        return self._trace_metrics.get(trace_id)

    def clear_stored_metrics(self) -> None:
        """Clear all stored trace metrics.

        Useful for testing or when memory needs to be reclaimed.
        """
        self._trace_metrics.clear()

    def shutdown(self) -> None:
        """Clean up stored metrics."""
        self._trace_metrics.clear()

    def force_flush(self) -> None:
        """No buffering, nothing to flush."""
