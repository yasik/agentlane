# pylint: disable=E0402

"""Collector for accumulating span metrics within a trace."""

import threading
from collections import defaultdict

from ._registry import MetricsRegistry, get_metrics_registry
from ._types import AggregatedMetric, MetricRecord, MetricValue


class MetricsCollector:
    """Collects and aggregates metrics for a single trace.

    Thread-safe collector that accumulates MetricRecords from spans
    and computes trace-level aggregations on demand.

    Example:
        >>> collector = MetricsCollector(trace_id="trace_123")
        >>> collector.record(MetricRecord(
        ...     name="search_count",
        ...     value=5,
        ...     span_id="span_1",
        ...     trace_id="trace_123",
        ...     timestamp="2024-01-01T00:00:00Z",
        ... ))
        >>> aggregated = collector.aggregate()
    """

    def __init__(
        self,
        trace_id: str,
        registry: MetricsRegistry | None = None,
    ) -> None:
        """Initialize the collector.

        Args:
            trace_id: The trace ID this collector belongs to.
            registry: Optional custom registry (uses global if not provided).
        """
        self._trace_id = trace_id
        self._registry = registry or get_metrics_registry()
        self._records: list[MetricRecord] = []
        self._lock = threading.Lock()

    @property
    def trace_id(self) -> str:
        """Get the trace ID."""
        return self._trace_id

    def record(self, metric: MetricRecord) -> None:
        """Record a metric emission.

        Args:
            metric: The metric record to add.
        """
        with self._lock:
            self._records.append(metric)

    def get_records(self) -> list[MetricRecord]:
        """Get all recorded metrics (copy).

        Returns:
            List of all metric records.
        """
        with self._lock:
            return list(self._records)

    def get_records_for_span(self, span_id: str) -> list[MetricRecord]:
        """Get metrics for a specific span.

        Args:
            span_id: The span ID to filter by.

        Returns:
            List of metrics from that span.
        """
        with self._lock:
            return [r for r in self._records if r.span_id == span_id]

    def aggregate(self) -> dict[str, AggregatedMetric]:
        """Compute trace-level aggregations for all metrics.

        Returns:
            Dictionary mapping metric names to aggregated results.
        """
        with self._lock:
            # Group values by metric name
            grouped: dict[str, list[MetricValue]] = defaultdict(list)
            for record in self._records:
                grouped[record.name].append(record.value)

            # Aggregate each metric
            results: dict[str, AggregatedMetric] = {}
            for name, values in grouped.items():
                aggregator = self._registry.get_aggregator(name)
                results[name] = aggregator.aggregate(name, values)

            return results

    def aggregate_as_dict(self) -> dict[str, MetricValue]:
        """Compute aggregations as a simple name->value dict.

        Useful for direct integration with Braintrust metrics.

        Returns:
            Dictionary mapping metric names to aggregated values.
        """
        aggregated = self.aggregate()
        return {name: agg.value for name, agg in aggregated.items()}

    def clear(self) -> None:
        """Clear all recorded metrics."""
        with self._lock:
            self._records.clear()


_collectors: dict[str, MetricsCollector] = {}
_collectors_lock = threading.Lock()


def get_collector(trace_id: str) -> MetricsCollector:
    """Get or create a collector for a trace.

    Args:
        trace_id: The trace ID.

    Returns:
        The MetricsCollector for this trace.
    """
    with _collectors_lock:
        if trace_id not in _collectors:
            _collectors[trace_id] = MetricsCollector(trace_id)
        return _collectors[trace_id]


def peek_collector(trace_id: str) -> MetricsCollector | None:
    """Get a collector for a trace without creating one.

    Unlike get_collector(), this does not create a new collector if
    one doesn't exist. Use this when you want to check for metrics
    without side effects.

    Args:
        trace_id: The trace ID.

    Returns:
        The MetricsCollector if it exists, None otherwise.
    """
    with _collectors_lock:
        return _collectors.get(trace_id)


def remove_collector(trace_id: str) -> MetricsCollector | None:
    """Remove and return a collector for a trace.

    Args:
        trace_id: The trace ID.

    Returns:
        The removed collector or None if not found.
    """
    with _collectors_lock:
        return _collectors.pop(trace_id, None)


def clear_all_collectors() -> None:
    """Clear all collectors.

    Useful for testing to ensure clean state between tests.
    """
    with _collectors_lock:
        _collectors.clear()
