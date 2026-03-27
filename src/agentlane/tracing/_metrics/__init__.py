"""Custom metrics system for ``agentlane.tracing``.

Provides span-level metric emission and trace-level aggregation.

Example:
    >>> from agentlane.tracing import MetricsRegistry, agent_span, emit_metric, trace
    >>>
    >>> # Configure registry
    >>> registry = MetricsRegistry()
    >>> registry.register("search_count", "sum")
    >>> registry.register("avg_relevance", "avg")
    >>>
    >>> # Use in code
    >>> with trace("my_workflow"):
    ...     with agent_span("search_agent"):
    ...         emit_metric("search_count", 1)
    ...         emit_metric("avg_relevance", 0.85)
"""

from ._aggregators import (
    AvgAggregator,
    CountAggregator,
    FirstAggregator,
    LastAggregator,
    MaxAggregator,
    MetricAggregator,
    MinAggregator,
    SumAggregator,
    add_aggregator,
    get_aggregator,
    get_aggregators,
)
from ._collector import (
    MetricsCollector,
    clear_all_collectors,
    get_collector,
    peek_collector,
    remove_collector,
)
from ._emit import emit_metric, emit_metrics
from ._registry import (
    MetricsRegistry,
    get_metrics_registry,
    reset_metrics_registry,
    set_metrics_registry,
)
from ._types import AggregatedMetric, AggregationType, MetricRecord, MetricValue

__all__ = [
    # Types
    "MetricValue",
    "MetricRecord",
    "AggregatedMetric",
    "AggregationType",
    # Aggregators
    "MetricAggregator",
    "SumAggregator",
    "CountAggregator",
    "AvgAggregator",
    "MinAggregator",
    "MaxAggregator",
    "FirstAggregator",
    "LastAggregator",
    "get_aggregators",
    "get_aggregator",
    "add_aggregator",
    # Registry
    "MetricsRegistry",
    "get_metrics_registry",
    "set_metrics_registry",
    "reset_metrics_registry",
    # Collector
    "MetricsCollector",
    "get_collector",
    "peek_collector",
    "remove_collector",
    "clear_all_collectors",
    # Emission API
    "emit_metric",
    "emit_metrics",
]
