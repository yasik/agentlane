"""Tests for the tracing metrics system."""

from typing import Any
from unittest.mock import MagicMock

from agentlane_braintrust import BraintrustProcessor

from agentlane.tracing import (
    AgentSpanData,
    AggregatedMetric,
    AvgAggregator,
    CountAggregator,
    DefaultTraceProvider,
    FirstAggregator,
    LastAggregator,
    MaxAggregator,
    MetricRecord,
    MetricsCollector,
    MetricsProcessor,
    MetricsRegistry,
    MinAggregator,
    SumAggregator,
    agent_span,
    emit_metric,
    emit_metrics,
    get_aggregator,
    get_collector,
    get_metrics_registry,
    remove_collector,
    set_metrics_registry,
    set_trace_provider,
    trace,
)
from agentlane.tracing._span import NoOpSpan


def test_sum_aggregator_returns_sum() -> None:
    """The sum aggregator should sum all metric values."""
    aggregator = SumAggregator()
    result = aggregator.aggregate("test", [1, 2, 3, 4, 5])

    assert result.name == "test"
    assert result.aggregation == "sum"
    assert result.value == 15
    assert result.count == 5


def test_count_aggregator_counts_emissions() -> None:
    """The count aggregator should count metric emissions."""
    aggregator = CountAggregator()
    result = aggregator.aggregate("test", [1, 2, 3])

    assert result.name == "test"
    assert result.aggregation == "count"
    assert result.value == 3
    assert result.count == 3


def test_avg_aggregator_calculates_average() -> None:
    """The avg aggregator should calculate the arithmetic mean."""
    aggregator = AvgAggregator()
    result = aggregator.aggregate("test", [10, 20, 30])

    assert result.name == "test"
    assert result.aggregation == "avg"
    assert result.value == 20.0
    assert result.count == 3


def test_avg_aggregator_handles_empty_values() -> None:
    """The avg aggregator should return zero for empty inputs."""
    aggregator = AvgAggregator()
    result = aggregator.aggregate("test", [])

    assert result.value == 0.0
    assert result.count == 0


def test_min_aggregator_returns_minimum() -> None:
    """The min aggregator should return the minimum value."""
    aggregator = MinAggregator()
    result = aggregator.aggregate("test", [5, 1, 9, 3])

    assert result.value == 1
    assert result.aggregation == "min"


def test_max_aggregator_returns_maximum() -> None:
    """The max aggregator should return the maximum value."""
    aggregator = MaxAggregator()
    result = aggregator.aggregate("test", [5, 1, 9, 3])

    assert result.value == 9
    assert result.aggregation == "max"


def test_first_aggregator_returns_first_value() -> None:
    """The first aggregator should return the first value."""
    aggregator = FirstAggregator()
    result = aggregator.aggregate("test", [10, 20, 30])

    assert result.value == 10
    assert result.aggregation == "first"


def test_last_aggregator_returns_last_value() -> None:
    """The last aggregator should return the last value."""
    aggregator = LastAggregator()
    result = aggregator.aggregate("test", [10, 20, 30])

    assert result.value == 30
    assert result.aggregation == "last"


def test_get_aggregator_returns_registered_aggregator() -> None:
    """Built-in aggregators should be retrievable by name."""
    assert isinstance(get_aggregator("sum"), SumAggregator)
    assert isinstance(get_aggregator("avg"), AvgAggregator)


def test_metrics_registry_supports_registration_and_export() -> None:
    """The metrics registry should store per-metric aggregation rules."""
    registry = MetricsRegistry(default_aggregation="sum")
    registry.register("metric_a", "avg")
    registry.register_many({"metric_b": "count", "metric_c": "max"})

    assert registry.get_aggregator("metric_a").aggregation_type == "avg"
    assert registry.get_aggregator("metric_b").aggregation_type == "count"
    assert registry.get_aggregator("metric_c").aggregation_type == "max"
    assert registry.is_registered("metric_a")
    assert not registry.is_registered("metric_d")
    assert registry.export() == {
        "default_aggregation": "sum",
        "registered_metrics": {
            "metric_a": "avg",
            "metric_b": "count",
            "metric_c": "max",
        },
    }


def test_global_metrics_registry_can_be_overridden() -> None:
    """The global metrics registry should be replaceable for tests and apps."""
    registry = MetricsRegistry(default_aggregation="count")
    set_metrics_registry(registry)

    assert get_metrics_registry() is registry


def test_metrics_collector_records_and_aggregates() -> None:
    """A collector should record raw metrics and aggregate them by registry rules."""
    registry = MetricsRegistry()
    registry.register("sum_metric", "sum")
    registry.register("avg_metric", "avg")
    collector = MetricsCollector(trace_id="test_trace", registry=registry)

    for index in range(5):
        collector.record(
            MetricRecord(
                name="sum_metric",
                value=10,
                span_id=f"span_{index}",
                trace_id="test_trace",
                timestamp="2024-01-01T00:00:00Z",
            )
        )
        collector.record(
            MetricRecord(
                name="avg_metric",
                value=index * 2,
                span_id=f"span_{index}",
                trace_id="test_trace",
                timestamp="2024-01-01T00:00:00Z",
            )
        )

    records = collector.get_records()
    assert len(records) == 10
    assert collector.get_records_for_span("span_2")[0].name == "sum_metric"

    aggregated = collector.aggregate()
    assert aggregated["sum_metric"].value == 50
    assert aggregated["avg_metric"].value == 4.0
    assert collector.aggregate_as_dict() == {"sum_metric": 50, "avg_metric": 4.0}


def test_global_collector_functions_manage_instances() -> None:
    """Global collector helpers should reuse and remove collectors by trace id."""
    collector_a = get_collector("trace_1")
    collector_b = get_collector("trace_1")
    collector_c = get_collector("trace_2")

    assert collector_a is collector_b
    assert collector_c is not collector_a
    assert remove_collector("trace_1") is collector_a
    assert remove_collector("trace_1") is None


def test_emit_metric_without_active_span_is_dropped() -> None:
    """Emitting a metric without an active span should be a no-op."""
    emit_metric("test_metric", 42)


def test_emit_metric_records_metric_for_current_span() -> None:
    """Metrics emitted inside a span should be recorded against the current trace."""
    provider = DefaultTraceProvider()
    set_trace_provider(provider)

    with trace("test_trace") as current_trace:
        with agent_span("test_agent"):
            emit_metric("test_metric", 42)

        collector = get_collector(current_trace.trace_id)
        records = collector.get_records()
        assert len(records) == 1
        assert records[0].name == "test_metric"
        assert records[0].value == 42


def test_emit_metrics_records_multiple_values() -> None:
    """Bulk metric emission should record each metric entry."""
    provider = DefaultTraceProvider()
    set_trace_provider(provider)

    with trace("test_trace") as current_trace:
        with agent_span("test_agent"):
            emit_metrics({"metric_a": 1, "metric_b": 2, "metric_c": 3})

        records = get_collector(current_trace.trace_id).get_records()
        assert len(records) == 3
        assert {record.name for record in records} == {
            "metric_a",
            "metric_b",
            "metric_c",
        }


def test_metrics_processor_aggregates_on_trace_end() -> None:
    """The metrics processor should compute trace-level aggregates on completion."""
    aggregated_metrics: dict[str, dict[str, AggregatedMetric]] = {}

    def on_metrics(trace_id: str, metrics: dict[str, AggregatedMetric]) -> None:
        aggregated_metrics[trace_id] = metrics

    processor = MetricsProcessor(on_trace_metrics=on_metrics)
    provider = DefaultTraceProvider()
    provider.register_processor(processor)
    set_trace_provider(provider)

    with trace("test_trace") as current_trace:
        with agent_span("test_agent"):
            emit_metric("counter", 1)
            emit_metric("counter", 1)
            emit_metric("counter", 1)

    assert current_trace.trace_id in aggregated_metrics
    assert aggregated_metrics[current_trace.trace_id]["counter"].value == 3


def test_metrics_processor_uses_registry_for_aggregation_types() -> None:
    """The processor should respect registered aggregation strategies."""
    aggregated_metrics: dict[str, dict[str, AggregatedMetric]] = {}

    def on_metrics(trace_id: str, metrics: dict[str, AggregatedMetric]) -> None:
        aggregated_metrics[trace_id] = metrics

    registry = MetricsRegistry()
    registry.register("score", "avg")
    registry.register("events", "count")
    set_metrics_registry(registry)

    processor = MetricsProcessor(on_trace_metrics=on_metrics)
    provider = DefaultTraceProvider()
    provider.register_processor(processor)
    set_trace_provider(provider)

    with trace("test_trace") as current_trace:
        with agent_span("test_agent"):
            emit_metric("score", 0.8)
            emit_metric("score", 0.9)
            emit_metric("score", 1.0)
            emit_metric("events", 1)
            emit_metric("events", 1)

    trace_metrics = aggregated_metrics[current_trace.trace_id]
    assert abs(trace_metrics["score"].value - 0.9) < 1e-9
    assert trace_metrics["score"].aggregation == "avg"
    assert trace_metrics["events"].value == 2
    assert trace_metrics["events"].aggregation == "count"


def test_metrics_processor_stores_trace_metrics_for_later_lookup() -> None:
    """The metrics processor should retain computed aggregates for later inspection."""
    processor = MetricsProcessor()
    provider = DefaultTraceProvider()
    provider.register_processor(processor)
    set_trace_provider(provider)

    with trace("test_trace") as current_trace:
        with agent_span("test_agent"):
            emit_metric("test", 42)

    metrics = processor.get_trace_metrics(current_trace.trace_id)
    assert metrics is not None
    assert metrics["test"].value == 42


def test_aggregated_metric_to_dict_omits_raw_values() -> None:
    """Serialized aggregate metrics should keep only public output fields."""
    metric = AggregatedMetric(
        name="test",
        aggregation="sum",
        value=100,
        count=5,
        raw_values=[10, 20, 30, 40],
    )

    assert metric.to_dict() == {
        "name": "test",
        "aggregation": "sum",
        "value": 100,
        "count": 5,
    }


def test_braintrust_processor_attaches_span_metrics_to_logged_span() -> None:
    """Span-level custom metrics should be attached to the span event."""
    mock_span: Any = MagicMock()
    mock_logger: Any = MagicMock()
    mock_logger.start_span.return_value = mock_span
    mock_span.start_span.return_value = mock_span

    processor = BraintrustProcessor(logger=mock_logger)
    provider = DefaultTraceProvider()
    provider.register_processor(processor)
    set_trace_provider(provider)

    with trace("test_trace"):
        with agent_span("test_agent"):
            emit_metric("custom_score", 0.95)
            emit_metric("search_count", 5)

    found_metrics = False
    for call in mock_span.log.call_args_list:
        metrics = call.kwargs.get("metrics")
        if metrics is None:
            continue
        if "custom_score" in metrics and "search_count" in metrics:
            found_metrics = True
            assert metrics["custom_score"] == 0.95
            assert metrics["search_count"] == 5
            break

    assert found_metrics


def test_braintrust_processor_ignores_metrics_for_noop_span() -> None:
    """Adding metrics for a no-op span should be a safe no-op."""
    processor = BraintrustProcessor(logger=MagicMock())
    noop_span = NoOpSpan(AgentSpanData(name="test"))
    event: dict[str, Any] = {}

    processor_any: Any = processor
    processor_any._add_span_metrics(event, noop_span)

    assert "metrics" not in event
