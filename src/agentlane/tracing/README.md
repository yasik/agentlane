# agentlane.tracing

Core tracing primitives for AgentLane with span factories, context propagation,
and trace-level metrics.

## Quick Start

```python
from agentlane.tracing import generation_span, trace

# Create a trace for your workflow
with trace("my_workflow"):
    # Add spans for specific operations
    with generation_span(model="gpt-5.2", usage={"tokens": 100}):
        # Your LLM call here
        pass
```

## Span Types

```python
from agentlane.tracing import (
    agent_span,
    custom_span,
    function_span,
    generation_span,
    trace,
)

with trace("example_workflow"):
    # Agent span - for agent-based operations
    with agent_span(name="planner", tools=["search", "calculate"]):
        # Function span - child of the agent span; spans auto-parent to the
        # current span/trace, so no explicit parent is needed.
        with function_span(name="process_data", inputs="raw_data"):
            pass

    # Generation span - sibling of the agent span
    with generation_span(model="gpt-5.2", usage={"input_tokens": 50}):
        pass

    # Custom span - for anything else
    with custom_span(name="custom_operation", data={"key": "value"}):
        pass
```

## Accessing Current Context

```python
from agentlane.tracing import get_current_span, get_current_trace

current_trace = get_current_trace()  # Get active trace
current_span = get_current_span()    # Get active span
```

## Disabling Tracing

```python

# Disable individual traces
with trace("my_workflow", disabled=True):
    pass  # This trace won't be recorded
```

## Metrics

The metrics system provides span-level metric emission and trace-level aggregation.

### Basic Usage

```python
from agentlane.tracing import agent_span, emit_metric, emit_metrics, trace

with trace("my_workflow"):
    with agent_span("search_agent"):
        # Emit individual metrics
        emit_metric("items_processed", 42)
        emit_metric("success_rate", 0.95)

        # Or emit multiple metrics at once
        emit_metrics({
            "searches_executed": 5,
            "citations_found": 12,
            "avg_relevance": 0.85,
        })
```

### Configuring Aggregation

Metrics are aggregated at the trace level. Configure how metrics should be aggregated using the registry:

```python
from agentlane.tracing import get_metrics_registry

# Get the global registry
registry = get_metrics_registry()

# Register metrics with specific aggregation strategies
registry.register("search_count", "sum")      # Sum all values
registry.register("relevance_score", "avg")   # Average all values
registry.register("max_latency", "max")       # Keep maximum value

# Or register multiple at once
registry.register_many({
    "total_tokens": "sum",
    "avg_confidence": "avg",
    "error_count": "count",
})
```

### Aggregation Types

| Type    | Description                   |
| ------- | ----------------------------- |
| `sum`   | Sum all metric values         |
| `count` | Count the number of emissions |
| `avg`   | Calculate average of values   |
| `min`   | Return minimum value          |
| `max`   | Return maximum value          |
| `first` | Return first emitted value    |
| `last`  | Return last emitted value     |

The default aggregation for unregistered metrics is `sum`.

### Advanced: Custom Registry

For isolated metric configurations (e.g., in tests or separate domains):

```python
from agentlane.tracing import MetricsCollector, MetricsRegistry

# Create a custom registry with a different default
registry = MetricsRegistry(default_aggregation="avg")
registry.register("search_count", "count")

# Create a collector with the custom registry
collector = MetricsCollector(trace_id="trace_123", registry=registry)
```

### Custom Aggregators

The seven built-in aggregation types map to aggregator classes (`SumAggregator`,
`CountAggregator`, `AvgAggregator`, `MinAggregator`, `MaxAggregator`,
`FirstAggregator`, `LastAggregator`). To add your own strategy, implement the
`MetricAggregator` protocol and register it under a new aggregation type:

```python
from agentlane.tracing import (
    AggregatedMetric,
    MetricAggregator,
    add_aggregator,
    get_aggregator,
    get_aggregators,
)


class MedianAggregator:
    @property
    def aggregation_type(self) -> str:
        return "median"

    def aggregate(self, name, values):
        ordered = sorted(values)
        mid = ordered[len(ordered) // 2]
        return AggregatedMetric(
            name=name,
            aggregation="median",
            value=mid,
            count=len(values),
            raw_values=list(values),
        )


add_aggregator("median", MedianAggregator())
aggregator = get_aggregator("median")  # look up a single aggregator
all_aggregators = get_aggregators()    # the full type -> aggregator map
```

### Registry And Collector Lifecycle

The global registry and per-trace collectors can be swapped or cleared, which is
mainly useful for tests and isolated domains:

```python
from agentlane.tracing import (
    MetricsRegistry,
    clear_all_collectors,
    get_collector,
    peek_collector,
    remove_collector,
    reset_metrics_registry,
    set_metrics_registry,
)

# Replace the global registry, then reset it back to the default later
set_metrics_registry(MetricsRegistry(default_aggregation="avg"))
reset_metrics_registry()

# Collector helpers, keyed by trace_id
get_collector("trace_123")      # get or create a collector
peek_collector("trace_123")     # get without creating (None if absent)
remove_collector("trace_123")   # remove and return a collector
clear_all_collectors()          # drop every collector
```

### Span Errors

When a span exits with an exception, the failure is attached as a `SpanError`
(a `dict` subclass holding `message` and optional `data`). You can also set one
explicitly via `span.set_error(...)`:

```python
from agentlane.tracing import SpanError, custom_span

with custom_span(name="risky_step") as span:
    span.set_error(SpanError(message="validation failed", data={"code": 422}))
```

### Integrating With Other Processors

`MetricsProcessor` aggregates trace metrics and, when given an `on_trace_metrics`
callback, hands the aggregated result to that callback when a trace ends. The
callback receives `(trace_id, metrics)` where `metrics` is a
`dict[str, AggregatedMetric]`:

```python
from agentlane.tracing import AggregatedMetric, MetricsProcessor


def on_metrics(trace_id: str, metrics: dict[str, AggregatedMetric]) -> None:
    print(f"Trace {trace_id} metrics: {metrics}")


provider.register_processor(MetricsProcessor(on_trace_metrics=on_metrics))
```

### How It Works

1. **Emission**: Call `emit_metric()` from within a span context
2. **Collection**: Metrics are collected per-trace automatically
3. **Aggregation**: When a trace ends, metrics are aggregated using the registry's strategies
4. **Export**: Aggregated metrics are passed to processors (e.g., Braintrust)
