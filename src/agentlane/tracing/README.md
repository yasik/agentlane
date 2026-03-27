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
    custom_span
    function_span,
    generation_span,
)

with trace("example_workflow"):
    # Agent span - for agent-based operations
    with agent_span(name="planner", tools=["search", "calculate"]):
        pass

        # Function span - for function calls
        with function_span(name="process_data", inputs="raw_data"):
            pass

    # Generation span - for LLM generations
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

### How It Works

1. **Emission**: Call `emit_metric()` from within a span context
2. **Collection**: Metrics are collected per-trace automatically
3. **Aggregation**: When a trace ends, metrics are aggregated using the registry's strategies
4. **Export**: Aggregated metrics are passed to processors (e.g., Braintrust)
