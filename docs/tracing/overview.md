# Tracing Overview

`agentlane.tracing` provides the core tracing primitives used to instrument
runtime, model, and application code.

It gives AgentLane a shared trace and span model with context propagation and
trace-level metrics. Use this layer when you want structured observability
without coupling tracing logic to a specific runtime or provider.

## What It Includes

1. `trace(...)` for top-level trace scopes.
2. `agent_span(...)`, `function_span(...)`, `generation_span(...)`, and
   `custom_span(...)` for common span types.
3. context helpers such as `get_current_trace()`, `get_current_span()`, and
   parent-context propagation functions.
4. a metrics registry and aggregation helpers for trace-level metrics.
5. processor interfaces for exporting traces and metrics.

## Boundaries

1. This package defines tracing primitives and metrics aggregation.
2. It does not own runtime delivery, model calls, or harness orchestration.
3. Runtime and model layers can emit tracing data through this shared surface
   instead of each layer defining its own tracing contract.

## Minimal Example

```python
from agentlane.tracing import emit_metric, generation_span, trace


with trace("customer_support_run"):
    with generation_span(model="gpt-5.4-mini"):
        emit_metric("llm_calls", 1)
```

## Metrics

Metric emission happens inside an active span. Aggregation happens at the trace
level through the metrics registry.

Common aggregation modes are:

1. `sum`
2. `count`
3. `avg`
4. `min`
5. `max`
6. `first`
7. `last`

## Related Docs

1. [Runtime: Engine and Execution](../runtime/engine-and-execution.md)
2. [Harness Runner](../harness/runner.md)
