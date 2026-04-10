# Tracing Overview

`agentlane.tracing` provides the core tracing primitives used to instrument
runtime, model, and application code.

It gives AgentLane a shared trace and span model with context propagation and
trace-level metrics. Use this layer when you want structured observability
without coupling tracing logic to a specific runtime or provider.
The core tracing types are
[`TraceProvider`](../../src/agentlane/tracing/_provider.py),
[`DefaultTraceProvider`](../../src/agentlane/tracing/_provider.py),
[`Trace`](../../src/agentlane/tracing/_trace.py),
[`Span`](../../src/agentlane/tracing/_span.py),
[`MetricsProcessor`](../../src/agentlane/tracing/_metrics_processor.py), and
[`TracingProcessor`](../../src/agentlane/tracing/_processor_interface.py).

## What It Includes

1. `trace(...)` for top-level trace scopes.
2. `agent_span(...)`, `function_span(...)`, `generation_span(...)`, and
   `custom_span(...)` for common span types.
3. context helpers such as `get_current_trace()`, `get_current_span()`, and
   parent-context propagation functions.
4. a metrics registry and aggregation helpers for trace-level metrics.
5. provider and processor interfaces for exporting traces and metrics.

## Boundaries

1. This package defines tracing primitives and metrics aggregation.
2. It does not own runtime delivery, model calls, or harness orchestration.
3. Runtime and model layers can emit tracing data through this shared surface
   instead of each layer defining its own tracing contract.

## Provider Setup

Tracing uses one global
[`TraceProvider`](../../src/agentlane/tracing/_provider.py).

The default setup path is:

1. create a `DefaultTraceProvider`
2. register one or more processors on it
3. install it with `set_trace_provider(...)`

```python
from agentlane.tracing import (
    DefaultTraceProvider,
    MetricsProcessor,
    set_trace_provider,
)

provider = DefaultTraceProvider()
provider.register_processor(MetricsProcessor())
set_trace_provider(provider)
```

Use `get_trace_provider()` when you need to inspect or extend the installed
provider later in process startup.

## Minimal Example

```python
from agentlane.tracing import emit_metric, generation_span, trace


with trace("customer_support_run"):
    with generation_span(model="gpt-5.4-mini"):
        emit_metric("llm_calls", 1)
```

## Context Propagation

Use the propagation helpers when a trace must cross an async boundary that does
not automatically preserve the current tracing scope.

The public helpers are:

1. `capture_parent_context(message_id)` before handing work off
2. `adopt_parent_context(message_id)` while processing that work
3. `discard_parent_context(message_id)` to drop stored context explicitly

These helpers are useful for message-driven runtime flows where the producer and
consumer execute in different tasks or processes.

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

## Processors And Export

[`TracingProcessor`](../../src/agentlane/tracing/_processor_interface.py) is
the export boundary for traces and spans.

Common processor setup patterns are:

1. `MetricsProcessor` to aggregate per-trace metrics and optionally hand them to
   another exporter callback
2. a custom `TracingProcessor` implementation for your own sink
3. `agentlane-braintrust` with `BraintrustProcessor` when you want Braintrust
   export for traces, spans, and aggregated metrics

Example Braintrust registration:

```python
from agentlane.tracing import DefaultTraceProvider, set_trace_provider
from agentlane_braintrust import BraintrustProcessor

provider = DefaultTraceProvider()
provider.register_processor(
    BraintrustProcessor(api_key="...", project_id="...")
)
set_trace_provider(provider)
```

## Related Docs

1. [Runtime: Engine and Execution](../runtime/engine-and-execution.md)
2. [Harness Runner](../harness/runner.md)
