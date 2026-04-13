# Tracing Overview

Tracing is the shared observability layer for AgentLane. It gives runtime code,
model code, and application code a common trace and span model so the whole
workflow can be understood as one unit instead of as isolated logs.

The public tracing surface stays small. A
[`TraceProvider`](../../src/agentlane/tracing/_provider.py) creates
[`Trace`](../../src/agentlane/tracing/_trace.py) and
[`Span`](../../src/agentlane/tracing/_span.py) values, the default
[`DefaultTraceProvider`](../../src/agentlane/tracing/_provider.py) gives most
applications a ready-made implementation, and processors such as
[`MetricsProcessor`](../../src/agentlane/tracing/_metrics_processor.py) or a
custom [`TracingProcessor`](../../src/agentlane/tracing/_processor_interface.py)
consume what was recorded.

## What Tracing Captures

Tracing is built around a simple idea:

1. a trace represents one workflow or run
2. spans represent meaningful operations inside that workflow
3. processors export or aggregate what happened

That makes tracing useful for both debugging and operations. You can see how a
request moved through a system, and you can also aggregate metrics about that
system over time.

## Getting Started

Tracing uses one global
[`TraceProvider`](../../src/agentlane/tracing/_provider.py). The usual setup is
to create a [`DefaultTraceProvider`](../../src/agentlane/tracing/_provider.py),
register one or more processors, and install it with `set_trace_provider(...)`.

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

Once a provider is installed, traces and spans can be created anywhere in the
process through the shared tracing API.

```python
from agentlane.tracing import emit_metric, generation_span, trace


with trace("customer_support_run"):
    with generation_span(model="gpt-5.4-mini"):
        emit_metric("llm_calls", 1)
```

## Context Propagation

Tracing becomes more interesting once work crosses task or process boundaries.
That is why the package exposes explicit propagation helpers.

Use:

1. `capture_parent_context(message_id)` before handing work off
2. `adopt_parent_context(message_id)` while processing that work
3. `discard_parent_context(message_id)` when stored context should be dropped

Those helpers are especially useful for runtime-driven message flows where the
producer and consumer do not share the same synchronous call stack.

## Metrics

Metrics are emitted inside spans and aggregated at the trace level. That keeps
the metric story close to the trace story: one workflow can contain many metric
events, but the final result can still be summarized once the workflow ends.

[`MetricsProcessor`](../../src/agentlane/tracing/_metrics_processor.py) is the
default building block for that aggregation path.

## Exporters And Processors

[`TracingProcessor`](../../src/agentlane/tracing/_processor_interface.py) is
the extension point for exporting traces and spans to another sink.

The common patterns are:

1. use `MetricsProcessor` when you need aggregated trace metrics
2. implement a custom processor when you need your own export destination
3. install `agentlane[braintrust]` and use `BraintrustProcessor` when you want
   Braintrust export

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
