# Documentation

The pages here focus on how the framework is organized, what each
layer is responsible for, and which public types you work with when building
applications.

AgentLane is easiest to understand as a small stack of cooperating layers:

1. runtime and messaging for delivery, routing, and execution
2. transport for payload serialization across boundaries
3. models for prompts, schemas, tools, and provider-facing model calls
4. harness for agent loops, handoffs, and resumable runs
5. tracing for observability across the other layers

Viewed as the main application-facing stack, the framework can also be pictured
as nested layers with application code wrapped around the core runtime:

```text
+------------------------------------------------------+
| Application                                          |
| user code built on the framework                     |
|                                                      |
|  +------------------------------------------------+  |
|  | Harness                                        |  |
|  | agent loops, handoffs, and resumable runs      |  |
|  |                                                |  |
|  |  +------------------------------------------+  |  |
|  |  | Models                                   |  |  |
|  |  | prompts, schemas, tools, model clients   |  |  |
|  |  |                                          |  |  |
|  |  |  +------------------------------------+  |  |  |
|  |  |  | Transport                          |  |  |  |
|  |  |  | payload serialization boundaries   |  |  |  |
|  |  |  |                                    |  |  |  |
|  |  |  |  +------------------------------+  |  |  |  |
|  |  |  |  | Core Runtime + Messaging     |  |  |  |  |
|  |  |  |  | delivery, routing, execution |  |  |  |  |
|  |  |  |  +------------------------------+  |  |  |  |
|  |  |  +------------------------------------+  |  |  |
|  |  +------------------------------------------+  |  |
|  +------------------------------------------------+  |
+------------------------------------------------------+
```

## Getting Started

If you are new to the codebase, this path is the shortest way to build a
working mental model:

1. Read the project [README](../README.md) for the high-level architecture.
2. Choose your entrypoint:
   - use the runtime directly if you want explicit message passing
   - use the harness if you want a default agent loop on top of the runtime
3. Read the matching docs pages below.
4. Run one of the examples under [examples/](../examples/README.md).

## Contents

### Runtime

1. [Runtime: Engine and Execution](./runtime/engine-and-execution.md)
   Core runtime flow, lifecycle, scheduling, and local versus distributed execution.
2. [Runtime: Distributed Runtime Usage](./runtime/distributed-runtime-usage.md)
   Practical guide for `distributed_runtime()`, explicit hosts, and explicit workers.
3. [Runtime: Distributed Host/Worker Architecture](./runtime/distributed-runtime-architecture.md)
   Design view of host responsibilities, worker responsibilities, and cross-worker routing.

### Messaging

1. [Messaging: Routing and Delivery](./messaging/routing-and-delivery.md)
   Direct send, publish fan-out, subscriptions, delivery modes, and ordering.

### Transport

1. [Transport Serialization](./transport/serialization.md)
   How payloads become wire-safe values and when custom serializers are useful.

### Models

1. [Models Overview](./models/overview.md)
   Shared model-facing primitives: prompts, schemas, tools, retries, and run-scoped helpers.
2. [Models: Prompt Templating](./models/prompt-templating.md)
   How `PromptTemplate`, `MultiPartPromptTemplate`, and `PromptSpec` turn typed values into model input.

### Harness

1. [Harness Architecture](./harness/architecture.md)
   How the harness sits on top of the runtime and models layers.
2. [Harness Tasks](./harness/tasks.md)
   The thin task abstraction for orchestration work that does not need an LLM loop.
3. [Harness Agents](./harness/agents.md)
   The default agent type, its descriptor, tool surface, and resumable state.
4. [Harness Runner](./harness/runner.md)
   The stateless loop that builds requests, calls the model, and processes tools and handoffs.

### Tracing

1. [Tracing Overview](./tracing/overview.md)
   Traces, spans, metrics, processors, and context propagation.
