# Documentation

AgentLane is built for AI systems where the runtime is part of the product. At
the center is a messaging runtime: agents have stable addresses, receive direct
messages and published events, preserve per-recipient ordering, and keep state
when work needs to live longer than one model turn. That makes the framework a
good fit for long-horizon agentic tasks, background specialists, fan-out and
fan-in flows, and applications that mix deterministic services with
model-driven components.

Many adjacent frameworks start from the agent loop itself. They are often optimized 
either for interactive local or cloud sessions and delegated background work, or 
for durable runs expressed as explicit workflows and graphs.

AgentLane starts one layer lower: it treats addressed messaging,
routing, delivery outcomes, and instance reuse as the core abstraction, then
layers prompts, tools, and a default harness on top.

In practical terms, this gives you a clean progression. Use
[`single_threaded_runtime()`](../src/agentlane/runtime/_context.py) and the
harness when you want one local agent loop or one in-process service. Move to
[`distributed_runtime()`](../src/agentlane/runtime/_context.py) or explicit
host/worker runtimes when the same system needs cross-worker routing, worker
placement, or cloud execution. The public messaging model stays the same, so
you do not need to redesign the application around a different orchestration
surface just because deployment changed.

At a high level:

1. runtime and messaging define delivery, routing, and execution
2. transport turns payloads into wire-safe values
3. models describe prompts, tools, schemas, and model-call behavior
4. harness adds reusable agent loops, handoffs, and resumable runs
5. tracing cuts across those layers and provides observability

The main application-facing stack looks like this:

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

Tracing is not shown as another wrapper because it is a cross-cutting concern:
it instruments work that happens in every layer rather than owning a separate
execution boundary.

## Start Here

If you are new to the codebase, start with the shortest path to a working
mental model:

1. Read the project [README](../README.md) for the top-level architecture.
2. If you want explicit message passing, start with
   [Runtime: Engine and Execution](./runtime/engine-and-execution.md).
3. If you want the default agent loop, read
   [Harness Architecture](./harness/architecture.md) and then
   [Harness Runner](./harness/runner.md).
4. If you want to understand prompt construction or tool exposure, read
   [Models Overview](./models/overview.md) and
   [Models: Prompt Templating](./models/prompt-templating.md).
5. Run one of the examples under [examples/](../examples/README.md).

## Choose A Topic

Use this table when you know the kind of problem you are solving but not which
page to read first.

| If you want to understand... | Start here |
| --- | --- |
| How messages are sent, published, scheduled, and completed | [Runtime: Engine and Execution](./runtime/engine-and-execution.md) |
| How cross-worker delivery works | [Runtime: Distributed Runtime Usage](./runtime/distributed-runtime-usage.md) and [Runtime: Distributed Runtime Architecture](./runtime/distributed-runtime-architecture.md) |
| How routing, subscriptions, and delivery modes behave | [Messaging: Routing and Delivery](./messaging/routing-and-delivery.md) |
| When serialization matters and when the defaults are enough | [Transport Serialization](./transport/serialization.md) |
| How prompts, tools, and structured outputs fit together | [Models Overview](./models/overview.md) |
| How prompt templates are authored and rendered | [Models: Prompt Templating](./models/prompt-templating.md) |
| How the harness organizes tasks, agents, and runs | [Harness Architecture](./harness/architecture.md) |
| What the default agent owns | [Harness Agents](./harness/agents.md) |
| How the default loop calls models, tools, and handoffs | [Harness Runner](./harness/runner.md) |
| How tracing, metrics, and processors are wired up | [Tracing Overview](./tracing/overview.md) |

## Contents

### Runtime

1. [Runtime: Engine and Execution](./runtime/engine-and-execution.md)
2. [Runtime: Distributed Runtime Usage](./runtime/distributed-runtime-usage.md)
3. [Runtime: Distributed Runtime Architecture](./runtime/distributed-runtime-architecture.md)

### Messaging

1. [Messaging: Routing and Delivery](./messaging/routing-and-delivery.md)

### Transport

1. [Transport Serialization](./transport/serialization.md)

### Models

1. [Models Overview](./models/overview.md)
2. [Models: Prompt Templating](./models/prompt-templating.md)

### Harness

1. [Harness Architecture](./harness/architecture.md)
2. [Harness Tasks](./harness/tasks.md)
3. [Harness Agents](./harness/agents.md)
4. [Harness Runner](./harness/runner.md)

### Tracing

1. [Tracing Overview](./tracing/overview.md)

### Releases

1. [v0.3.0](./releases/v0.3.0.md)
