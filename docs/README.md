# Documentation

AgentLane is built for AI systems where the runtime is part of the product. At
the center is a messaging runtime: agents have stable addresses, receive direct
messages and published events, preserve per-recipient ordering, and keep state
when work needs to live longer than one model turn. That makes the framework a
good fit for long-horizon agentic tasks, background specialists, fan-out and
fan-in flows, and applications that mix deterministic services with
model-driven components.

Many adjacent frameworks start from the agent loop itself. They are often
optimized either for interactive local or cloud sessions and delegated
background work, or for durable runs expressed as explicit workflows and
graphs.

AgentLane starts one layer lower: it treats addressed messaging, routing,
delivery outcomes, and instance reuse as the core abstraction, then layers
prompts, tools, and a default harness on top.

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
3. If you want the smallest local harness surface, read
   [Harness Default Agents](./harness/default-agents.md), then
   [Harness Runner](./harness/runner.md).
4. If you want to understand how the harness fits together under that local
   surface, read [Harness Architecture](./harness/architecture.md).
5. If you want to extend harness behavior without changing core harness types,
   read [Harness Shims](./harness/shims.md).
6. If you want to understand the first shim-driven extension module, read
   [Harness Skills](./harness/skills.md) after
   [Harness Shims](./harness/shims.md).
7. If you want first-party local workspace tools, read
   [Harness Tools](./harness/tools.md).
8. If you want to understand prompt construction or tool exposure, read
   [Models Overview](./models/overview.md) and
   [Models: Prompt Templating](./models/prompt-templating.md).
9. If you specifically want streaming, read [Models Overview](./models/overview.md)
   first, then [Harness Default Agents](./harness/default-agents.md).
10. Run one of the examples under [examples/](../examples/README.md).

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

1. [Architecture](./harness/architecture.md)
2. [Tasks](./harness/tasks.md)
3. [Agents](./harness/agents.md)
4. [Default Agents](./harness/default-agents.md)
5. [Shims](./harness/shims.md)
6. [Skills](./harness/skills.md)
7. [Tools](./harness/tools.md)
8. [Runner](./harness/runner.md)

### Tracing

1. [Tracing Overview](./tracing/overview.md)
