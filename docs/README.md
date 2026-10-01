# Documentation

AgentLane is a Python runtime for agents that send messages, receive events,
and keep state across tasks. Use these guides to build an agent app, extend
the harness, or run agents across workers.

## Start here

To install AgentLane and run your first example, follow the
[project quickstart](../README.md#quick-start).

Choose a guide for the part of your app that you need to build:

- **Local agent loops:** Start with [Default agents](./harness/default-agents.md).
- **Message-driven services:** Start with
  [Engine and execution](./runtime/engine-and-execution.md).
- **Distributed agents:** Start with
  [Distributed runtime usage](./runtime/distributed-runtime-usage.md).
- **TypeScript apps:** Start with the [Process bridge](./process-bridge/README.md).

For runnable applications, see the [example index](../examples/README.md).

## Architecture

The application-facing layers have the following responsibilities:

- **Runtime and messaging:** Agent addresses, message routing, delivery, and
  execution.
- **Transport:** Payload serialization across process boundaries.
- **Models:** Prompts, tools, schemas, and model calls.
- **Harness:** Agent loops, handoffs, and resumable runs.
- **Tracing:** Execution records across all layers.

Use a local runtime for one process. Use a distributed runtime when you need
cross-worker routing or worker placement. Both use the same public messaging
model.

## Contents

Use the following references to explore each part of AgentLane.

### Runtime

Create local runtimes and distribute work across hosts:

- [Runtime: Engine and Execution](./runtime/engine-and-execution.md)
- [Runtime: Distributed Runtime Usage](./runtime/distributed-runtime-usage.md)
- [Runtime: Distributed Runtime Architecture](./runtime/distributed-runtime-architecture.md)

### Messaging

Route messages and track delivery:

- [Messaging: Routing and Delivery](./messaging/routing-and-delivery.md)

### Transport

Serialize payloads for transport:

- [Transport Serialization](./transport/serialization.md)

### Models

Configure model calls and prompts:

- [Models Overview](./models/overview.md)
- [Models: Prompt Templating](./models/prompt-templating.md)

### Harness

Build agent loops and extend their behavior:

- [Architecture](./harness/architecture.md)
- [Tasks](./harness/tasks.md)
- [Agents](./harness/agents.md)
- [Default Agents](./harness/default-agents.md)
- [Shims](./harness/shims.md)
- [Compaction](./harness/compaction.md)
- [Tools](./harness/tools.md)
- [Skills](./harness/skills.md)
- [Markdown Agent Definitions](./harness/agent-definitions.md)
- [Runner](./harness/runner.md)
- [Run Event Serialization](./harness/event-serialization.md)
- [Distributed Agents](./harness/distributed-agents.md)
- [File I/O Adapters](./harness/filesystem.md)

### Process bridge

Connect a TypeScript app to a Python backend:

- [Overview](./process-bridge/README.md)
- [Runtime Configuration](./process-bridge/runtime-configuration.md)
- [Protocol and Lifecycle](./process-bridge/protocol.md)
- [Development](./process-bridge/development.md)

### Tracing

Inspect execution with tracing:

- [Tracing Overview](./tracing/overview.md)

### Project

Read project history and contribution guidance:

- [Changelog](../CHANGELOG.md)
- [Code style](./code-style/README.md)
- [Maintain documentation](./code-style/documentation.md)
