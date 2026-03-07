# AgentLane

**AgentLane** is an event-based runtime and messaging framework for AI agents.

It is intentionally unopinionated: AgentLane does not dictate how you build
agent intelligence. You can plug in OpenAI Agents SDK, Claude-based agents, or
your own LLM loops and provider clients. AgentLane provides the rails those
agents run on and communicate through.

It was initially inspired by Microsoft's [autogen](https://github.com/microsoft/autogen) framework, which is no longer actively developed. AgentLane's design has evolved to provide better transport guarantees and a clearer path to distributed execution, while maintaining the core vision of an event-driven agent runtime.

![agentlane](https://ossfiles-6842.s3.us-west-2.amazonaws.com/agentlane-gh-banner-1280x640.png)

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-CC--BY--4.0-lightgrey?style=flat-square" alt="License: CC BY 4.0"></a>
  <a href="https://img.shields.io/badge/python-3.12-blue?style=flat-square"><img src="https://img.shields.io/badge/python-3.12-blue?style=flat-square" alt="Python 3.12"></a>
</p>

## Why AgentLane

1. **Event-first communication model**: direct RPC (`send_message`) and pub/sub
   fan-out (`publish_message`) are first-class.
2. **Sensible defaults**: common payload types work out of the box
   (dataclass/pydantic/protobuf/JSON-like values).
3. **Clear runtime guarantees**: per-recipient FIFO ordering, fair scheduling,
   and explicit delivery outcomes.
4. **Scalability path**: in-process engine available today, distributed runtime
   contract in place for transport/placement evolution.
5. **Framework-agnostic agent authoring**: handlers are plain async Python with
   typed payloads and explicit context.

## What AgentLane Provides

1. Runtime orchestration for agent delivery lifecycle.
2. Agent registration and instance lifecycle (`register_factory`,
   `register_instance`).
3. Topic subscriptions with route-key affinity and delivery mode selection.
4. Structured outcomes and acknowledgments:
   - `DeliveryOutcome` for direct sends,
   - `PublishAck` for publish enqueue confirmation.
5. Correlation and idempotency primitives (`correlation_id`,
   `idempotency_key`).
6. Transport serialization registry with inference and extension points.

## What AgentLane Does Not Dictate

1. LLM provider choice.
2. Prompting/orchestration strategy inside handlers.
3. Agent framework selection (you can mix and match per project).

## Core Messaging Semantics

1. `send_message(...)`
   - one recipient,
   - waits for terminal `DeliveryOutcome`,
   - use for request/response paths.
2. `publish_message(...)`
   - routes one event to all matching subscriptions,
   - returns `PublishAck` after enqueue,
   - does not wait for downstream handler completion.
3. `DeliveryMode.STATEFUL`
   - recipient key is derived from topic route key,
   - reuses same instance for that `(agent_type, route_key)` identity.
4. `DeliveryMode.STATELESS`
   - uses per-delivery transient recipient identity,
   - avoids instance reuse guarantees.

## Architecture Overview

```text
Application / Agent Handlers
          |
          v
RuntimeEngine API
  - send_message
  - publish_message
          |
          v
RoutingEngine -> Publish routes (topic + delivery mode)
          |
          v
PerAgentMailboxScheduler
  - FIFO per AgentId
  - fair round-robin across recipients
          |
          v
Dispatcher -> AgentRegistry -> @on_message handler invocation
          |
          v
DeliveryOutcome / PublishAck
```

Current execution engine:

1. `SingleThreadedRuntimeEngine` is the active runtime path in v1.
2. `DistributedRuntimeEngine` is currently a contract placeholder and raises
   `NotImplementedError` on delivery submission.

## Quickstart

### Prerequisites

1. Python `3.12`
2. [`uv`](https://docs.astral.sh/uv/)

### Install Workspace Dependencies

```bash
uv sync --all-extras
```

### Minimal Example

```python
import asyncio
from dataclasses import dataclass

from agentlane.messaging import AgentId, MessageContext
from agentlane.runtime import BaseAgent, Engine, on_message, single_threaded_runtime


@dataclass(slots=True)
class Ping:
    text: str


class EchoAgent(BaseAgent):
    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)

    @on_message
    async def handle(self, payload: Ping, context: MessageContext) -> object:
        _ = context
        return {"echo": payload.text}


async def main() -> None:
    async with single_threaded_runtime() as runtime:
        runtime.register_factory("echo", EchoAgent)
        outcome = await runtime.send_message(
            Ping(text="hello"),
            recipient=AgentId.from_values("echo", "session-1"),
        )
        print(outcome.status.value, outcome.response_payload)


asyncio.run(main())
```

### Run Included Examples

```bash
# High-throughput mixed RPC + publish demo
uv run python examples/throughput/high_throughput_messaging/main.py

# Multi-agent workflow demo
uv run python examples/runtime/multi_agent_workflow/main.py
```

## Documentation Map

1. [Documentation Index](./docs/README.md)
2. [Runtime: Engine and Execution](./docs/runtime/engine-and-execution.md)
3. [Messaging: Routing and Delivery](./docs/messaging/routing-and-delivery.md)
4. [Agent Handler Patterns](./docs/agents/handler-patterns.md)
5. [Transport Serialization](./docs/transport/serialization.md)
6. [Examples Index](./examples/README.md)

## Local Development

```bash
/usr/bin/make format
/usr/bin/make lint
/usr/bin/make tests
```

Run a single test with:

```bash
uv run pytest -s -k <test_name>
```

## Contributing

1. Keep changes small and focused.
2. Add/update tests when behavior changes.
3. Use the PR template at
   `.github/PULL_REQUEST_TEMPLATE/pull_request_template.md`.
4. Ensure formatting, linting, type-checking, and tests pass before opening a
   PR.
