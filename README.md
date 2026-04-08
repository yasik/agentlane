# AgentLane

**AgentLane** is an event-driven framework for building AI systems with three
composable layers:

1. a runtime for message delivery, routing, and scheduling
2. a model layer for prompts, schemas, tools, and provider adapters
3. a harness for agent loops, tool execution, handoffs, and delegated sub-agents

It is intentionally unopinionated about the intelligence inside an agent. You
can use the runtime directly, build on the harness, or plug in your own
orchestration strategy on top of the model primitives.

It was initially inspired by Microsoft's
[autogen](https://github.com/microsoft/autogen) framework. AgentLane's design
has since evolved toward clearer runtime guarantees, a thinner provider
boundary, and a cleaner path from local execution to distributed messaging.

![agentlane](https://ossfiles-6842.s3.us-west-2.amazonaws.com/agentlane-gh-banner-1280x640.png)

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License: MIT"></a>
  <a href="https://img.shields.io/badge/python-3.12-blue?style=flat-square"><img src="https://img.shields.io/badge/python-3.12-blue?style=flat-square" alt="Python 3.12"></a>
</p>

## Key Differentiators

Many agent frameworks stop at an in-process agent loop. AgentLane is built
around a runtime and messaging model first, then layers model primitives and a
default harness on top of that foundation.

1. **Runtime-first, not loop-first**: agents communicate through explicit
   messaging and routing primitives instead of assuming everything is a local
   function call inside one process.
2. **Same mental model from local to distributed**: the same `send_message(...)`
   and `publish_message(...)` semantics work in single-threaded and distributed
   runtime setups.
3. **Clear delivery guarantees**: per-recipient FIFO ordering, fair scheduling,
   explicit delivery outcomes, and durable instance identity are part of the
   framework contract.
4. **Harness built on the runtime**: handoffs and delegated
   sub-agents route through the same runtime messaging model as everything else,
   which keeps local and future distributed behavior aligned.
6. **Run-oriented public API**: developers work with descriptors, tools,
   prompts, `RunState`, and `RunResult` instead of assembling low-level model
   wire payloads by hand.
7. **Framework flexibility**: you can use only the
   runtime, only the model layer, or the full harness, instead of being forced
   into one monolithic abstraction stack.

## Key Concepts

### Runtime

The runtime is the foundation:

1. direct request/response via `send_message(...)`
2. publish / subscribe fan-out via `publish_message(...)`
3. explicit registration and instance reuse
4. FIFO delivery per `AgentId` with fair scheduling across recipients
5. in-process and distributed runtime entrypoints

### Models

`agentlane.models` is the shared LLM-facing layer:

1. typed prompt templates via `PromptTemplate`, `MultiPartPromptTemplate`, and `PromptSpec`
2. structured output via `OutputSchema`
3. native tools via `Tool`, `Tools`, `ToolSpec`, and `@as_tool`
4. provider-agnostic `Model` and canonical `ModelResponse`
5. retries, rate limiting, and response helpers

### Harness

`agentlane.harness` is the higher-level orchestration layer:

1. `Task` for top-level units of work
2. `AgentDescriptor` for static agent configuration
3. `Agent` for per-agent lifecycle and resumable state
4. `Runner` for the generic LLM loop
5. runner-owned tool execution
6. first-class handoffs
7. agent-as-tool subroutines
8. resumable `RunState` and final `RunResult`

## Why AgentLane

1. **Event-first communication model**: direct RPC and pub/sub are first-class.
2. **Clear runtime guarantees**: FIFO per recipient, explicit delivery
   outcomes, and fair scheduling.
3. **Provider-agnostic model layer**: prompts, schemas, tools, and retries live
   behind one shared contract.
4. **Thin provider adapters**: provider clients accept canonical requests and
   return canonical responses; they do not own orchestration policy.
5. **Built-in harness**: the framework now ships a default agent loop with tool
   execution, delegation, and resumable state.
6. **Scalability path**: the same messaging model works in-process and in
   distributed host / worker setups.

## Layered Architecture

```text
Application code
      |
      +---------------------------+
      |                           |
      v                           v
Runtime-first usage         Harness usage
send_message / publish      Task / Agent / Runner
      |                           |
      +-------------+-------------+
                    |
                    v
            RuntimeEngine API
                    |
                    v
         routing + mailbox scheduling
                    |
                    v
             @on_message handlers

Harness path adds:

AgentDescriptor -> AgentLifecycle -> Runner
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
               ToolExecutor                   runtime messaging
               local tools                    delegated agents
                    |
                    v
             agentlane.models
     prompts / schema / tools / ModelResponse
                    |
                    v
              provider clients
                    |
                    v
                   LLM
```

## Core Messaging Semantics

1. `send_message(...)`
   - one recipient
   - waits for terminal `DeliveryOutcome`
   - use for request / response paths
2. `publish_message(...)`
   - routes one event to all matching subscriptions
   - returns `PublishAck` after enqueue
   - does not wait for downstream handler completion
3. `DeliveryMode.STATEFUL`
   - recipient key is derived from topic route key
   - reuses the same instance for that `(agent_type, route_key)` identity
4. `DeliveryMode.STATELESS`
   - uses per-delivery transient recipient identity
   - avoids instance reuse guarantees

## Quick Start

### Prerequisites

1. Python `3.12`
2. [`uv`](https://docs.astral.sh/uv/)

### Install Workspace Dependencies

```bash
uv sync --all-extras
```

### Minimal Runtime Example

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

### Harness Examples

The harness examples show the newer orchestration surface:

1. multi-turn conversation plus `RunState` resume
2. tool calling with a native `@as_tool` function
3. predefined and generic agent-as-tool flows
4. predefined and generic handoff flows

These examples use `agentlane-openai` and require `OPENAI_API_KEY`.

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/customer_support_conversation/main.py
OPENAI_API_KEY=sk-... uv run python examples/harness/tool_calling_search_answer/main.py
OPENAI_API_KEY=sk-... uv run python examples/harness/agent_as_tool_policy_specialist/main.py
OPENAI_API_KEY=sk-... uv run python examples/harness/default_agent_tool_note_writer/main.py
OPENAI_API_KEY=sk-... uv run python examples/harness/handoff_to_returns_specialist/main.py
OPENAI_API_KEY=sk-... uv run python examples/harness/default_handoff_takeover/main.py
```

### Runtime Examples

```bash
uv run python examples/throughput/high_throughput_messaging/main.py
uv run python examples/runtime/multi_agent_workflow/main.py
uv run python examples/runtime/distributed_publish_fan_in/main.py
uv run python examples/runtime/distributed_scatter_gather/main.py
uv run python examples/runtime/simple/distributed_publish_fan_in.py
uv run python examples/runtime/simple/distributed_scatter_gather.py
```

## Documentation Map

### Runtime And Messaging

1. [Documentation Index](./docs/README.md)
2. [Runtime: Engine and Execution](./docs/runtime/engine-and-execution.md)
3. [Runtime: Distributed Host/Worker Architecture](./docs/runtime/distributed-runtime-architecture.md)
4. [Runtime: Distributed Runtime Usage](./docs/runtime/distributed-runtime-usage.md)
5. [Messaging: Routing and Delivery](./docs/messaging/routing-and-delivery.md)
6. [Agent Handler Patterns](./docs/agents/handler-patterns.md)
7. [Transport Serialization](./docs/transport/serialization.md)

### Harness And Models

1. [Harness Architecture](./docs/harness/architecture.md)
2. [Harness Tasks](./docs/harness/tasks.md)
3. [Harness Agents](./docs/harness/agents.md)
4. [Harness Runner](./docs/harness/runner.md)
5. [Examples Index](./examples/README.md)
6. [Models Layer Overview](./src/agentlane/models/README.md)

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
2. Add or update tests when behavior changes.
3. Update public docs and examples when the developer-facing surface changes.
4. Use the PR template at `.github/PULL_REQUEST_TEMPLATE/pull_request_template.md`.
5. Ensure formatting, linting, type-checking, and tests pass before opening a PR.
