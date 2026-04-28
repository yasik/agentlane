# AgentLane

**AgentLane is a runtime-first orchestration layer for building reliable, inspectable, production AI agents and workflows workflows.**

It is designed for systems where agent behavior needs to be explicit, structured, testable, and operable — especially in serious domains like healthcare where opaque prompt chains and autonomous demo loops are not enough.

AgentLane helps you turn fragile prompt/tool chains into workflows with clear runtime boundaries: long-lived agents, addressed messaging, structured model interactions, tool execution, handoffs, pub/sub flows, and a path from local development to distributed runtime execution.

Most agent frameworks start with the agent loop.

AgentLane starts one layer lower: with runtime, identity, and addressed messaging.

That makes it useful when you want to build AI workflows that are easier to inspect, test, route, scale, and operate.

AgentLane gives you three layers that can be used together or independently:

1. `agentlane.runtime` — delivery, routing, scheduling, pub/sub, and agent identity
2. `agentlane.models` — prompts, schemas, tools, structured outputs, and model clients
3. `agentlane.harness` — agent loops, tool execution, handoffs, and high-level agents

```text
╔════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                    ║
║    █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗      █████╗ ███╗   ██╗███████╗   ║
║   ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██║     ██╔══██╗████╗  ██║██╔════╝   ║
║   ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║     ███████║██╔██╗ ██║█████╗     ║
║   ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║     ██╔══██║██║╚██╗██║██╔══╝     ║
║   ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ███████╗██║  ██║██║ ╚████║███████╗   ║
║   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝   ║
║                                                                                    ║
║                reliable, inspectable AI agent workflows                            ║
║                                                                                    ║
║              runtime • messaging • model primitives • harness                      ║
║                                                                                    ║
║                 from local agents → distributed agent systems                      ║
║                                                                                    ║
╚════════════════════════════════════════════════════════════════════════════════════╝
```

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![PyPI](https://img.shields.io/pypi/v/agentlane.svg)

## Why AgentLane?

Many agent systems start as a prompt, a few tools, and a loop.

That works for demos. But production systems usually need more structure:

1. stable agent identity across turns and tasks
2. explicit message routing instead of hidden in-process calls
3. background specialists that can run independently
4. fan-out, fan-in, and pub/sub workflows
5. bounded handoffs between agents and tools
6. structured model calls that can be tested and reused
7. a path from local execution to distributed workers
8. runtime behavior that application code can reason about

AgentLane lets you start with simple local agents, then grow into addressed services, background workers, and multi-agent workflows without changing the core communication model.

## What AgentLane is

AgentLane is a framework for building AI systems as explicit workflows of addressed agents, model calls, tools, messages, and handoffs.

It is useful when you care about runtime behavior: who receives work, where state lives, how messages are routed, how agents coordinate, and how a local prototype can evolve into a distributed system.

AgentLane is designed for builders who want production AI workflows to be:

1. **Reliable** — agent execution should be structured enough to reason about, test, and debug.
2. **Inspectable** — important behavior should be visible through explicit messages, tools, handoffs, and runtime boundaries.
3. **Composable** — agents, model calls, tools, and services should be reusable building blocks.
4. **Operable** — workflows should have a path from local development to long-running services and distributed workers.
5. **Bounded** — agent autonomy should live inside application-controlled orchestration, not behind an opaque loop.

## What AgentLane is not

AgentLane is not a single autonomous agent loop.

It does not try to hide application architecture behind a provider-owned abstraction. The goal is to keep orchestration, routing, and workflow design in application code, where they can be inspected, tested, and evolved.

## Serious domains need serious agent infrastructure

In low-stakes demos, it may be enough to let an LLM call tools in a loop until it produces a plausible result.

In serious domains — healthcare, finance, compliance, operations, infrastructure, or any product where users rely on the system — agent behavior needs stronger guarantees.

You often need to know:

1. which agent or service handled a task
2. what messages were exchanged
3. which tools were called
4. where state was stored
5. how work was delegated
6. where a human should review or intervene
7. how the workflow can be reproduced, tested, and improved

AgentLane is built around that worldview: production agents should be explicit systems, not invisible loops.

## When to use AgentLane

Use AgentLane when you are building AI systems that need one or more of:

1. local agents with tools, delegation, handoffs, or resumable runs
2. long-lived agents or services with stable identities
3. explicit routing between agents, tools, and background workers
4. fan-out, fan-in, or pub/sub workflows
5. structured model calls with schemas, tools, and provider adapters
6. a path from local development to distributed execution
7. application-level control instead of provider-owned orchestration

AgentLane is especially useful when the agent workflow is part of the product architecture, not just a wrapper around a model call.

## Design principles

1. **Runtime first** — agent behavior should be part of the application runtime, not hidden inside a black-box loop.
2. **Addressable by default** — agents and services should have stable identities that can receive messages directly.
3. **Composable layers** — use the runtime, model primitives, or harness independently when needed.
4. **Provider-thin** — keep orchestration in application code instead of outsourcing it to a model provider.
5. **Local to distributed** — start in one process and preserve the same communication model as the system grows.
6. **Explicit over magical** — prefer inspectable workflows, messages, tools, and handoffs over implicit control flow.
7. **Human-compatible** — design workflows so humans can review, intervene, and understand what happened when needed.

## Installation

Install AgentLane with `uv`:

```bash
uv add agentlane
```

If you are trying the repository directly instead:

```bash
uv sync --all-extras
```

## Quick Start

The harness gives you a simple agent interface when you want one, while still letting you drop down into explicit runtime and messaging primitives as your system grows.

After installing the package, define an agent against your model client:

```python
from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent


class SupportAgent(DefaultAgent):
    descriptor = AgentDescriptor(
        name="Support",
        model=model,
        instructions="You are a concise support agent.",
    )


agent = SupportAgent()
result = await agent.run("My order arrived damaged. What should I do first?")
```

This is the simplest entry point.

For workflows that need explicit routing, background specialists, pub/sub, or distributed execution, use the runtime layer directly.

## Repository examples

If you are running from a repository checkout, run one runtime example:

```bash
uv run python examples/runtime/multi_agent_workflow/main.py
```

Run one high-level harness example with a real model:

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/default_agent_quickstart/main.py
```

The runtime example shows explicit message passing.

## Choose the layer you need

### Runtime

Use the runtime when agent identity, message routing, pub/sub, scheduling, or distributed execution are part of your application design.

Start here:

1. [Runtime: Engine and Execution](docs/runtime/engine.md)
2. [Messaging: Routing and Delivery](docs/runtime/messaging.md)

### Models

Use the models layer when you want reusable prompt templates, schemas, structured outputs, tools, or provider clients without adopting the full agent harness.

Start here:

1. [Overview](docs/models/overview.md)
2. [Prompt Templating](docs/models/prompt_templating.md)

### Harness

Use the harness when you want high-level agents, reusable loops, tool execution, handoffs, or agent-as-tool patterns on top of the lower-level primitives.

Start here:

1. [Default Agents](docs/harness/default_agents.md)
2. [Architecture](docs/harness/architecture.md)

## Documentation

Use the documentation index for the full docs tree:

1. [Documentation Index](docs/README.md)
2. [Examples Index](examples/README.md)

## Origins

AgentLane was initially inspired by Microsoft AutoGen, but takes a runtime-first approach focused on addressed messaging, explicit orchestration, and local-to-distributed execution.

## Development

Format, lint, and test:

```bash
/usr/bin/make format
/usr/bin/make lint
/usr/bin/make tests
```

Run one test with:

```bash
uv run pytest -s -k <test_name>
```

## Contributing

1. Keep changes small and focused.
2. Add or update tests when behavior changes.
3. Update public docs and examples when the developer-facing surface changes.
4. Ensure formatting, linting, and tests pass before opening a PR.
