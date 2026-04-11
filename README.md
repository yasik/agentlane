# AgentLane

**AgentLane** is a runtime-first framework for building AI agents with
addressed messaging, model primitives, and a reusable agent harness.

(Initially inspired by Microsoft's [autogen](https://github.com/microsoft/autogen) framework that is no longer being actively developed).

It gives you three layers that can be used together or independently:

1. `agentlane.runtime` for delivery, routing, scheduling, and agent identity
2. `agentlane.models` for prompts, schemas, tools, and model clients
3. `agentlane.harness` for agent loops, tool execution, handoffs, and
   high-level agents

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
║                    runtime-first AI agent framework                                ║
║                                                                                    ║
║        addressed messaging • model primitives • reusable agent harness             ║
║                                                                                    ║ 
╠════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                    ║
║   Runtime      → delivery • routing • scheduling • identity                        ║
║   Models       → prompts • schemas • tools • model clients                         ║
║   Harness      → loops • execution • handoffs • agents                             ║
║                                                                                    ║
║   from local agents → to distributed multi-agent systems                           ║
║                                                                                    ║
╚════════════════════════════════════════════════════════════════════════════════════╝
```

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License: MIT"></a>
  <a href="https://img.shields.io/badge/python-3.12-blue?style=flat-square"><img src="https://img.shields.io/badge/python-3.12-blue?style=flat-square" alt="Python 3.12"></a>
</p>

## Why AgentLane?

AgentLane is built for AI systems where messaging, addressing, and runtime
behavior are part of the application design.

It is a good fit when you want:

1. one programming model from local runs to distributed execution
2. explicit message routing instead of hidden in-process orchestration
3. a high-level agent interface without giving up lower-level runtime control
4. thin provider adapters instead of provider-owned orchestration logic
5. a path from simple local agents to multi-agent systems with tools and
   handoffs

## Why Runtime And Messaging Matter

Many agent frameworks center on a single in-process agent loop or workflow
graph. That is a good fit for one request running in one place.

AgentLane starts one layer lower, at addressed messaging and runtime
execution. That matters because it gives application code a stable way to:

1. send work to a specific long-lived agent or service by address
2. keep state attached to that addressed instance across multiple turns
3. publish one event to many subscribers without hard-coding the fan-out
4. split work across specialists and gather results back
5. move from local execution to distributed workers without changing the core
   communication model

In practice, that means the same framework can cover both:

1. local agent(s) running in one process
2. a system of addressed agents, background specialists, and pub/sub flows
   spread across workers

The messaging layer is what makes those two use cases part of one model instead
of two different frameworks glued together.

## When To Use It

Use AgentLane for:

1. local agents that need tools, delegation, or resumable runs
2. background specialists and addressed services that communicate by message
3. fan-out, fan-in, and pub/sub workflows
4. applications that start in one process and may later move to a distributed
   runtime

## Quick Start

If you are trying the repository directly:

```bash
uv sync --all-extras
```

Run one runtime example:

```bash
uv run python examples/runtime/multi_agent_workflow/main.py
```

Run one high-level harness example with a real model:

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/default_agent_quickstart/main.py
```

The runtime example shows explicit message passing.

The harness example shows the smallest local agent surface:

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

## Choose Your Level

### Runtime

Use the runtime when you want explicit messaging, stable identities, pub/sub,
or distributed execution.

Start here:

1. [Runtime: Engine and Execution](./docs/runtime/engine-and-execution.md)
2. [Messaging: Routing and Delivery](./docs/messaging/routing-and-delivery.md)

### Models

Use the models layer when you want prompt templates, structured outputs, native
tools, or provider clients without the full harness.

Start here:

1. [Overview](./docs/models/overview.md)
2. [Prompt Templating](./docs/models/prompt-templating.md)

### Harness

Use the harness when you want reusable agent loops, tool execution, handoffs,
agent-as-tool, or high-level local agents.

Start here:

1. [Default Agents](./docs/harness/default-agents.md)
2. [Architecture](./docs/harness/architecture.md)

## Documentation

Use the documentation index for the full docs tree:

1. [Documentation Index](./docs/README.md)
2. [Examples Index](./examples/README.md)

## Development

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
