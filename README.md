# AgentLane

**AgentLane is a runtime-first framework for building reliable, inspectable AI
agent systems.**

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![PyPI](https://img.shields.io/pypi/v/agentlane.svg)

AgentLane is for AI workflows where agent behavior is part of the application
architecture. It gives agents stable identities, routes work through explicit
messages, and lets a local agent loop grow into background workers, pub/sub
flows, and distributed runtimes without changing the core communication model.

Most agent frameworks start with a prompt, a few tools, and a loop. AgentLane
starts one layer lower: runtime, addressed messaging, delivery outcomes, and
agent instance reuse. Model calls, tools, handoffs, and the default harness sit
on top of that runtime foundation.

That shape matters when users depend on the system. You need to know which agent
handled work, where state lives, which messages and tools were involved, how
work was delegated, and how the workflow can be tested, reproduced, and
operated.

## What You Get

AgentLane is organized into layers that can be used together or independently:

1. **[Runtime](src/agentlane/runtime/) and
   [Messaging](src/agentlane/messaging/)** — addressed agents, direct sends,
   scheduling, pub/sub, delivery outcomes, local execution, and distributed
   workers.
2. **[Models](src/agentlane/models/)** — prompt templates, schemas, structured
   outputs, native tools, and provider clients.
3. **[Harness](src/agentlane/harness/)** — `DefaultAgent`, resumable run state,
   tool execution, handoffs, agent-as-tool delegation, shims, and skills.
4. **[Transport](src/agentlane/transport/)** — wire-safe serialization
   boundaries for distributed payloads.
5. **[Tracing](src/agentlane/tracing/)** — observability across runtime, model,
   and harness execution.

These layers let you start with a simple local agent and keep the same runtime
model as the workflow grows into addressed services, background specialists,
fan-out and fan-in, or distributed execution.

## When To Use AgentLane

Use AgentLane when you are building AI systems that need one or more of:

1. local agents with tools, handoffs, delegation, or resumable runs
2. stable identities for agents, services, and background specialists
3. explicit routing between model-backed agents and deterministic workers
4. fan-out, fan-in, pub/sub, or human-review workflows
5. structured model calls with schemas, tools, and provider adapters
6. a path from local development to distributed execution
7. orchestration that stays in application code

AgentLane is especially useful when the agent workflow is part of the product
architecture and carries responsibilities beyond a single model call.

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

After installing the package, define an agent descriptor against your model client:

```python
from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent


descriptor = AgentDescriptor(
    name="Care Navigation",
    model=model,
    instructions="You are a concise patient care navigation agent.",
)

# The descriptor is the static model, prompt, and tool contract for this agent.
agent = DefaultAgent(descriptor=descriptor)

# Each run executes one user turn and stores resumable run state on the agent.
result = await agent.run("I feel dizzy after starting a new medication. What should I do first?")
```

This is the simplest entry point.

For distributed execution, keep the user-facing `DefaultAgent` at the top and
send focused work to another addressed agent:

```python
from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.messaging import AgentId, MessageContext
from agentlane.models import Tools
from agentlane.runtime import BaseAgent, distributed_runtime, on_message


class SafetyReviewAgent(BaseAgent):
    @on_message
    async def handle(self, case: str, context: MessageContext) -> object:
        # Worker agent: receives addressed work through the runtime.
        _ = context
        return {"recommendation": "same-day clinician review"}


async with distributed_runtime() as runtime:
    # 1. Register the worker agent type with the distributed runtime.
    runtime.register_factory("safety_review", SafetyReviewAgent)

    async def request_safety_review(case: str) -> object:
        """Ask the addressed worker agent for a focused safety review."""
        # 2. Bridge the model-facing tool call into runtime message routing.
        outcome = await runtime.send_message(
            case,
            recipient=AgentId.from_values("safety_review", "case-1"),
        )
        return outcome.response_payload

    descriptor = AgentDescriptor(
        name="Care Navigation",
        model=model,
        instructions="Call `request_safety_review` for safety-sensitive cases.",
        tools=Tools(tools=[request_safety_review]),
    )

    # 3. Run the user-facing agent on the same distributed runtime.
    agent = DefaultAgent(
        descriptor=descriptor,
        runtime=runtime,
    )
    result = await agent.run("Review this case and identify the next step.")
```

For explicit worker placement, pub/sub, or multi-process execution, use the
runtime layer directly.

## Repository examples

If you are running from a repository checkout, run one runtime example:

```bash
uv run python examples/runtime/multi_agent_workflow/main.py
```

Run one high-level harness example with a real model:

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/default_agent_quickstart/main.py
```

Run the distributed harness agent smoke test:

```bash
uv run python examples/harness/distributed_clinical_inbox_copilot/main.py \
  --multiprocess \
  --smoke-review
```

The runtime example shows explicit message passing. The distributed harness
example shows a top-level agent coordinating worker runtimes through
publish-based fan-out and fan-in.

## Choose the layer you need

### Runtime

Use the runtime when agent identity, message routing, pub/sub, scheduling, or distributed execution are part of your application design.

Start here:

1. [Runtime: Engine and Execution](docs/runtime/engine-and-execution.md)
2. [Runtime: Distributed Runtime Usage](docs/runtime/distributed-runtime-usage.md)
3. [Messaging: Routing and Delivery](docs/messaging/routing-and-delivery.md)

### Models

Use the models layer when you want reusable prompt templates, schemas, structured outputs, tools, or provider clients without adopting the full agent harness.

Start here:

1. [Overview](docs/models/overview.md)
2. [Prompt Templating](docs/models/prompt-templating.md)

### Harness

Use the harness when you want high-level agents, reusable loops, tool execution, handoffs, or agent-as-tool patterns on top of the lower-level primitives.

Start here:

1. [Default Agents](docs/harness/default-agents.md)
2. [Architecture](docs/harness/architecture.md)
3. [Tools](docs/harness/tools.md)
4. [Shims](docs/harness/shims.md)
5. [Skills](docs/harness/skills.md)
6. [Distributed Agents](docs/harness/distributed-agents.md)

## Documentation

Use the documentation index for the full docs tree:

1. [Documentation Index](docs/README.md)
2. [Examples Index](examples/README.md)
3. [Runtime: Distributed Runtime Usage](docs/runtime/distributed-runtime-usage.md)
4. [Harness Distributed Agents](docs/harness/distributed-agents.md)
5. [Tracing Overview](docs/tracing/overview.md)
6. [Changelog](CHANGELOG.md)

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
