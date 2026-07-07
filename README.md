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
3. **[Harness](src/agentlane/harness/)** — `DefaultAgent`, markdown agent
   definitions, resumable run state, tool execution, handoffs, agent-as-tool
   delegation, shims, skills, and a local stdio process bridge for TypeScript
   app shells.
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

Define an agent in a markdown file — frontmatter for config, the body as its
system prompt — and run it. The four steps below are one progression: the same
agent grows from a single file to a coordinated team, and from one local
process to a distributed runtime, without changing the model or the run loop.

All four reuse one model client:

```python
import asyncio
import os

from agentlane_openai import ResponsesClient

from agentlane.harness.agents import DefaultAgent
from agentlane.models import Config

model = ResponsesClient(
    config=Config(api_key=os.environ["OPENAI_API_KEY"], model="gpt-5.4-mini"),
)
```

### 1. An agent from a markdown file

`care_navigator.md`:

```markdown
---
name: care-navigator
description: Guides patients to a clear next step for a new symptom or concern.
---
You are a concise patient care navigation agent. Give one clear next step. When
a clinical tool is available, use it before advising on a medication.
```

```python
async def main() -> None:
    agent = DefaultAgent.from_markdown("care_navigator.md", model=model)
    result = await agent.run(
        "I feel dizzy after starting a new blood-pressure medication. What first?"
    )
    print(result.final_output)


asyncio.run(main())
```

One file, one `run(...)`. The frontmatter configures the agent; the body is its
system prompt. It runs on a local single-threaded runtime by default, and each
`run(...)` stores resumable state on the agent.

### 2. Two markdown agents on a distributed runtime

Add a specialist as a sub-agent and bind the pair to a distributed runtime. The
specialist becomes an addressed agent the lead delegates to — ready to move onto
its own worker later. Only `subagents=` and `runtime=` change.

`med_safety.md`:

```markdown
---
name: med-safety
description: Use to check a medication for interactions and safety flags before advising.
model: inherit
---
You review a medication for interactions and safety flags, and return a short
note that says clearly when something is urgent.
```

```python
from agentlane.runtime import distributed_runtime


async def main() -> None:
    async with distributed_runtime() as runtime:
        agent = DefaultAgent.from_markdown(
            "care_navigator.md",
            model=model,
            subagents=["med_safety.md"],
            runtime=runtime,
        )
        result = await agent.run(
            "I started lisinopril yesterday and feel lightheaded. Is that expected?"
        )
        print(result.final_output)


asyncio.run(main())
```

The lead calls the `med_safety` specialist as a tool, gets its result, and
answers. `model: inherit` lets the specialist reuse the lead's model.

### 3. Full programmatic control

When you need real Python tools, tuned model calls, and run-loop limits, build
the descriptor directly. A plain function becomes a tool — no decorators or
registration:

```python
from agentlane.harness import AgentDescriptor, Runner
from agentlane.models import Tools


def lookup_medication(name: str) -> str:
    """Return basic guidance for a medication by name."""
    return f"{name}: take with food; report severe dizziness or fainting to your care team."


async def main() -> None:
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="care-navigator",
            model=model,
            instructions="Call lookup_medication before advising on a medication. Give one next step.",
            tools=Tools(tools=[lookup_medication]),
            model_args={"reasoning_effort": "low"},
        ),
        runner=Runner(max_turns=6),
    )
    result = await agent.run("Can I keep taking metformin if it upsets my stomach?")
    print(result.final_output)


asyncio.run(main())
```

The markdown agent and this one share the same `DefaultAgent` contract — markdown
is the fast path, the descriptor is the full surface.

### 4. A multi-agent team on a distributed runtime

Attach specialists with `subagents=` and run the team on a distributed runtime.
`DefaultAgent` wires each sub-agent in for you — no manual `as_tool()`. The
coordinator owns the model loop; each specialist runs as an addressed agent the
runtime can place on its own worker:

```python
med_safety = AgentDescriptor(
    name="med-safety",
    model=model,
    tools=None,
    instructions="Flag medication interactions or safety concerns in one sentence.",
)
guidelines = AgentDescriptor(
    name="guidelines",
    model=model,
    tools=None,
    instructions="Cite the relevant care guideline for the symptom in one sentence.",
)


async def main() -> None:
    async with distributed_runtime() as runtime:
        triage = DefaultAgent(
            descriptor=AgentDescriptor(
                name="triage-lead",
                model=model,
                instructions="Send medication questions to `med_safety` and symptom questions to `guidelines`, then give one next step.",
            ),
            subagents=[med_safety, guidelines],
            runtime=runtime,
        )
        result = await triage.run(
            "New chest tightness after a dose increase of my heart medication. What should I do?"
        )
        print(result.final_output)


asyncio.run(main())
```

`subagents=` is the one way to attach sub-agents — markdown paths via
`from_markdown` (step 2) or descriptors here — and `tools=[child.as_tool()]`
stays as a manual escape hatch. The same runtime concepts scale on to background
specialists, pub/sub fan-out, and multi-process workers — see
[Harness Distributed Agents](docs/harness/distributed-agents.md).

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
2. [Markdown Agent Definitions](docs/harness/agent-definitions.md)
3. [Architecture](docs/harness/architecture.md)
4. [Tools](docs/harness/tools.md)
5. [Shims](docs/harness/shims.md)
6. [Compaction](docs/harness/compaction.md)
7. [Skills](docs/harness/skills.md)
8. [Distributed Agents](docs/harness/distributed-agents.md)

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
