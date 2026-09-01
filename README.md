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
║                the open runtime for persistent, addressable agents                 ║
║                                                                                    ║
║          identity • inbox • state • delivery • local → distributed                 ║
║                                                                                    ║
╚════════════════════════════════════════════════════════════════════════════════════╝
```

**AgentLane gives an AI agent a stable address, an inbox, and state that lasts
longer than a single run.** You can swap the model, the harness, the process,
or the machine underneath it, and the agent is still the same agent.

The idea comes from a simple observation: a single agent loop stops scaling
once you need background jobs, long-running work, human review, specialist
agents, and plain deterministic services working together. Those are not
prompt problems. They are distributed systems problems, and the fix is to
treat agents like members of an organization, each with an identity, an inbox,
and state of its own. The full argument is in
[Distributed Agents Are What Make AI Systems Work Like Organizations](https://www.yasik.org/writings/distributed-agents).

![PyPI](https://img.shields.io/pypi/v/agentlane.svg)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![npm](https://img.shields.io/npm/v/%40agentlanejs%2Fprocess-bridge.svg?label=process-bridge)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

[See it](#see-it) · [Why](#why-agentlane) · [Install](#install) · [Quick start](#quick-start) · [Other harnesses](#connect-other-harnesses) · [Layers](#layers) · [Docs](docs/README.md) · [Examples](examples/README.md) · [Changelog](CHANGELOG.md)

## See it

Two processes, two models, one agent. The agent keeps its address and its
memory in a file. Everything else changes underneath it.

`care_navigator.md`:

```markdown
---
name: care-navigator
description: Follows a patient's medication questions over time.
---
You are a concise patient care navigation agent. Remember what you were told
about a patient and give one clear next step.
```

`monday.py`:

```python
import asyncio
import os

from agentlane_openai import ResponsesClient

from agentlane.harness.agents import DefaultAgent
from agentlane.models import Config


async def main() -> None:
    agent = DefaultAgent.from_markdown(
        "care_navigator.md",
        model=<model_client_openai>,
        state_path=".agentlane/care-navigator.json",
    )
    await agent.run("Patient 4471 started lisinopril today. Keep an eye on it.")
    print(agent.agent_id)


asyncio.run(main())
```

`tuesday.py`, a new process with a different model:

```python
async def main() -> None:
    agent = DefaultAgent.from_markdown(
        "care_navigator.md",
        model=<model_client_claude>,
        state_path=".agentlane/care-navigator.json",
    )
    result = await agent.run("Patient 4471 feels lightheaded this morning. What now?")
    print(agent.agent_id)  # same address as Monday
    print(result.final_output)  # knows about yesterday's lisinopril
```

Nothing was passed between the two scripts except the state file. The agent's
identity, conversation, and turn count live with the agent. The model, the
process, and the run loop are supplied fresh each time. Swap `state_path=`
for your own `StateStore` when a file is not enough, and bind the agent to a
distributed runtime when it needs to live on a worker. Same agent, same
address.

## Why AgentLane

Most agent frameworks start with a prompt, a few tools, and a loop. AgentLane
starts one layer lower. Every agent has an identity, a job, permissions, tools,
state, and a place in the system. Some agents are long-lived employees with
ongoing responsibilities. Others are temporary contractors that fan out, do
their part, and go away. Both talk to each other the same way: through
addressed messages.

That gives you three things:

1. **State stays with whoever owns it.** Review status, user preferences, and
   conversation history belong to the agent or task that owns them, not to one
   chat transcript.
2. **You can see what happened.** Which agent got the task, which worker ran it,
   which messages and tools were involved, and what came back.
3. **Local grows into distributed.** The agent you run in one process today
   can run on a pool of workers tomorrow. The way agents talk to each other
   does not change.

## Install

```bash
uv add agentlane
```

Add a provider or integration as an extra:

```bash
uv add "agentlane[openai]"            # OpenAI Responses client (default provider)
uv add "agentlane[litellm]"           # any model LiteLLM supports
uv add "agentlane[claude-agent-sdk]"  # Claude Agent SDK coworkers
uv add "agentlane[braintrust]"        # export traces to Braintrust
```

Working from a checkout of this repo:

```bash
uv sync --all-extras
```

## Quick start

An agent is a markdown file. The frontmatter is the config and the body is the
system prompt. The two steps below build on each other: the same agent goes
from a single file to a team on a distributed runtime. The model and the run
loop never change.

Both steps share one model client:

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

One file, one `run(...)`. By default it runs on a local single-threaded
runtime, and every run leaves resumable state on the agent. Add `state_path=`
to keep that state across processes, as in [See it](#see-it).

### 2. A team on a distributed runtime

Add a specialist with `subagents=` and bind both to a distributed runtime. The
specialist becomes an addressed agent the lead can delegate to, and the runtime
can later move it onto its own worker.

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

The lead calls `med_safety` as a tool, gets the note back, and answers.
`model: inherit` lets the specialist reuse the lead's model.

Markdown is the fast path. When you need real Python tools, tuned model calls,
or run-loop limits, build an `AgentDescriptor` directly. Any plain function can
be a tool, with no decorator or registration. See
[Default Agents](docs/harness/default-agents.md),
[Markdown Agent Definitions](docs/harness/agent-definitions.md), and
[Distributed Agents](docs/harness/distributed-agents.md).

## Connect other harnesses

An AgentLane address does not need an AgentLane model loop behind it. Anything
bound to the runtime as a `Task` can receive addressed work.

**Claude Agent SDK coworker.** Give a Claude identity an address on the runtime.
A native AgentLane agent sends it a task the usual way and uses the reply in
its own run.

```python
from agentlane_claude_agent_sdk import ClaudeAgent
from agentlane.messaging import AgentId

claude = AgentId.from_values("claude-sdk", "analyst")
ClaudeAgent.bind(runtime, claude)

outcome = await runtime.send_message(
    "Summarize the interaction risks for lisinopril.",
    sender=lead_id,
    recipient=claude,
)
```

Every addressed task starts a fresh SDK session. See
[Harness Tasks](docs/harness/tasks.md) and the
[coworker example](examples/harness/claude_agent_sdk_coworker/).

**TypeScript app shells.** `@agentlanejs/process-bridge` starts a local Python
AgentLane backend as a child process and streams typed session events over
stdio. See [Process Bridge](docs/process-bridge/README.md).

## Layers

Use them together or pick the one you need.

| Layer | What it does | Start here |
| --- | --- | --- |
| [Runtime](src/agentlane/runtime/) | agent identity, execution, scheduling, local and distributed workers | [Engine and Execution](docs/runtime/engine-and-execution.md) · [Distributed Runtime](docs/runtime/distributed-runtime-usage.md) |
| [Messaging](src/agentlane/messaging/) | addressed sends, pub/sub, delivery outcomes, per-recipient ordering | [Routing and Delivery](docs/messaging/routing-and-delivery.md) |
| [Models](src/agentlane/models/) | prompt templates, schemas, structured output, native tools, provider clients | [Overview](docs/models/overview.md) · [Prompt Templating](docs/models/prompt-templating.md) |
| [Harness](src/agentlane/harness/) | `DefaultAgent`, markdown definitions, resumable and persistent state, handoffs, sub-agents, shims, skills, compaction | [Default Agents](docs/harness/default-agents.md) · [Architecture](docs/harness/architecture.md) |
| [Transport](src/agentlane/transport/) | wire-safe serialization across process boundaries | [Serialization](docs/transport/serialization.md) |
| [Tracing](src/agentlane/tracing/) | spans and metrics across runtime, model, and harness | [Tracing Overview](docs/tracing/overview.md) |

Provider and integration packages live under [`packages/`](packages/):
`agentlane-openai`, `agentlane-litellm`, `agentlane-claude-agent-sdk`,
`agentlane-braintrust`, `agentlane-process-bridge`, and
`@agentlanejs/process-bridge`.

## Development

```bash
/usr/bin/make format
/usr/bin/make lint
/usr/bin/make tests
```

Run a single test:

```bash
uv run pytest -s -k <test_name>
```

## Contributing

1. Keep changes small and focused.
2. Add or update tests when behavior changes.
3. Update the public docs and examples when the developer-facing surface changes.
4. Make sure formatting, linting, and tests pass before you open a PR.

## License

[MIT](LICENSE)
