# Harness Tasks

Tasks are the smallest harness abstraction. They are useful when you want a
clear place for application orchestration logic, but you do not want to opt
into the default model loop.

[`Task`](../../src/agentlane/harness/_task.py) is a thin layer over
[`BaseAgent`](../../src/agentlane/runtime/_base_agent.py). That is why it feels
familiar if you already understand the runtime: it keeps the same message
handlers, the same delivery model, and the same instance reuse rules.

## When To Use A Task

Use a task when the work is application logic rather than model-driven
reasoning. Typical examples include:

1. coordinating multiple runtime recipients
2. calling databases, services, or webhooks
3. shaping a workflow before or after a model-backed agent runs

If the code needs the runtime but not an LLM loop, a task is usually the right
level.

## Why Tasks Stay Thin

Tasks do not introduce a second scheduler or a second execution model. They
exist mostly to make intent clearer and to provide a few small registration
helpers.

That means the important runtime ideas still apply:

1. one concrete `AgentId` means one reusable instance
2. message handlers are still declared with `@on_message`
3. orchestration still uses `send_message(...)` and `publish_message(...)`

## Registration And State

There are two common patterns:

1. `Task.register(...)` when the runtime should create instances lazily
2. `Task.bind(...)` when you want to create and bind one concrete instance

The choice is really about state ownership.

Use registration when state should follow the normal `AgentId` reuse model. Use
binding when you already have a concrete stateful instance and want that exact
instance tied to one identity.

## External Agent Tasks

A `Task` can give an external agent harness an AgentLane identity. The optional
Claude Agent SDK package provides one small integration:

```bash
uv sync --extra claude-agent-sdk
```

```python
from agentlane_claude_agent_sdk import ClaudeAgent

ClaudeAgent.bind(runtime, claude_agent_id)
```

`ClaudeAgent` accepts addressed text messages. It starts one fresh Claude Agent
SDK query for each message and returns the final text through the AgentLane
delivery result. Its default options disable tools, skills, settings sources,
and MCP servers. They also limit the query to one turn. Explicit SDK options
can change those settings, but session continuation options are rejected.

SDK failures and invalid terminal results become AgentLane handler errors. The
SDK child process inherits the parent process environment, and message content
is sent to Anthropic. Run it with only the required credentials and use data
that is safe to send to that provider.

The runnable
[`claude_agent_sdk_coworker`](../../examples/harness/claude_agent_sdk_coworker/)
example shows a native AgentLane agent sending an addressed task to Claude and
using the returned text to complete its own run. The result returns through the
delivery call. The example does not show a second addressed message from Claude
or shared session continuity between messages.

## A Useful Rule Of Thumb

If you start adding prompt construction, model configuration, or tool policies
to a task, it is probably time to move up to the default harness
[`Agent`](../../src/agentlane/harness/_agent.py).
