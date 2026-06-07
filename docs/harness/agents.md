# Harness Agents

This page documents the lower-level addressed
[`agentlane.harness.Agent`](../../src/agentlane/harness/_agent.py).

For the higher-level local agent interface that exposes `run(...)` directly, see
[Harness Default Agents](./default-agents.md).

This lower-level harness `Agent` is the type you bind to a runtime and address
by `AgentId`. It is where a long-lived agent definition meets one concrete
conversation. It keeps static configuration in
[`AgentDescriptor`](../../src/agentlane/harness/_lifecycle.py), persists the
conversation as [`RunState`](../../src/agentlane/harness/_run.py), and hands
execution to any bound shims and then to the
[`Runner`](../../src/agentlane/harness/_runner.py).

That split is what makes an addressed agent feel stable across turns. The
descriptor says what kind of agent this is. The run state says where one
concrete interaction currently stands.

`AgentDescriptor.name` is always populated. Provide one when the application
has a stable human-readable name for logs, tracing, or tool correlation. If it
is omitted or blank, the framework generates a Faker-backed adjective-noun
fallback name, such as `deterministic-butterfly`, at descriptor construction so
the runner does not need defensive name handling later.

## What The Lower-Level Agent Owns

1. the descriptor that defines instructions, tools, schema, handoffs, and
   optional shims
2. the lifecycle that queues later inputs while one run is active
3. any bound shims declared by the descriptor
4. the runner that executes each run

The runner does not own long-lived agent state. The lifecycle does not own the
model loop. The agent is the place where those pieces meet.

## Input And Recovery

The public input surface is intentionally small. This lower-level harness agent
accepts:

1. a `str` for a normal user turn
2. a list of supported run-history items for richer multi-item input
3. a [`RunState`](../../src/agentlane/harness/_run.py) when resuming an
   existing conversation

That is enough to support normal chat-like turns, richer prompt input, and
recovery after a restart.

Supported run-history items include:

1. canonical message dicts,
2. prior `ModelResponse` values,
3. `PromptSpec` values, and
4. user-side content values such as strings, JSON-like values, or Pydantic
   models.

When you ask an agent for its current run state through
[`Agent.run_state`](../../src/agentlane/harness/_agent.py), you get a snapshot
of the resumable state for that concrete agent instance.

## Tools And Handoffs

An agent's model-visible tool surface is built from three sources:

1. direct descriptor tools resolved from `AgentDescriptor.tools`
2. tools contributed by bound shims such as `HarnessToolsShim`
3. predefined handoff targets and an optional generic handoff path

That design matters because handoffs are intentionally model-visible. From the
model's point of view, a handoff looks like a tool choice. From the framework's
point of view, it is a transfer of control to another agent.

The same distinction shows up in agent-as-tool usage. When an agent is exposed
through `Agent.as_tool(...)` or `AgentDescriptor.as_tool(...)`, the model sees a
tool schema, but the framework still executes that tool call by routing work to
another agent run.

The base `agent` tool is the generic spawned-helper version of agent-as-tool.
It lets the model create a fresh helper for a delegated task without inheriting
the parent conversation or parent system prompt. The helper name is used for
logging and tracing; the task text is the instruction the helper receives.

That generic helper tool is
[`DefaultAgentTool`](../../src/agentlane/harness/_lifecycle.py), exported from
`agentlane.harness`. It defaults to the name `agent` and lets you configure the
helper's `description`, `instructions`, `model`, `model_args`, `output_schema`,
and `tools`. It is normally wired through the tools shim rather than constructed
directly: `base_harness_tools(...)` (via `HarnessToolsShim`) contributes it as a
model-visible tool with prompt guidance.

The generic handoff path is configured with
[`DefaultHandoff`](../../src/agentlane/harness/_lifecycle.py), also exported from
`agentlane.harness`, and enabled through `AgentDescriptor.default_handoff`. It
defaults to the name `handoff` and exposes `description`, `instructions`,
`model`, `model_args`, `schema`, and `tools` for the spawned transfer agent.

```python
from agentlane.harness import AgentDescriptor, DefaultHandoff

descriptor = AgentDescriptor(
    name="Care Navigation",
    model=model,
    instructions="You are a patient care navigation agent.",
    default_handoff=DefaultHandoff(
        instructions="You are a focused follow-up agent.",
    ),
)
```

The configured handoff is surfaced at runtime through the
`Agent.default_handoff` property.

Default spawned helpers inherit two things through different paths. Direct
descriptor tools are resolved by `ToolConfig`: `INHERIT_TOOLS` keeps parent
tools, `OVERRIDE_TOOLS` replaces them, and `RESTRICT_TOOLS.only(...)` filters
them by name before adding child-local tools. Descriptor shims are inherited as
shim definitions and then bind for the child agent. If those inherited shims
include `HarnessToolsShim(base_harness_tools(...))`, the helper gets the same
configured first-party tools and prompt guidance through the normal
prepared-turn path. Those shim-contributed tools are not direct descriptor
tools, but they still become model-visible for the prepared turn.

Predefined handoffs are another category again. The model sees a handoff schema
like a tool, but the runner intercepts the call and transfers control instead
of executing a normal tool handler.

## Concurrency Per Agent

The harness preserves the runtime rule that a single concrete `AgentId` is not
re-entered concurrently.

That means:

1. one input may actively run
2. later inputs for the same agent are queued
3. queued inputs are drained in FIFO order

This is what makes resumable state practical. The framework does not need to
guess how to merge multiple overlapping runs for the same agent instance.

## Shims

If the descriptor includes shims, the lifecycle binds them once for the
concrete agent instance before the first queued run starts.

Those bound sessions are then reused on later runs for the same addressed
agent. This allows shims to keep private in-memory state per agent instance
while still writing resumable state into the persisted `ShimState` stored on
`RunState.shim_state` when needed.

For the actual shim contract, see [Harness Shims](./shims.md).

## When To Read The Runner Docs

If you are trying to understand how tools are executed, how a handoff changes
control flow, or how the next model request is built, move on to
[Harness Runner](./runner.md). The agent owns the pieces. The runner explains
their behavior inside a run.
