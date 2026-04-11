# Harness Agents

This page documents the runtime-facing
[`agentlane.harness.Agent`](../../src/agentlane/harness/_agent.py).

For the higher-level local wrapper that exposes `run(...)` directly, see
[Harness Default Agents](./default-agents.md).

The runtime-facing harness `Agent` is where a long-lived agent definition meets
one concrete addressed conversation. It keeps static configuration in
[`AgentDescriptor`](../../src/agentlane/harness/_lifecycle.py), persists the
conversation as [`RunState`](../../src/agentlane/harness/_run.py), and hands
execution to the [`Runner`](../../src/agentlane/harness/_runner.py).

That split is what makes an addressed agent feel stable across turns. The
descriptor says what kind of agent this is. The run state says where one
concrete interaction currently stands.

## What The Runtime-Facing Agent Owns

1. the descriptor that defines instructions, tools, schema, and handoffs
2. the lifecycle that queues later inputs while one run is active
3. the runner that executes each run

The runner does not own long-lived agent state. The lifecycle does not own the
model loop. The agent is the place where those pieces meet.

## Input And Recovery

The public input surface is intentionally small. A runtime-facing agent
accepts:

1. a `str` for a normal user turn
2. a `list[object]` for richer multi-item input
3. a [`RunState`](../../src/agentlane/harness/_run.py) when resuming an
   existing conversation

That is enough to support normal chat-like turns, richer prompt input, and
recovery after a restart.

When you ask an agent for its current run state through
[`Agent.run_state`](../../src/agentlane/harness/_agent.py), you get a snapshot
of the resumable state for that concrete agent instance.

## Tools And Handoffs

An agent's model-visible tool surface is built from three sources:

1. its ordinary tool set
2. predefined handoff targets
3. an optional generic handoff path

That design matters because handoffs are intentionally model-visible. From the
model's point of view, a handoff looks like a tool choice. From the framework's
point of view, it is a transfer of control to another agent.

The same distinction shows up in agent-as-tool usage. When an agent is exposed
through `Agent.as_tool(...)` or `AgentDescriptor.as_tool(...)`, the model sees a
tool schema, but the framework still executes that tool call by routing work to
another agent run.

## Concurrency Per Agent

The harness preserves the runtime rule that a single concrete `AgentId` is not
re-entered concurrently.

That means:

1. one input may actively run
2. later inputs for the same agent are queued
3. queued inputs are drained in FIFO order

This is what makes resumable state practical. The framework does not need to
guess how to merge multiple overlapping runs for the same agent instance.

## When To Read The Runner Docs

If you are trying to understand how tools are executed, how a handoff changes
control flow, or how the next model request is built, move on to
[Harness Runner](./runner.md). The agent owns the pieces. The runner explains
their behavior inside a run.
