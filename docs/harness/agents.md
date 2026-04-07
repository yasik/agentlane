# Harness Agents

## What The Default Agent Owns

The default harness `Agent` builds on `Task` and owns:

1. one static `AgentDescriptor`
2. one `AgentLifecycle` for queued input and persisted `RunState`
3. one reusable `Runner`
4. optional `RunnerHooks`
5. model-visible tools and handoffs for the current instance

The agent does not build model requests itself. It owns lifecycle and
delegation wiring; the runner owns the model loop.

## `AgentDescriptor`

`AgentDescriptor` is the canonical static specification for one agent.

It includes:

1. `name`
2. `description`
3. `model`
4. `instructions`
5. `model_args`
6. `schema`
7. `tools`
8. `handoffs`
9. `default_handoff`
10. reserved `skills`, `context`, and `memory`

This separation matters:

1. descriptor data is static configuration
2. `RunState` is mutable conversation state
3. recovery should restore `RunState`, not mutate shared descriptor values

## Input Surface

The default agent accepts `RunInput`:

1. `str`
2. `list[object]`
3. `RunState`

Typical usage:

1. send a `str` for a normal user turn
2. send `list[object]` for richer prompt input or replayable items
3. send `RunState` to resume a previously persisted conversation

The agent keeps that input generic. It does not collapse it into model message
dictionaries at the public boundary.

## Runtime Guarantee

The harness preserves the runtime rule that only one handler for a given
`AgentId` is active at a time.

That means:

1. one inbound input may actively execute the runner
2. later inputs for the same `AgentId` are queued
3. the queue drains in FIFO order
4. the agent never re-enters concurrently for the same id

Queued inputs are processed one runner invocation at a time. The lifecycle does
not batch all pending inputs into one larger run.

## Run State And Recovery

`Agent.run_state` returns a snapshot of the current resumable state.

That state currently includes:

1. `original_input`
2. `continuation_history`
3. `responses`
4. `turn_count`

To resume after a crash or restart, bind or construct the agent with
`run_state=...`. The next turn continues from that restored state.

## Tool Surface

`Agent.tools` is the model-visible tool catalog for the current instance.

It is composed from:

1. the resolved base tool set from `AgentDescriptor.tools`
2. predefined handoff tool specs from `AgentDescriptor.handoffs`
3. the optional generic handoff tool from `AgentDescriptor.default_handoff`

The base tool rules are:

1. omitted tools inherit the parent tool set
2. explicit `Tools(...)` overrides the parent
3. explicit `None` disables base tools

Handoffs are exposed to the model like tools, but the runner still intercepts
them with transfer semantics instead of normal tool execution.

## Agent-As-Tool

An agent can expose itself as declarative tool metadata through:

1. `AgentDescriptor.as_tool(...)`
2. `Agent.as_tool(...)`

That API returns tool schema, not an eager execution wrapper.

The important boundary is:

1. `as_tool(...)` describes the tool the model can call
2. the runner decides how to execute that tool call through runtime messaging

For predefined sub-agents, the delegated child receives exactly the validated
payload described by the declared tool args model. There is no reserved
framework-level `task` field on this path unless the args model itself defines
one.

## Handoffs

The harness supports two transfer-style handoff surfaces:

1. predefined transfer targets through `AgentDescriptor.handoffs`
2. one generic fresh-agent handoff through `AgentDescriptor.default_handoff`

Handoff is not a subroutine call. It transfers the run to another agent.

When a handoff happens:

1. the triggering assistant handoff turn is preserved in transferred history
2. a synthetic transfer acknowledgement is recorded
3. an optional delegation message is appended
4. the downstream agent continues the run
5. the original agent does not resume

The downstream handoff agent keeps its own instructions. The framework does not
inject a second handoff-specific system prompt on top.
