# Harness Agents

Date: 2026-04-05
Status: Phase 4 baseline

## What The Default Agent Owns

The default harness `Agent` builds on `Task` and owns:

1. agent identity and descriptive metadata,
2. the canonical model client used by the default runner,
3. low-level model call settings via `model_args` and `schema`,
4. canonical `Tools` configuration shared by model calls and later harness phases,
5. persisted run state for multi-turn continuation,
6. queued inbound run inputs, and
7. delegation of each turn to a stateless `Runner`.

The static portion of that configuration is grouped into `AgentDescriptor` and
passed to the `Agent` constructor as one value instead of duplicating the same
fields across both `Agent` and lifecycle state. That includes instructions,
tools, model, model args, schema, and the other descriptive fields.

Recovered run state is separate from that static descriptor. If you need crash
recovery or want to resume an existing conversation, pass `run_state` when
constructing or binding the concrete agent instance.

## Runtime Guarantee Preserved

The harness preserves the current runtime rule that only one handler for a given
`AgentId` is active at a time.

That means "new input while running" is modeled as:

1. append the new input to the agent's internal queue,
2. finish the currently running runner turn, and then
3. process the queued input on the next loop turn before the agent becomes idle.

The harness does not introduce concurrent re-entry for the same `AgentId`.

## Drain Strategy

Queued run inputs are drained one runner invocation at a time.

That means:

1. the lifecycle pops one queued input,
2. creates the next working `RunState` for that input,
3. invokes the runner once, and then
4. repeats for the next queued input if any remain.

The lifecycle does not batch all outstanding queued inputs into one larger runner call.

## Conversation Lifecycle

### New `AgentId`

When a new agent instance receives its first input:

1. a new `RunState` is initialized,
2. the original input is stored on that state, and
3. the runner later builds the concrete model request from instructions plus run state.

### Recovered `AgentId`

When an agent instance is created with preloaded `run_state`:

1. that recovered run state becomes the lifecycle baseline,
2. the next inbound input is merged into a copied working state, and
3. the runner continues from that state on the next invocation.

### Existing idle `AgentId`

When an existing idle agent receives another input:

1. the prior `RunState` is preserved,
2. the new input is appended to continuation history in a copied working state, and
3. the runner is invoked again with that updated state.

### Existing running `AgentId`

When a new input arrives while the agent is already executing a runner turn:

1. the input is queued internally, and
2. it is processed on the next turn immediately after the active turn completes.

## Input Shape

The public runtime input is `RunInput`.

That means the default harness agent accepts exactly:

1. `str`
2. `list[object]`
3. `RunState`

The agent exposes one `@on_message` handler per concrete runtime payload type so
it stays compatible with the runtime's exact-type dispatch rules.

At the developer surface, those raw input items can still include higher-level
primitives such as `agentlane.models.PromptSpec`. The agent does not normalize
them itself. The runner resolves them later when building the concrete model request.

## Run State

`RunState` is the persisted lifecycle surface for this phase.

It contains only:

1. `original_input`
2. `continuation_history`
3. `responses`
4. `turn_count`

The agent exposes `agent.run_state` as a snapshot suitable for persistence and later recovery.

## Phase Boundary

The Phase 3 lifecycle semantics remain unchanged in Phase 4.

Phase 4 adds three new agent seams here and keeps two boundaries explicit:

1. `AgentDescriptor` is the canonical static agent configuration passed into `Agent`.
2. `agent.model`, `agent.model_args`, `agent.schema`, `agent.tools`, and `agent.instructions` are projections from that descriptor.
3. `MessageDict` construction is no longer part of the public agent boundary. The runner builds model requests internally from `RunState`.
4. Tool execution behavior is still not implemented here.
5. Handoffs and sub-agent delegation are still not implemented here, so agents do not expose `as_tool` yet.
