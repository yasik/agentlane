# Harness Architecture v1

Date: 2026-04-05
Status: Phase 4 baseline

## Goal

Freeze the initial harness package shape and canonical data contracts before
implementing lifecycle, runner, tooling, or handoff behavior.

## Decisions

1. The implementation target is `src/agentlane/harness`.
2. The `srs/agentlane/harness` path from the design note is treated as a typo for this repository unless explicitly revisited in review.
3. The harness builds on top of `agentlane.runtime.BaseAgent` and existing runtime messaging semantics.
4. The harness reuses the existing OpenAI-native aliases already exposed by `agentlane.models`:
   - `ModelResponse` for canonical model responses,
   - `ToolCall` for canonical tool-call records.
5. `MessageDict` remains the canonical model-call wire shape, but it is now a runner-internal concern rather than a public harness boundary.
6. Provider-specific clients continue adapting into those canonical shapes instead of introducing harness-specific response wrapper models.

## Package Boundaries

Root package:

1. `src/agentlane/harness/_task.py`
   Defines the top-level task primitive that sits upstream of LLM-driven agents,
   including small runtime-aligned registration helpers.
2. `src/agentlane/harness/_agent.py`
   Defines the default harness agent type built on top of `Task`.
3. `src/agentlane/harness/_runner.py`
   Defines the stateless runner contract.
4. `src/agentlane/harness/_hooks.py`
   Defines runner lifecycle hooks.
5. `src/agentlane/harness/_lifecycle.py`
   Reserved for agent-loop state transitions and queueing behavior.
6. `src/agentlane/harness/_handoff.py`
   Reserved for delegation and sub-agent handoffs.
7. `src/agentlane/harness/_tooling.py`
   Reserved for tool visibility and tool-loop behavior.
8. `src/agentlane/harness/_skills.py`
   Reserved for skill descriptors and later loading behavior.

Sub-packages:

1. `src/agentlane/harness/context`
   Reserved for conversation, mailbox, and session primitives.
2. `src/agentlane/harness/memory`
   Reserved for scratchpad, history, and store primitives.

## Phase 1 Public Surface

The public export surface is intentionally small:

1. `Task`
2. `Agent`
3. `Runner`
4. `RunnerHooks`

Everything else remains private until later phases define stable behavior.

## Phase 2 Additions

Phase 2 keeps `Task` thin and runtime-native:

1. `Task.create_factory(...)` builds a runtime-compatible task factory.
2. `Task.register(runtime, agent_type, ...)` registers lazy task factories.
3. `Task.bind(runtime, agent_id, ...)` registers one explicit stateful task instance.
4. `Task.task_id` exposes the bound runtime identity without inventing a second identity model.

## Phase 3 Additions

Phase 3 adds the default agent lifecycle while preserving the runtime execution model:

1. `Agent` owns descriptive metadata and queued next-turn inputs.
2. Running agents queue additional inputs for the next loop turn instead of allowing concurrent re-entry for the same `AgentId`.
3. Queued inputs are drained one runner invocation at a time rather than being batch-appended into one larger runner call.
4. The single-handler-per-`AgentId` runtime guarantee is preserved.

## Phase 4 Additions

Phase 4 turns `Runner` into the default stateless loop engine and resets the
public harness boundary around runs instead of model messages:

1. `AgentDescriptor` is the canonical static agent configuration shared between the public `Agent` surface and lifecycle state.
2. That descriptor now carries instructions, tools, model, model args, schema, and the other descriptive fields.
3. `Agent` projects those values as properties such as `agent.model`, `agent.model_args`, `agent.schema`, `agent.tools`, and `agent.instructions`.
4. The public agent input is now `RunInput = str | list[object] | RunState`.
5. Recovery now uses `RunState` instead of persisted `message_history`.
6. `Runner` accepts `RunState`, builds the concrete `list[MessageDict]` request internally, and returns `RunResult`.
7. `RunState` is intentionally minimal for now: `original_input`, `continuation_history`, `responses`, and `turn_count`.
8. `RunResult` is intentionally minimal for now: `final_output`, `responses`, and `turn_count`.
9. `RunnerHooks.on_agent_start` and `RunnerHooks.on_agent_end` now observe run-level values rather than message-history payloads.
10. `PromptSpec` remains the developer-facing typed prompt input for both instructions and user-side input items, but only the runner resolves it into model messages.
11. `Runner` accumulates raw `ModelResponse` values across turns without wrapping them in new harness-specific response models.
12. Tool-calling responses still fail fast with `ModelBehaviorError` until Phase 5 adds tool execution.
13. Runner-level retries are optional and reuse `agentlane.models.retry_on_errors`.

## Current Non-Goals

1. No tool loop implementation yet
2. No handoff implementation yet
3. No memory persistence semantics yet
4. No provider-specific harness abstractions beyond the existing `agentlane.models` aliases
5. No event-log, approval-state, or resumable interruption envelope until a later phase needs them
