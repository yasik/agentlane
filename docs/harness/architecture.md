# Harness Architecture v1

Date: 2026-04-02
Status: Phase 1 baseline

## Goal

Freeze the initial harness package shape and canonical data contracts before
implementing lifecycle, runner, tooling, or handoff behavior.

## Decisions

1. The implementation target is `src/agentlane/harness`.
2. The `srs/agentlane/harness` path from the design note is treated as a typo for this repository unless explicitly revisited in review.
3. The harness builds on top of `agentlane.runtime.BaseAgent` and existing runtime messaging semantics.
4. The harness reuses the existing OpenAI-native aliases already exposed by `agentlane.models`:
   - `MessageDict` for conversation input items,
   - `ModelResponse` for canonical model responses,
   - `ToolCall` for canonical tool-call records.
5. Provider-specific clients continue adapting into those canonical shapes instead of introducing harness-specific response wrapper models.

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

1. `Agent` owns conversation history, descriptive metadata, and queued user turns.
2. `UserMessage` is the public runtime payload for one user turn, and `Agent.user_message(content)` is the convenience constructor.
3. New conversations begin with the configured system prompt when present.
4. Idle agents continue from prior history on the next inbound message.
5. Running agents queue additional user turns for the next loop turn instead of allowing concurrent re-entry for the same `AgentId`.
6. Queued user turns are drained one runner turn at a time rather than being batch-appended into one larger runner invocation.

## Non-Goals for Phase 1

1. No agent lifecycle execution semantics yet
2. No tool loop implementation yet
3. No handoff implementation yet
4. No memory persistence semantics yet
5. No provider-specific harness abstractions beyond the existing `agentlane.models` aliases
