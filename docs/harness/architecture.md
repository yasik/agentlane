# Harness Architecture

## Overview

The harness is the higher-level orchestration layer built on top of the
runtime. It keeps runtime delivery semantics intact while adding a reusable
agent loop, tool execution, delegation, and resumable multi-turn state.

At a high level:

1. `agentlane.harness.agents.DefaultAgent` is the ergonomic local wrapper.
2. `Task` is the top-level unit of work.
3. `Agent` builds on `Task` and owns per-agent lifecycle state.
4. `Runner` executes the generic LLM loop for one run.
5. `agentlane.models` remains the model-facing foundation.
6. Provider packages stay thin and adapt request and response traffic into the
   shared model contract.

The key design choice is to keep the public harness boundary run-oriented.
Developers work with descriptors, run input, and run state. The harness itself
decides how those values become canonical model messages.

## System View

```text
Application / caller
        v
+---------------------------+
| harness.agents.          |
| DefaultAgent             |
| local run(...) wrapper   |
+-------------+-------------+
              |
              | send_message(run_input)
              v
+---------------------------+
| RuntimeEngine             |
| routing + instance reuse  |
+-------------+-------------+
              |
              v
+---------------------------+
| Task                      |
| top-level unit of work    |
+-------------+-------------+
              |
              v
+---------------------------+
| Agent                     |
| descriptor + lifecycle    |
+-------------+-------------+
              |
              v
+---------------------------+
| AgentLifecycle            |
| queue + persisted state   |
| RunState ownership        |
+-------------+-------------+
              |
              v
+---------------------------+
| Runner                    |
| stateless loop            |
| request building          |
| tool / handoff control    |
+------+------+-------------+
       |      |
       |      +--------------------+
       |                           |
       v                           v
+--------------+          +-------------------+
| ToolExecutor |          | runtime messaging |
| local tools  |          | delegated agents  |
+--------------+          +---------+---------+
                                      |
                                      v
                            +-------------------+
                            | child Agent       |
                            | tool or handoff   |
                            +-------------------+

              Runner
                |
                v
      +-----------------------+
      | agentlane.models      |
      | prompts, schema,      |
      | tools, ModelResponse  |
      +-----------+-----------+
                  |
                  v
      +-----------------------+
      | provider client       |
      | thin request/response |
      +-----------+-----------+
                  |
                  v
                LLM
```

## Package Layout

The implementation lives under `src/agentlane/harness`.

Core modules:

1. `_task.py`: the thin top-level task primitive.
2. `_agent.py`: the default harness agent built on top of `Task`.
3. `_lifecycle.py`: agent descriptor types plus per-agent queue and run-state
   management.
4. `_runner.py`: the stateless generic LLM loop.
5. `_handoff.py`: delegation payloads and helper functions shared by handoffs
   and delegated sub-agents.
6. `_tooling.py`: tool inheritance and merge helpers.
7. `_hooks.py`: runner lifecycle hooks.
8. `_run.py`: minimal run-state and result contracts.

Public subpackages:

1. `agents/`: developer-facing local wrappers such as `DefaultAgent`

Reserved subpackages:

1. `context/`
2. `memory/`

Those package seams exist so future context and memory features can grow
without changing the current harness boundary.

## Core Contracts

### `Task`

`Task` is a thin wrapper over the existing runtime model. It does not introduce
its own scheduler or dispatcher.

### `agents.DefaultAgent`

`agentlane.harness.agents.DefaultAgent` is the ergonomic local wrapper.

It owns:

1. descriptor resolution
2. optional local runtime provisioning
3. optional runner provisioning
4. persisted `RunState` across repeated `run(...)` calls

It does not replace the runtime-facing `Agent`. It binds and routes through it.

### `AgentDescriptor`

`AgentDescriptor` is the canonical static configuration for one agent. It owns:

1. descriptive metadata such as `name` and `description`
2. the model client
3. `instructions`
4. `model_args`
5. `schema`
6. `tools`
7. predefined `handoffs`
8. `default_handoff`
9. reserved `skills`, `context`, and `memory` seams

The descriptor is static. Mutable conversation state is kept in `RunState`.

### `RunInput`

The default harness agent accepts:

1. `str`
2. `list[object]`
3. `RunState`

That keeps the public input surface simple while still allowing rich prompt
input, replay, and resume.

### `RunState`

`RunState` is the minimal resumable state for one agent run:

1. `original_input`
2. `continuation_history`
3. `responses`
4. `turn_count`

It is intentionally small. The harness does not currently expose a larger event
log, approval state, or actor graph.

### `RunResult`

`RunResult` is the minimal final result returned by the default runner:

1. `final_output`
2. `responses`
3. `turn_count`
4. `run_state`

The harness keeps raw `ModelResponse` values intact instead of wrapping them in
new harness-specific response models.

## Request Ownership

The public harness boundary is not `MessageDict`-oriented.

Instead:

1. `agents.DefaultAgent` accepts local `run(...)` calls and forwards them into
   the runtime-facing harness.
2. `Agent` accepts `RunInput`.
3. `AgentLifecycle` persists and queues `RunState`.
4. `Runner` converts `instructions + original_input + continuation_history`
   into canonical model requests.
5. Provider clients receive canonical model messages and return canonical
   `ModelResponse` values.

This keeps message normalization inside the runner and prevents low-level model
wire types from leaking into the public harness API.

## Lifecycle Model

Each concrete `AgentId` preserves the runtime guarantee of a single active
handler at a time.

When new input arrives:

1. if the agent is idle, the lifecycle creates a private working copy of the
   current `RunState` and invokes the runner
2. if the agent is already running, the lifecycle appends the input to an
   internal FIFO queue
3. queued inputs are drained one runner invocation at a time

The lifecycle uses copy-on-write state handling so failed turns do not corrupt
the persisted baseline.

```text
new input for AgentId
        |
        v
+---------------------------+
| AgentLifecycle.enqueue    |
+-------------+-------------+
              |
      +-------+--------+
      | already active?|
      +---+--------+---+
          |        |
        yes        no
          |        |
          v        v
 +----------------+  +------------------------+
 | append to FIFO |  | copy persisted state   |
 | pending queue  |  | create working state   |
 +--------+-------+  +-----------+------------+
          |                      |
          +----------+-----------+
                     |
                     v
           +--------------------+
           | Runner.run(...)    |
           +---------+----------+
                     |
                     v
           +--------------------+
           | success?           |
           +----+----------+----+
                |          |
               no         yes
                |          |
                v          v
      +----------------+  +--------------------+
      | keep baseline  |  | persist new state  |
      | unchanged      |  | and drain next     |
      +----------------+  | queued input       |
                          +--------------------+
```

## Runner Ownership

`Runner` owns the generic loop for one run:

1. build the next model request
2. call the model
3. accumulate the raw `ModelResponse`
4. classify the model output
5. either stop, execute tools, or transfer control
6. continue until a terminal answer is produced

The runner also owns:

1. tool execution
2. model-facing tool visibility
3. optional outer retry policy
4. hook dispatch
5. handoff interception
6. delegated agent-as-tool execution

## Delegation Model

The harness supports two distinct delegation patterns that both appear to the
model as tool choices.

### First-class handoff

Handoff transfers control to another agent.

Semantics:

1. the model emits a handoff tool call
2. the runner intercepts it specially
3. full conversation history is transferred to the downstream agent
4. the triggering handoff turn is preserved in transferred history
5. an optional delegation message is appended
6. the downstream agent continues the run
7. the original agent does not resume

Predefined transfer targets live in `AgentDescriptor.handoffs`. A generic
fresh-agent path can also be exposed through `DefaultHandoff`.

### Agent-as-tool

Agent-as-tool behaves like a subroutine call.

Semantics:

1. the model emits a normal tool call with a JSON schema
2. the runner validates arguments into a Pydantic model
3. that structured payload is sent to another agent through runtime messaging
4. the child agent runs to completion
5. the child result is converted into a tool result string
6. the caller agent continues its own loop

Predefined sub-agents are exposed through `AgentDescriptor.as_tool(...)` or
`Agent.as_tool(...)`. A generic spawned-helper path is exposed as one model
tool through `DefaultAgentTool`.

## Tool Visibility And Inheritance

`AgentDescriptor.tools` uses an inheritance-aware policy:

1. omitted tools inherit the parent tool set
2. explicit `Tools(...)` overrides the parent tool set
3. explicit `None` disables tools for that agent

The concrete `Agent.tools` property merges that resolved tool catalog with any
model-visible handoff tools for the current instance.

## Model And Provider Boundary

The harness reuses the shared model contract from `agentlane.models`:

1. `ModelResponse`
2. `ToolCall`
3. `MessageDict`
4. `Tools`
5. `OutputSchema`

Provider adapters stay thin:

1. they accept canonical request input
2. they return canonical model responses
3. they do not run their own tool loop
4. they do not own handoff logic

## Current Scope

The current harness intentionally does not define:

1. memory persistence semantics
2. approval interrupts
3. event-log envelopes
4. distributed orchestration beyond the existing runtime messaging contract
5. provider-specific harness abstractions beyond `agentlane.models`
