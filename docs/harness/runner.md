# Harness Runner

## What The Runner Owns

`Runner` is the stateless default loop engine for harness agents.

The runner is used both by:

1. the runtime-facing `agentlane.harness.Agent`, and
2. the higher-level local `agentlane.harness.agents.DefaultAgent`, which wraps
   that lower-level agent for a simpler `run(...)` surface.

For one run it owns:

1. building the next model request from `RunState`
2. calling the configured model
3. recording raw `ModelResponse` values
4. executing tools
5. intercepting handoffs
6. routing delegated sub-agents
7. enforcing loop and tool limits
8. returning the final `RunResult`

The runner is reusable. It holds configuration, not per-conversation state.

## Runner And Lifecycle Flow

```text
queued run input
      |
      v
+---------------------------+
| DefaultAgent.run(...)     |
| optional local wrapper    |
+-------------+-------------+
              |
              v
+---------------------------+
| AgentLifecycle            |
| passes working RunState   |
+-------------+-------------+
              |
              v
+---------------------------+
| Runner.run                |
+-------------+-------------+
              |
              v
+---------------------------+
| turn += 1                 |
| enforce max_turns         |
+-------------+-------------+
              |
              v
+---------------------------+
| build request             |
| instructions + history    |
+-------------+-------------+
              |
              v
+---------------------------+
| call model                |
+-------------+-------------+
              |
              v
+---------------------------+
| append ModelResponse      |
| to RunState.responses     |
+-------------+-------------+
              |
              v
      +-------+--------+------------------+
      |                |                  |
      v                v                  v
+-------------+ +--------------+ +----------------+
| tool calls? | | handoff call?| | final answer?  |
+------+------+ +------+-------+ +--------+-------+
       |               |                  |
      yes             yes                yes
       |               |                  |
       v               v                  v
+--------------+ +----------------+ +----------------+
| execute tool  | | transfer state | | extract final  |
| batch         | | to child agent | | output         |
+------+-------+ +-------+--------+ +--------+-------+
       |                 |                   |
       v                 v                   v
+--------------+ +----------------+ +----------------+
| append tool   | | child returns  | | return         |
| result msgs   | | RunResult      | | RunResult      |
+------+-------+ +----------------+ +----------------+
       |
       v
   next turn
```

## Loop Shape

At a high level the loop is:

1. increment `turn_count`
2. build the next model request from `instructions + original_input + continuation_history`
3. call the model
4. append the raw response to `RunState.responses`
5. inspect the response
6. if it contains tool calls, execute them and continue
7. if it contains a handoff, transfer the run
8. otherwise extract the final answer and return

The lifecycle owns queueing and persistence. The runner owns the loop inside one
working run state.

```text
continuation_history
        |
        v
+---------------------------+
| build canonical messages  |
+-------------+-------------+
              |
              v
+---------------------------+
| model response            |
+-------------+-------------+
              |
              v
 assistant tool calls?
      |
   +--+-----------------------------+
   |                                |
  no                               yes
   |                                |
   v                                v
+----------------------+   +-----------------------+
| final assistant turn |   | append assistant turn |
| -> final_output      |   | execute tools         |
+----------------------+   | append tool results   |
                           +-----------+-----------+
                                       |
                                       v
                                  next model turn
```

## Model Boundary

The runner uses the shared `agentlane.models` contract directly.

It expects the current agent to expose:

1. `model`
2. `model_args`
3. `schema`
4. `tools`
5. `instructions`

On each model call the runner forwards:

1. canonical request messages
2. `model_args` through the model call surface
3. `schema`
4. the visible `Tools` configuration for that turn

The runner is also where request normalization happens. `MessageDict` stays an
internal model-call detail, not a public harness input type.

## Result Boundary

The runner returns `RunResult`:

1. `final_output`
2. `responses`
3. `turn_count`
4. `run_state`

The result stays intentionally small. The harness does not currently expose a
larger event log or approval envelope.

## Tool Execution

When the model returns tool calls:

1. the raw assistant response is appended to continuation history
2. each tool call is executed
3. each tool result is appended back into continuation history
4. the next model turn sees both the original tool-call response and the tool
   result messages

Normal executable tools run through `ToolExecutor`.

The runner also enforces tool visibility policy:

1. per-tool call limits
2. maximum tool round trips

Provider clients do not own that logic.

## Delegated Agent Tools

Agent-as-tool uses the same model-facing contract as any other tool:

1. the model sees a tool name and a JSON schema
2. it returns a tool call with arguments
3. the runner validates those arguments into a Pydantic model
4. the structured payload is sent to a delegated child agent
5. the child result is converted into a tool-result string
6. the caller agent continues its own loop

The two agent-as-tool variants are:

1. predefined agent tools via `AgentDescriptor.as_tool(...)`
2. the generic spawned-helper tool via `DefaultAgentTool`

Predefined agent tools receive exactly their declared payload shape. The generic
`agent` tool uses its own default schema with `name`, optional `description`,
and optional `task`.

## Handoffs

Handoff is different from agent-as-tool even though the model still sees it as
a tool-like choice.

When the runner intercepts a handoff call:

1. it resolves the target agent descriptor
2. it copies the current run state
3. it preserves the triggering handoff turn in transferred history
4. it appends a synthetic transfer acknowledgement
5. it appends the optional delegation message, or a default one if empty
6. it sends the transferred `RunState` to the next agent
7. it returns that downstream `RunResult`

The original agent does not resume after a handoff.

The two handoff variants are:

1. predefined transfer targets through `AgentDescriptor.handoffs`
2. the generic fresh-agent path through `DefaultHandoff`

## Hooks

`RunnerHooks` observes runner lifecycle events:

1. `on_agent_start`
2. `on_llm_start`
3. `on_llm_end`
4. `on_tool_call_start`
5. `on_tool_call_end`
6. `on_agent_end`

For successful runs, tool hooks may repeat across multiple turns while the
agent-level hooks still wrap the overall run once.

## Retry Policy

The runner adds only an optional outer retry layer.

Rules:

1. `Runner(max_attempts=1)` means no extra runner retry
2. higher values reuse `agentlane.models.retry_on_errors`
3. provider clients may still perform their own transport-level retries

This keeps retry behavior narrow and explicit.

## Thin Provider Boundary

Provider adapters should remain thin:

1. accept canonical request input
2. return canonical model responses
3. do not execute tools
4. do not run a client-side tool loop
5. do not own handoff behavior

That orchestration belongs in the harness runner.
