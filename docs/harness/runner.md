# Harness Runner

Date: 2026-04-05
Status: Phase 6 implementation ready for review

## What The Default Runner Owns

The default harness `Runner` is a reusable stateless service object.

For Phase 6 it owns the generic loop needed to execute direct answers, tool turns, first-class handoffs, and agent-as-tool subroutine calls:

1. build the next model request from `instructions + original_input + continuation_history`,
2. call the agent's configured `agentlane.models.Model`,
3. emit lifecycle hooks around the agent run and each LLM attempt,
4. accumulate the raw `ModelResponse` onto `RunState.responses`,
5. append the right continuation items back into `RunState.continuation_history`,
6. execute tool calls when the model returns them,
7. intercept handoffs specially and transfer the run when needed,
8. continue the loop for the next model turn when the caller remains active, and
9. return a minimal `RunResult` once a terminal assistant answer is produced.

The runner is where model-facing normalization happens. The agent boundary is
run-oriented, not `MessageDict`-oriented.

## Model Boundary

The runner does not introduce a second client abstraction.

It expects the task being run to expose a configured `model` compatible with:

1. `agentlane.models.Model`
2. canonical `MessageDict` request input at call time
3. canonical `ModelResponse` output

The default `Agent` now exposes that model directly as `agent.model`, but the
underlying static configuration lives in `AgentDescriptor`.

Phase 4 also preserves the broader native call surface exposed by
`agentlane.models.Model`. When present, the runner forwards:

1. `agent.model_args` via the model interface's `extra_call_args` seam
2. `agent.schema`
3. `agent.tools`

There is only one canonical tool configuration on the agent: `Tools`. The
runner uses that same value both for model visibility and for actual harness
tool execution.

## Run Boundary

The public runner boundary for this phase is intentionally small:

1. input state is `RunState`
2. final return value is `RunResult`
3. the model-call request `list[MessageDict]` is built internally by the runner

`RunState` currently contains only:

1. `original_input`
2. `continuation_history`
3. `responses`
4. `turn_count`

`RunResult` currently contains only:

1. `final_output`
2. `responses`
3. `turn_count`
4. `run_state`

Future concerns such as event logs and resumable interruption status are still
deferred until they are actually needed.

## Hook Order

For a successful run, the default hook order is:

1. `on_agent_start`
2. `on_llm_start`
3. model call
4. `on_llm_end`
5. zero or more `on_tool_call_start` / `on_tool_call_end` pairs
6. repeat steps 2-5 for additional model turns if tool calls were executed
7. `on_agent_end`

If a retryable model failure occurs, the runner repeats the `on_llm_start` -> model call
sequence for the next attempt. The agent-level hooks still wrap the overall run once.

## Retry Boundary

Runner retries are intentionally narrow in Phase 4.

1. Provider clients may still perform their own transport or schema retries internally.
2. The harness runner adds an optional outer retry policy via `Runner(max_attempts=...)`.
3. That outer retry policy reuses `agentlane.models.retry_on_errors` instead of adding new retry infrastructure.

The default `max_attempts` is `1`, so the runner does not add an extra retry layer unless explicitly configured.

## Result Shape

The runner returns `RunResult`.

That result is intentionally minimal:

1. `final_output` is the direct answer extracted from the terminal assistant turn
2. `responses` is the accumulated list of raw `ModelResponse` objects
3. `turn_count` is the completed model-turn count for the run
4. `run_state` is the final resumable run state when the runner completed successfully

## Tool Boundary

Phase 5 and Phase 6 keep the tool boundary simple.

1. Tool calls returned by the model are executed by the harness runner, not by provider clients.
2. Native executable tools still go through the shared `ToolExecutor`.
3. Predefined agent-as-tool calls are routed through runtime `send_message`, receive exactly the validated args-model payload for that tool, and return a tool result string to the caller loop.
4. `DefaultAgentTool` is the generic spawned-helper path. It injects a default helper prompt and also sends the delegated task as user input.
5. First-class handoffs are also model-visible tool choices, but the runner intercepts them specially instead of treating them as normal tool results.
6. On handoff, the runner transfers the full conversation history, preserves the triggering handoff turn as downstream history, adds a synthetic transfer acknowledgement, and then appends the optional delegation message.
7. Handoff does not inject a new default system prompt. The downstream agent uses only its own configured `instructions` when present.
8. `RunResult.run_state` is carried through so lifecycle persistence can continue from the delegated child after a transfer.
9. The runner also owns later-turn tool visibility by applying per-tool call limits and maximum tool round-trip limits from accumulated `RunState.responses`.
10. Provider clients stay thin: they accept tool definitions in the request and return raw tool-call responses, but they do not execute tools or run their own tool loop.
