# Harness Runner

Date: 2026-04-05
Status: Phase 4 baseline

## What The Default Runner Owns

The default harness `Runner` is a reusable stateless service object.

For Phase 4 it owns the minimal generic loop needed to execute one terminal model turn:

1. build the next model request from `instructions + original_input + continuation_history`,
2. call the agent's configured `agentlane.models.Model`,
3. emit lifecycle hooks around the agent run and each LLM attempt,
4. accumulate the raw `ModelResponse` onto `RunState.responses`,
5. interpret the response as a direct terminal answer for the current phase,
6. append the raw `ModelResponse` to `RunState.continuation_history`, and
7. return a minimal `RunResult`.

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
runner uses that same value for model visibility now, and later phases will
reuse it for harness tool-loop behavior.

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

Future concerns such as event logs, resumable interruption status, and handoff
state are deferred until they are actually needed.

## Hook Order

For a successful run, the default hook order is:

1. `on_agent_start`
2. `on_llm_start`
3. model call
4. `on_llm_end`
5. `on_agent_end`

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

## Phase Boundary

Phase 4 stops before tool execution and handoffs.

1. If the model returns tool calls, the runner raises `ModelBehaviorError`.
2. `on_tool_call_start` and `on_tool_call_end` remain reserved hook points for Phase 5.
3. Sub-agent delegation and handoffs remain out of scope here.
