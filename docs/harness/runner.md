# Harness Runner

Once a run has been accepted and a working
[`RunState`](../../src/agentlane/harness/_run.py) exists, the runner takes
over. It turns that state into one or more model calls, executes tool work, and
decides whether control stays local or moves to another agent.

That behavior lives in [`Runner`](../../src/agentlane/harness/_runner.py).
[`RunResult`](../../src/agentlane/harness/_run.py) records what came out of the
loop, [`RunnerHooks`](../../src/agentlane/harness/_hooks.py) exposes useful
observation points, and
[`DefaultAgentTool`](../../src/agentlane/harness/_lifecycle.py) plus
[`DefaultHandoff`](../../src/agentlane/harness/_lifecycle.py) are the bridge
that lets delegation appear to the model as part of the same tool surface. If
shims are configured, the runner also consumes the prepared turn they build for
each model call.

`Runner` does not persist conversation state. It does own process-local
execution guards such as retry limits, max turns, and generic spawned-agent
depth/thread limits. Those spawned-agent limits apply inside the current runner
process; distributed runtimes need their own coordinator-level limits if work
crosses a process boundary.

The runner is used both by:

1. the runtime-facing `agentlane.harness.Agent`
2. the higher-level local `agentlane.harness.agents.DefaultAgent`, which uses
   that lower-level agent for the simpler `run(...)`, `run_stream(...)`, and
   `run_events(...)` surfaces

## The Loop

At a high level, one run looks like this:

1. prepare the next turn from instructions and current history
2. let shims adjust that prepared turn
3. build the next request
4. call the model
5. record the raw response
6. inspect the response
7. either finish, execute tools, or hand off to another agent

```text
queued run input
      |
      v
+---------------------------+
| DefaultAgent.run(...)     |
| or run_stream(...)        |
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
| prepare turn              |
| shims may mutate it       |
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
| inspect response          |
+------+------+-------------+
       |      |
       |      +--------------------+
       |                           |
       v                           v
+--------------+          +-------------------+
| execute tools |         | transfer handoff  |
+------+-------+          +---------+---------+
       |                            |
       v                            v
  next model turn             downstream run
```

The lifecycle owns queueing and persistence around this loop. The runner owns
the loop itself.

## Prepared Turns And Shims

Before each model call, the runner works from one
[`PreparedTurn`](../../src/agentlane/harness/shims/_types.py).

That object carries:

1. the working `RunState` (which itself holds the persisted instructions and
   history)
2. the effective visible tools for the turn
3. the effective model arguments for the turn
4. per-run transient state shared across shim callbacks

Instructions are not a separate top-level field. They live inside the run state
and are read or mutated through the run state, including the
`set_system_instruction(...)` and `append_system_instruction(...)` helpers on
the prepared turn.

If bound shims exist, they are called in descriptor order to adjust that
prepared turn before the runner builds the canonical message list. They may
also replace that final message list for one model call when
`transform_messages(...)` is needed.

## Streaming

The runner also owns live model streaming for one run.

At the lowest streaming level the harness does not define a second event model.
It reuses [`ModelStreamEvent`](../../src/agentlane/models/_streaming.py)
directly and adds one small harness handle:
[`RunStream`](../../src/agentlane/harness/_stream.py). A higher-level harness
event model is layered on top of these raw model events through
`run_events(...)`; see [Run Events](#run-events) below.

That split is deliberate:

1. `ModelStreamEvent` is the live per-model-call event type
2. `RunStream.result()` is the whole-run completion point

One streamed harness run may cross multiple model calls because of tools or
first-class handoff, so one `run_stream(...)` may emit more than one
`ModelStreamEventKind.COMPLETED`. The final whole-run result is still
`RunResult`.

Streaming remains local to the harness in this step. The runner and lifecycle
handle it without changing runtime `send_message(...)` delivery semantics.

## Streaming Boundaries

The streamed runner behavior has a few important boundaries:

1. first-class handoff keeps the stream going because control transfers to the
   delegated agent
2. agent-as-tool remains internal, so the child agent's own model events are
   not surfaced on the parent stream
3. streaming calls do not use the runner's outer retry wrapper after events
   have started emitting, because replaying another provider attempt on top of
   partial output would be incorrect
4. if you send work through `runtime.send_message(...)`, you still receive one
   final result after the run finishes
5. live per-event streaming is available through the local harness streaming
   APIs such as `DefaultAgent.run_stream(...)`

## Run Events

Alongside `run(...)` and `run_stream(...)`, the runner exposes a third entry
point: `run_events(...)`, which returns a
[`RunEventStream`](../../src/agentlane/harness/_events.py). This is a
higher-level harness event model layered over the raw `ModelStreamEvent`
stream.

`run_events(...)` accepts an optional `approval_events` async iterator. When
provided, brokered tool-approval lifecycle events are forwarded into the same
stream as `RunToolApprovalEvent`.

Each emitted item is a `RunEvent`, a union tagged by `RunEventKind`:

1. `RunModelStreamEvent` wraps one underlying `ModelStreamEvent`
2. `RunAgentStartEvent` / `RunAgentEndEvent` mark agent run boundaries
3. `RunLLMStartEvent` / `RunLLMEndEvent` mark model requests
4. `RunToolStartEvent` / `RunToolEndEvent` mark tool calls
5. `RunToolApprovalEvent` wraps one brokered approval event
6. `RunHandoffStartEvent` / `RunHandoffEndEvent` mark first-class handoff
   control transfers
7. `RunStateSnapshotEvent` carries a compact `RunStateSnapshot` at stable run
   boundaries
8. `RunPlanUpdatedEvent` carries the structured plan after a successful
   first-party plan-tool call

`RunStateSnapshotEvent` boundaries are named by `RunStateSnapshotBoundary`:
`run_start`, `turn_prepared`, `tool_round_end`, and `run_end`. The snapshot
itself is compact (`turn_count`, `history_length`, `response_count`, and a copy
of `shim_state`) rather than the full working state.

### Tool success and failure

`RunToolEndEvent` carries a framework-derived outcome alongside the raw
`result`:

- `ok: bool` — whether the runner considers the call successful.
- `error: ToolError | None` — a typed failure payload (`message`, optional
  stable `kind`) when `ok` is `False`.

Consumers read `event.ok` / `event.error` directly instead of inferring success
from result wording or reflecting over tool-specific result fields. The runner
derives the outcome with `tool_outcome(result)`, which recognizes, in order:

1. `ToolFailure` — the one public, typed way a tool implementation signals
   failure. `ToolFailure` is a `str` subclass carrying a `ToolError`, so the
   model still sees its text verbatim (the default formatter renders `str`
   values unchanged) while the framework reads the structured error. The
   first-party `bash` tool returns `ToolFailure` for timed-out, cancelled, and
   non-zero-exit outcomes.
2. A captured `BaseException` — a raised-and-captured tool error.
3. A mapping with a truthy `error` entry — the structured-mapping convention
   third-party tools commonly return.

Anything else is treated as `ok`. First-party tools that render operational
errors as plain strings stay `ok` because result wording is not a contract; a
tool that needs a failure marked returns `ToolFailure`.

`ToolFailure`, `ToolError`, the derived `ToolOutcome`, and the `tool_outcome`
helper are exported from `agentlane.harness`:

```python
from agentlane.harness import ToolError, ToolFailure, tool_outcome
```

A tool handler signals a typed failure by returning `ToolFailure(text=...,
error=ToolError(message=..., kind=...))`. The `text` is what the model sees; the
`error` is what `RunToolEndEvent.error` carries.

### Plan-updated events

After a successful first-party plan-tool call, the runner emits a
`RunPlanUpdatedEvent` carrying the structured plan as a tuple of `RunPlanItem`
(`step`, `status`) plus the optional model-supplied `explanation` and the
originating `tool_call`. Consumers render plan UX from this typed payload
instead of string-matching the plan tool's success message. The public
constants `PLAN_TOOL_NAME` and `PLAN_UPDATED_MESSAGE` are exported from
`agentlane.harness.tools` for callers that still need the wire literals.

### Lineage and scope semantics

Every task-carrying run event exposes lineage so consumers do not latch the
root task id and infer subagents by comparison:

- `parent_task_id: str | None` — the parent run's task id, or `None` for the
  stream root.
- `is_root: bool` — whether the event belongs to the stream's root run.

The emitter latches the first agent run's `task_id` as the stream root. In the
default single-agent-plus-delegation topology every event in one stream is a
root event, because a delegated child runs in its own runtime turn and surfaces
to the parent only as a tool call.

`RunToolStartEvent` / `RunToolEndEvent` additionally carry `is_delegation:
bool`, set structurally for agent-as-tool and handoff calls (by runner
tool-definition type, not by name heuristics), so consumers identify delegation
without an app-side tool-name registry.

Two scope contracts matter for consumers that aggregate run telemetry:

- `RunAgentEndEvent` fires for child tasks too, carrying that child's `result`,
  when an application shares run hooks across agents. Treating it as "the whole
  run ended" records child telemetry as root telemetry; gate root-only
  aggregation on `is_root`.
- State snapshots are root-stream-only by contract: a delegated child never
  emits `RunStateSnapshotEvent` into the parent stream.

## Stream Cancellation And Closure

`run_events(...)` and `run_stream(...)` both return a stream handle that
exposes `aclose()` (inherited from the shared
[`BaseRunStream`](../../src/agentlane/harness/_stream_base.py)). The
cancellation contract is:

1. **`aclose()` stops the underlying run.** When you pass a
   `cancellation_token` to `run_events(...)`, the runner wires
   `token.cancel` as the stream's `on_close` hook. Calling `await
   stream.aclose()` therefore cancels that token, which propagates through the
   async run chain and stops the in-flight provider request. `aclose()` also
   runs the stream's cleanup callbacks (cancelling the internal run task and any
   approval-forwarding task) and ends the iterator. Without a
   `cancellation_token`, `aclose()` still ends the iterator and runs cleanups,
   but there is no token to cancel, so a blocking provider call already in
   flight is not interrupted by closure alone.

2. **Cancelling the consuming task does not close the stream.** If you iterate
   the stream from a task and cancel that task, the `CancelledError` unwinds
   your `async for`, but it does not call `aclose()` for you. The underlying run
   task and provider request keep going until they finish or are cancelled
   another way. To stop the run when a consumer goes away, call `aclose()`
   explicitly — typically in a `finally` around the consumption loop, or by
   cancelling the same `cancellation_token` you handed to `run_events(...)`.

3. **`result()` must be retrieved after `aclose()`.** `aclose()` fails the
   stream's result future with `asyncio.CancelledError`. Await `result()` (and
   swallow `CancelledError`) after closing so asyncio does not log a
   "Future exception was never retrieved" warning. The shared
   `close_stream_callback` / `_close_stream` helpers in `_stream_base.py` follow
   exactly this close-then-drain pattern.

The recommended host pattern is to own the `cancellation_token`, pass it into
`run_events(...)`, consume the stream in a `try`/`finally`, and call
`await stream.aclose()` in the `finally` so a dying consumer always stops the
provider-side request:

```python
token = CancellationToken()
stream = await agent.run_events(prompt, cancellation_token=token)
try:
    async for event in stream:
        handle(event)
finally:
    await stream.aclose()
    with contextlib.suppress(asyncio.CancelledError):
        await stream.result()
```

### Cooperative cancellation for blocking I/O

[`CancellationToken`](../../src/agentlane/runtime/_cancellation.py) is
cooperative: it cannot interrupt a blocking call already running in a worker
thread. Its surface is `is_cancelled` (poll), `await wait_cancelled()` (await
the request), `link_future(future)` (cancel an `asyncio.Future`/task when the
token is cancelled), and `cancel()`.

For `asyncio.to_thread(...)`-style blocking calls (for example a synchronous
HTTP client, or `urllib.request.urlopen`), the intended pattern is:

1. Check `token.is_cancelled` before starting the blocking call and return early
   if cancellation is already requested.
2. Run the blocking call in a worker thread, keeping any per-call timeout short
   enough that a cancelled run does not linger longer than that timeout.
3. Check `token.is_cancelled` again after the call returns and discard the
   result if cancellation happened while it was in flight.

A blocking call such as `urlopen(...)` cannot be interrupted mid-flight by the
token, so a cancelled run may linger up to that call's own timeout before the
worker thread unblocks. Where you control the awaitable instead of a blocking
thread call, prefer `token.link_future(task)` so the token cancels it directly.
For cancellable network work, wrap the request in an `asyncio.Task` and link it,
or use an async HTTP client whose request future the token can cancel.

## Run Result And State

[`RunResult`](../../src/agentlane/harness/_run.py) records what came out of the
loop. Its fields are `final_output`, `responses`, `turn_count`, and an optional
`run_state`.

[`RunState`](../../src/agentlane/harness/_run.py) is the resumable shape. It
holds `instructions`, `history`, `responses`, persisted `shim_state`
([`ShimState`](../../src/agentlane/harness/_run.py)), and `turn_count`. It is
also one of the accepted `RunInput` forms (alongside a plain `str` and a
`list[RunHistoryItem]`), so a completed run's `run_state` can be passed back in
to resume the conversation. The persisted `shim_state` survives across resumes.

`turn_count` counts turns *started*, not completed: the runner increments it at
the start of each turn, before shims prepare the turn and before the model is
called. While turn N is being prepared or executed, `turn_count` is `N`, so the
first turn observes `turn_count == 1`; after the run finishes it equals the
total number of turns that ran (`RunResult.turn_count` is this final value).
Gate first-turn logic on `turn_count == 1` rather than reading it as a
completed-turns count.

## Request Ownership

One of the runner's main jobs is deciding how a high-level run turns into a
canonical model request.

That means the runner is the place where:

1. prepared instructions are combined with accumulated run history
2. any one-turn context items are added
3. visible tools are attached to the request
4. the structured-output schema is forwarded
5. model arguments are passed through

This is why the harness public API does not require application code to build
raw message dictionaries itself.

## Tool Calls

When the model returns tool calls, the runner appends that assistant turn to the
working history, executes the tool calls, appends tool results, and asks the
model again.

Ordinary executable tools run through
[`ToolExecutor`](../../src/agentlane/models/_tool_executor.py). The runner also
enforces tool visibility and loop-safety limits so a run cannot keep requesting
the same tools forever.

For each ordinary executable tool call, the runner also builds an explicit
`ToolExecutionContext` containing the run id, agent name, and model tool-call
id. `ToolExecutor` passes that context to the tool handler; permission policies
and approval callbacks can use it for audit or UI correlation without relying
on hidden process-local state. The executor accepts these contexts as a mapping
keyed by model tool-call id because one model response can contain multiple
tool calls. The tool handler receives only its own single context.

### Live Run-State View

The runner also stamps a read-only view of the live run onto each ordinary
tool's context as `context.run_state`, typed as
[`RunStateView`](../../src/agentlane/models/_tool.py) and implemented by
[`LiveRunStateView`](../../src/agentlane/harness/_run.py). It lets a tool
observe the run it is executing inside without an app-built side channel:

- `run_state.task_id` — the stable task identity for the run. `context.run_id`
  is guaranteed to equal `str(run_state.task_id)`, so per-run stores may be
  keyed by either field consistently. This is the documented, stable
  relationship between `run_id` and task identity.
- `run_state.shim_state` — the live persisted shim-owned state, read-only. The
  view reads through to current values, so state a shim writes earlier in the
  run is visible to later tool calls.
- `run_state.active_skill_names` — the names of skills active for the run. A
  skills shim records these under a `shim_state` key ending in
  [`ACTIVE_SKILL_NAMES_STATE_KEY_SUFFIX`](../../src/agentlane/harness/_run.py)
  (the shim name is the key prefix, so multiple skills shims never collide);
  the accessor unions every such key. Tools resolving skill-relative resources
  read this instead of reaching for a private state key.

`context.run_state` is `None` when a tool runs outside a runner loop (for
example a direct `tool.run(...)` call). The view is read-only; a tool that
needs to persist state should do so through its own shim, never by mutating the
run.

The read-only boundary is enforced at the top-level `shim_state` mapping:
inserting, replacing, or deleting keys through `context.run_state.shim_state`
raises `TypeError`. Values stored under those keys are ordinary shim-owned
objects. Shims that expose nested containers should expose immutable values when
they need stronger protection.

## Agent-As-Tool

Agent-as-tool uses the same model-facing pattern as any other tool: the model
selects a tool name and arguments, and the framework validates those arguments
before doing anything else.

The important difference is what happens next. Instead of calling a local
function, the runner routes the work to another agent and converts the child
result back into a tool result for the caller's loop.

Use this pattern when the caller should continue after the delegated work
returns.

The first-party base `agent` tool is the generic spawned-helper form of this
pattern. It accepts a one-word logging/tracing `name` plus a complete `task`
instruction, spawns a fresh helper, and returns that helper's result as tool
text. Generic spawned helpers do not inherit the parent conversation or parent
system prompt. They inherit parent direct tools by default and reuse the
parent's descriptor shims, so shim-contributed tools and prompt guidance flow
through the same prepared-turn path used by parent agents. The `Runner` carries
process-local depth and live-agent limits to prevent runaway recursion.

The runner resolves child-visible direct tools with `ToolConfig`.
`INHERIT_TOOLS` merges parent and child-local tools, `OVERRIDE_TOOLS` replaces
or clears parent tools, and `RESTRICT_TOOLS.only(...)` filters inherited parent
tools by name before adding child-local tools. Bare `Tools(...)` and `None`
preserve legacy override behavior.

`agent_max_depth` is inclusive for generic spawned agents. A direct child has
depth 1, so `Runner(agent_max_depth=4)` allows delegated agents through depth
4 and rejects the next nested spawn attempt.

## Handoffs

Handoffs also appear to the model as tool-like choices, but their semantics are
different. A handoff transfers control. The original agent does not resume
afterward.

That distinction is the reason the runner handles handoffs itself instead of
treating them as normal tool execution.

Use a handoff when the next agent should take over the conversation rather than
act as a subroutine.

## Hooks And Retries

[`RunnerHooks`](../../src/agentlane/harness/_hooks.py) give you lifecycle
callback points during the run, such as agent start and end, model calls, and
tool calls.

What a hook does at those points is up to the author. Common uses include
tracing, logging, metrics, policy checks, database writes, script execution,
and other application-specific side effects.

Public hook inputs accept either one hook instance or an ordered sequence of
hooks. When more than one hook is present, the harness composes them
internally and forwards callbacks in order.

For bound agents, that composition is resolved once per concrete agent
instance:

1. explicit developer-supplied hooks run first, in the order provided
2. shim-contributed hooks run second, in shim descriptor order

That keeps the runner model simple. It still receives one resolved hook object
for the run, even when several hook implementations are active behind it.

The runner can also add an outer retry layer. That retry is intentionally
narrow. Provider-specific retries still belong in the model client layer.

## Related Docs

1. [Harness Architecture](./architecture.md)
2. [Harness Agents](./agents.md)
3. [Harness Default Agents](./default-agents.md)
4. [Models Overview](../models/overview.md)
