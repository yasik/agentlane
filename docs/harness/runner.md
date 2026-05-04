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
   that lower-level agent for the simpler `run(...)` and `run_stream(...)`
   surfaces

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

That object carries the effective:

1. instructions
2. tools
3. model arguments
4. working `RunState`
5. one-turn context items
6. per-run transient state

If bound shims exist, they are called in descriptor order to adjust that
prepared turn before the runner builds the canonical message list. They may
also replace that final message list for one model call when
`transform_messages(...)` is needed.

## Streaming

The runner also owns live model streaming for one run.

The harness does not define a second event model here. It reuses
[`ModelStreamEvent`](../../src/agentlane/models/_streaming.py) directly and
adds one small harness handle:
[`RunStream`](../../src/agentlane/harness/_stream.py).

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
text. Generic spawned helpers do not inherit the parent conversation, parent
system prompt, or parent custom tools. They receive the standard base tools by
default through `HarnessToolsShim`, so tool guidance is appended through the
same prepared-turn path used by parent agents. The `Runner` carries
process-local depth and live-agent limits to prevent runaway recursion.

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
