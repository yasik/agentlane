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
that lets delegation appear to the model as part of the same tool surface.

The runner is used both by:

1. the runtime-facing `agentlane.harness.Agent`
2. the higher-level local `agentlane.harness.agents.DefaultAgent`, which uses
   that lower-level agent for the simpler `run(...)` and `run_stream(...)`
   surfaces

## The Loop

At a high level, one run looks like this:

1. build the next request from instructions and current history
2. call the model
3. record the raw response
4. inspect the response
5. either finish, execute tools, or hand off to another agent

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

## Request Ownership

One of the runner's main jobs is deciding how a high-level run turns into a
canonical model request.

That means the runner is the place where:

1. instructions are combined with accumulated run history
2. visible tools are attached to the request
3. the structured-output schema is forwarded
4. model arguments are passed through

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

## Handoffs

Handoffs also appear to the model as tool-like choices, but their semantics are
different. A handoff transfers control. The original agent does not resume
afterward.

That distinction is the reason the runner handles handoffs itself instead of
treating them as normal tool execution.

Use a handoff when the next agent should take over the conversation rather than
act as a subroutine.

## Hooks And Retries

[`RunnerHooks`](../../src/agentlane/harness/_hooks.py) let you observe the run
at meaningful points such as agent start and end, model calls, and tool calls.

The runner can also add an outer retry layer. That retry is intentionally
narrow. Provider-specific retries still belong in the model client layer.

## Related Docs

1. [Harness Architecture](./architecture.md)
2. [Harness Agents](./agents.md)
3. [Harness Default Agents](./default-agents.md)
4. [Models Overview](../models/overview.md)
