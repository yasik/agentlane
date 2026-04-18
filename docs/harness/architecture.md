# Harness Architecture

The harness is the layer you use when explicit message delivery is not enough
and you want a reusable agent loop on top of the runtime. It does not replace
the runtime. It uses the runtime as its execution substrate and adds a
structured way to turn model calls, tool calls, and handoffs into a coherent
workflow.

The pieces matter in relation to one another.
[`Task`](../../src/agentlane/harness/_task.py) gives ordinary orchestration
code a home above the runtime. The lower-level
[`Agent`](../../src/agentlane/harness/_agent.py) is the harness agent type you
bind to a runtime and address by `AgentId`. It binds addressed runs to one
descriptor, one lifecycle, and one runner. The local
[`agentlane.harness.agents.DefaultAgent`](../../src/agentlane/harness/agents/__init__.py)
provides the smaller high-level `run(...)` and `run_stream(...)` surface. The
[`agentlane.harness.shims`](../../src/agentlane/harness/shims/__init__.py)
package provides the mutating extension seam for instructions, tools, and
per-run context shaping. The [`Runner`](../../src/agentlane/harness/_runner.py)
then conducts the actual model loop and produces a
[`RunResult`](../../src/agentlane/harness/_run.py).

## What The Harness Adds

The runtime already knows how to deliver messages and preserve ordering. The
harness adds a higher-level story on top of that:

1. `Task` gives ordinary application orchestration a home above the runtime
2. the lower-level addressed `Agent` keeps static configuration and resumable
   conversation state together
3. `DefaultAgent` provides the smaller local `run(...)` surface for
   straightforward usage
4. `DefaultAgent` also provides `run_stream(...)` for live model events on that
   same primary conversation line
5. shims can adjust effective instructions, tools, and transient turn context
   without widening the core harness types
6. `Runner` executes the model loop for one run
7. tools and handoffs become first-class parts of that loop

That separation matters because queueing and persistence are different problems
from model reasoning. The harness keeps them apart.

## The Main Flow

At a high level, a run moves like this:

```text
Application / caller
        |
        | run(...), run_stream(...),
        | or send_message(run_input)
        v
+---------------------------+
| DefaultAgent             |
| high-level local agent   |
+-------------+-------------+
              |
              v
+---------------------------+
| RuntimeEngine             |
| routing + instance reuse  |
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
+-------------+-------------+
              |
              v
+---------------------------+
| Bound shims               |
| turn preparation +        |
| optional message changes  |
+-------------+-------------+
              |
              v
+---------------------------+
| Runner                    |
| model loop                |
| tools + handoffs          |
+------+------+-------------+
       |      |
       |      +--------------------+
       |                           |
       v                           v
+--------------+          +-------------------+
| ToolExecutor |          | runtime messaging |
| local tools  |          | delegated agents  |
+--------------+          +-------------------+
```

The important point is that the runtime still owns delivery and instance reuse.
The harness owns the meaning of a run.

## Why Lifecycle And Runner Are Separate

The lifecycle owns sequencing. If multiple inputs arrive for the same concrete
agent, it queues them and makes sure only one run is active at a time.

Bound shims sit between lifecycle and runner. They are bound once for the
concrete agent instance, then get a chance to shape each prepared turn before
the runner builds the next model request.

The runner owns the model loop inside that run. It builds the next request,
calls the model, inspects the response, executes tools or handoffs, and either
continues or returns a final result.

That split is what makes resumable runs practical. The lifecycle can keep a
stable baseline of `RunState`, while the runner works on a private copy for the
current turn.

[`agentlane.harness.agents.DefaultAgent`](../../src/agentlane/harness/agents/__init__.py)
sits one level above that lower-level path. It provisions a local runtime when
needed, keeps a primary `RunState` between repeated `run(...)` and
`run_stream(...)` calls, and still routes through the same lower-level
`Agent` plus `Runner` stack underneath.

## Request Ownership

The harness public boundary is deliberately not a low-level message-dict API.

Instead:

1. callers either use `DefaultAgent.run(...)` locally or send `RunInput` to the
   lower-level addressed `Agent`
2. the lifecycle turns that into a working `RunState`
3. bound shims can adjust the prepared turn and, if needed, the final message
   list for the next model call
4. the runner turns that prepared turn into canonical model messages
5. provider clients receive the shared `agentlane.models` request shape

That keeps raw provider wire formats out of application code.

The same boundary applies to streaming. `DefaultAgent.run_stream(...)` is a
local harness path built on top of the same lifecycle and runner ownership
model. When you use `runtime.send_message(...)`, you still get one final result
after the run finishes. Live per-event streaming is currently available through
the local harness streaming APIs instead. Distributed transport streaming is a
later concern.

## Where To Read Next

Start with [Harness Tasks](./tasks.md) if you need orchestration without an LLM
loop. Read [Harness Default Agents](./default-agents.md) for the smallest local
developer surface. Read [Harness Shims](./shims.md) if you need to extend run
behavior without changing the core harness types. Read [Harness Agents](./agents.md)
to understand the lower-level addressed agent type. Read [Harness Runner](./runner.md)
when you want the actual loop behavior.
