# Harness Architecture

The harness is the layer you use when explicit message delivery is not enough
and you want a reusable agent loop on top of the runtime. It does not replace
the runtime. It uses the runtime as its execution substrate and adds a
structured way to turn model calls, tool calls, and handoffs into a coherent
workflow.

The pieces matter in relation to one another.
[`Task`](../../src/agentlane/harness/_task.py) gives ordinary orchestration code
a home above the runtime. [`Agent`](../../src/agentlane/harness/_agent.py)
layers in reusable configuration through
[`AgentDescriptor`](../../src/agentlane/harness/_lifecycle.py) and resumable
state through [`RunState`](../../src/agentlane/harness/_run.py).
[`Runner`](../../src/agentlane/harness/_runner.py) then conducts the actual
model loop and produces a [`RunResult`](../../src/agentlane/harness/_run.py).

## What The Harness Adds

The runtime already knows how to deliver messages and preserve ordering. The
harness adds a higher-level story on top of that:

1. `Task` gives ordinary application orchestration a home above the runtime
2. `Agent` keeps static configuration and resumable conversation state together
3. `Runner` executes the model loop for one run
4. tools and handoffs become first-class parts of that loop

That separation matters because queueing and persistence are different problems
from model reasoning. The harness keeps them apart.

## The Main Flow

At a high level, a run moves like this:

```text
Application / caller
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

The runner owns the model loop inside that run. It builds the next request,
calls the model, inspects the response, executes tools or handoffs, and either
continues or returns a final result.

That split is what makes resumable runs practical. The lifecycle can keep a
stable baseline of `RunState`, while the runner works on a private copy for the
current turn.

## Request Ownership

The harness public boundary is deliberately not a low-level message-dict API.

Instead:

1. callers provide a `RunInput`
2. the lifecycle turns that into a working `RunState`
3. the runner turns `RunState` into canonical model messages
4. provider clients receive the shared `agentlane.models` request shape

That keeps raw provider wire formats out of application code.

## Where To Read Next

Start with [Harness Tasks](./tasks.md) if you need orchestration without an LLM
loop. Read [Harness Agents](./agents.md) to understand the default agent type.
Read [Harness Runner](./runner.md) when you want the actual loop behavior.
