# Default Agents

`agentlane.harness.agents.DefaultAgent` is the high-level local harness entry
point for straightforward agent usage.

Use it when you want to:

1. define an agent with `AgentDescriptor`,
2. call `run(...)` directly, and
3. optionally observe live model or high-level run events, and
4. let the framework manage the local runtime and default runner for you.

This is the primary local agent-building interface. Internally it uses the
lower-level addressed `agentlane.harness.Agent`, but from the developer standpoint it
is a first-class agent type, not a secondary adapter.

## Starting Point For High-Level Agents

`DefaultAgent` is the recommended starting point when you are developing a
high-level local agent. It gives you one direct `run(...)` surface while still
using the same descriptor, tools, shims, runner, and run-state contracts as the
lower-level harness.

A practical high-level agent usually starts with:

1. `AgentDescriptor` for the model, instructions, tools, shims, and optional
   explicit agent name
2. `DefaultAgent` for local runtime and runner ownership
3. `HarnessToolsShim` when the agent needs first-party workspace tools
4. `Tools` limits when you want predictable tool-loop behavior

If the descriptor name is omitted or blank, AgentLane generates a Faker-backed
adjective-noun fallback name. Applications that care about stable trace labels
should pass the name explicitly.

This path keeps early application code compact. If the agent later needs direct
runtime addressing, distributed workers, or custom orchestration, move the same
descriptor and tool definitions to the lower-level harness APIs. Read
[Harness Distributed Agents](./distributed-agents.md) for the pattern that keeps
`DefaultAgent` at the top while worker agents move behind explicit runtimes.

## Import Path

```python
from agentlane.harness.agents import AgentBase, DefaultAgent
```

`AgentBase` is the abstract base contract for future high-level harness agent
types. It defines four shared execution controls:

1. `run(...)`
2. `run_stream(...)`
3. `fork(...)`
4. `reset()`

`DefaultAgent` also exposes `run_events(...)` for high-level lifecycle
streaming without requiring every future `AgentBase` implementation to support
that broader event surface immediately.

## Two Authoring Styles

### Subclass-Based

```python
from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent


class CareNavigationAgent(DefaultAgent):
    descriptor = AgentDescriptor(
        name="Care Navigation",
        model=model,
        instructions="You are a patient care navigation agent.",
    )


agent = CareNavigationAgent()
result = await agent.run("I feel dizzy after starting a new medication.")
```

### Streaming

```python
stream = await agent.run_stream("I feel dizzy after starting a new medication.")

async for event in stream:
    ...

result = await stream.result()
```

### Direct Construction

```python
from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent

agent = DefaultAgent(
    descriptor=AgentDescriptor(
        name="Portfolio Risk",
        model=model,
        instructions="You are a portfolio risk analyst.",
    )
)

result = await agent.run("Review semiconductor exposure before rebalancing.")
```

## What `DefaultAgent` Owns

`DefaultAgent` owns the high-level local agent experience:

1. descriptor resolution
2. optional runtime provisioning
3. optional runner provisioning
4. persisted `RunState` between repeated `run(...)` calls
5. live streaming through `run_stream(...)`
6. high-level lifecycle streaming through `run_events(...)`
7. binding and reuse of any configured harness shims

It delegates the real orchestration to the existing runtime-facing harness
stack:

1. `agentlane.harness.Agent`
2. `AgentLifecycle`
3. bound shims, when configured
4. `Runner`

### Reading The Result

`run(...)`, `fork(...)`, and `stream.result()` all return a
[`RunResult`](../../src/agentlane/harness/_run.py) with four fields:

1. `final_output` is the answer extracted from the terminal run turn
2. `responses` is the raw `ModelResponse` list accumulated across the run
3. `turn_count` is the number of model turns completed
4. `run_state` is the final resumable `RunState` when available

```python
result = await agent.run("I feel dizzy after starting a new medication.")
print(result.final_output)
```

`result.reasoning` returns the reasoning from the most recent response that
carries it. The terminal turn is often a tool call or structured-output
emission with no reasoning, so this scans `responses` in reverse rather than
reading `responses[-1]`, which would drop reasoning emitted on an earlier turn.

## Runtime Behavior

If no runtime is supplied:

1. `DefaultAgent` provisions a local single-threaded runtime for the call
2. starts it automatically
3. drains it and tears it down cleanly on exit

If a runtime is supplied:

1. `DefaultAgent` uses that runtime
2. scopes startup and shutdown to the call only when needed
3. does not replace the supplied runtime with a different one

## Runner Behavior

If no runner is supplied:

1. `DefaultAgent` creates one default `Runner()`
2. reuses it on later `run(...)` calls for that agent instance

If a runner is supplied:

1. `DefaultAgent` uses it directly
2. does not replace it with a new default runner

## Repeated Runs

Repeated `run(...)` calls on the same agent instance continue the same
conversation by default.

```python
first = await agent.run("I feel dizzy after starting a new medication.")
second = await agent.run("Please summarize the next step.")
```

The second call continues from the persisted `RunState` returned by the first
call.

One `DefaultAgent` instance also serializes concurrent `run(...)` calls on
itself so its saved conversation state stays coherent.

`run_stream(...)` follows that same saved conversation and the same
serialization rule. A streamed run commits the updated `RunState` only after
the stream completes successfully.

`run_events(...)` follows the same state and serialization rule while yielding
high-level run events instead of only model stream events.

If the descriptor declares shims, `DefaultAgent` binds them once for that
concrete agent instance and reuses those bound sessions across repeated runs.

## Streaming Semantics

`run_stream(...)` returns a
[`RunStream`](../../src/agentlane/harness/_stream.py).

Use it like this:

1. iterate the stream for live [`ModelStreamEvent`](../../src/agentlane/models/_streaming.py) values
2. await `stream.result()` for the final `RunResult`

Important details:

1. `ModelStreamEventKind.COMPLETED` is model-call completion, not whole-run completion
2. one streamed harness run may emit more than one `COMPLETED` event because of
   tools or first-class handoff
3. early close or cancellation does not commit partial `RunState`
4. first-class handoff continues the outer stream because control transfers
5. agent-as-tool stays internal in the current streaming contract

## Run Events

`run_events(...)` returns a
[`RunEventStream`](../../src/agentlane/harness/_events.py). It is the broader
streaming surface for applications that need to drive dashboards, logs, or
debug UIs from one ordered event feed.

Use it like this:

```python
from agentlane.harness import RunEventKind

stream = await agent.run_events("Plan the documentation update.")

async for event in stream:
    if event.kind == RunEventKind.TOOL_START:
        ...

result = await stream.result()
```

The stream includes:

1. wrapped `ModelStreamEvent` values, preserving the existing model stream
   payload
2. agent start and end events
3. LLM request start and end events
4. tool call start and end events
5. brokered approval requested/resolved events when `approval_events=` is
   provided
6. first-class handoff start and end events
7. compact full `RunState` snapshots at stable boundaries

State snapshots are emitted at run start, after each prepared turn, after each
tool round, and at run end. Each snapshot contains the current turn count,
history length, response count, and a copied `shim_state`. It intentionally
does not include full conversation history or raw responses.

`run_stream(...)` remains unchanged and continues to yield only
`ModelStreamEvent` values. Use `run_events(...)` only when the caller wants the
broader lifecycle feed.

Pass `approval_events=broker.events()` when a host wants approval
requested/resolved events from `ToolApprovalBroker` merged into the same
ordered stream.

## Fork

`DefaultAgent.fork(...)` runs one branch without mutating the agent's
persisted primary conversation line.

Simple current behavior:

1. snapshot the current persisted baseline
2. run the branch under a fresh runtime agent id
3. return the branch `RunResult`
4. keep `DefaultAgent.run_state` unchanged

Example:

```python
await agent.run("Review semiconductor exposure before rebalancing.")
branch = await agent.fork("Draft a more formal risk note.")
```

After that call:

1. `branch.run_state` contains the branch result
2. `agent.run_state` still points at the original main conversation line

## Explicit Resume

`DefaultAgent.run(...)` also accepts `RunState` directly.

That path is an explicit resume request. The agent does not combine the passed
`RunState` with its already saved conversation state. It resumes from the
provided state only, then stores the returned updated state afterward.

`DefaultAgent.fork(...)` also accepts `RunState`. That creates a branch from
the provided state and still does not write the branch result back into the
agent's stored primary state.

## Reset

`DefaultAgent.reset()` clears the agent's saved `RunState`.

Use it when the same agent instance should start a new conversation instead
of continuing the previous one.

## Boundary

`DefaultAgent` is local and convenience-oriented.

`agentlane.harness.Agent` is the lower-level addressed harness agent.

Use:

1. `DefaultAgent` for the smallest local developer surface
2. `Agent` when you want direct runtime addressing, explicit binding, or lower
   level orchestration control
3. [Harness Shims](./shims.md) when you need to extend instructions, tools, or
   turn context without changing the core harness types
4. [Harness Distributed Agents](./distributed-agents.md) when a top-level
   harness agent coordinates worker runtimes
