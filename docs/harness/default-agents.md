# Default Agents

`agentlane.harness.agents.DefaultAgent` is the high-level local harness entry
point for straightforward agent usage.

Use it when you want to:

1. define an agent with `AgentDescriptor`,
2. call `run(...)` directly, and
3. let the framework manage the local runtime and default runner for you.

This is the primary local agent-building interface. Internally it uses the
runtime-facing `agentlane.harness.Agent`, but from the developer standpoint it
is a first-class agent type, not a secondary adapter.

## Import Path

```python
from agentlane.harness.agents import AgentBase, DefaultAgent
```

`AgentBase` is the abstract base contract for future high-level harness agent
types. It defines three shared execution controls:

1. `run(...)`
2. `fork(...)`
3. `reset()`

## Two Authoring Styles

### Subclass-Based

```python
from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent


class SupportAgent(DefaultAgent):
    descriptor = AgentDescriptor(
        name="Support",
        model=model,
        instructions="You are a support agent.",
    )


agent = SupportAgent()
result = await agent.run("My order arrived damaged.")
```

### Direct Construction

```python
from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent

agent = DefaultAgent(
    descriptor=AgentDescriptor(
        name="Support",
        model=model,
        instructions="You are a support agent.",
    )
)

result = await agent.run("My order arrived damaged.")
```

## What `DefaultAgent` Owns

`DefaultAgent` owns the high-level local agent experience:

1. descriptor resolution
2. optional runtime provisioning
3. optional runner provisioning
4. persisted `RunState` between repeated `run(...)` calls

It delegates the real orchestration to the existing runtime-facing harness
stack:

1. `agentlane.harness.Agent`
2. `AgentLifecycle`
3. `Runner`

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
first = await agent.run("My order arrived damaged.")
second = await agent.run("Please summarize the next step.")
```

The second call continues from the persisted `RunState` returned by the first
call.

One `DefaultAgent` instance also serializes concurrent `run(...)` calls on
itself so its local `RunState` stays coherent.

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
await agent.run("My order arrived damaged.")
branch = await agent.fork("Draft a more formal reply.")
```

After that call:

1. `branch.run_state` contains the branch result
2. `agent.run_state` still points at the original main conversation line

## Explicit Resume

`DefaultAgent.run(...)` also accepts `RunState` directly.

That path is an explicit resume request. The agent does not combine the passed
`RunState` with its stored baseline. It resumes from the provided state only,
then stores the returned updated state afterward.

`DefaultAgent.fork(...)` also accepts `RunState`. That creates a branch from
the provided state and still does not write the branch result back into the
agent's stored primary state.

## Reset

`DefaultAgent.reset()` clears the agent's locally persisted `RunState`.

Use it when the same agent instance should start a new conversation instead
of continuing the previous one.

## Boundary

`DefaultAgent` is local and convenience-oriented.

`agentlane.harness.Agent` is runtime-facing and message-oriented.

Use:

1. `DefaultAgent` for the smallest local developer surface
2. `Agent` when you want direct runtime addressing, explicit binding, or lower
   level orchestration control
