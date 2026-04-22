# Harness Shims

Shims are the harness extension point for behavior that needs to change what a
run actually does.

Use a shim when you need to:

1. adjust the single persisted system instruction,
2. add or adjust visible tools,
3. append visible items to the persisted conversation history,
4. update shim-owned persisted state, or
5. perform one-call message rewriting as an advanced escape hatch.

This is the mechanism intended for features such as skills, memory, and context
compaction. Those features should build on this seam instead of adding new
special-case fields to the core harness types.

The first major first-party example is [Harness Skills](./skills.md).

## Import Path

```python
from agentlane.harness.shims import PreparedTurn, Shim
```

## Mental Model

The shim contract is built around one persisted system instruction and one
append-only conversation history.

The important state surfaces are:

1. [`RunState.instructions`](../../src/agentlane/harness/_run.py)
   The single authoritative system instruction for the run.
2. [`RunState.history`](../../src/agentlane/harness/_run.py)
   The append-only conversation the model has seen so far.
3. [`RunState.shim_state`](../../src/agentlane/harness/_run.py)
   Persisted shim-owned state that survives later `run(...)` calls and explicit
   `RunState` resume paths.
4. `PreparedTurn.transient_state`
   Per-run in-memory state shared across shim callbacks and discarded when the
   run ends.

That leads to one recommended pattern:

1. bootstrap the system instruction before the first model turn if needed,
2. keep later changes append-only by writing conversation items into history,
3. keep temporary in-memory values in `transient_state`,
4. treat direct system-instruction replacement as the stronger escape hatch.

## Where Shims Are Declared

Shims are attached to
[`AgentDescriptor`](../../src/agentlane/harness/_lifecycle.py):

```python
descriptor = AgentDescriptor(
    name="Support",
    model=model,
    instructions="You are a support assistant.",
    shims=(ReplyPrefixShim(), TurnCounterShim()),
)
```

Descriptor order is execution order. Later shims see the effects of earlier
ones.

## Common Authoring Shape

Most shims should be one class.

Subclass `Shim` and override the lifecycle callbacks you need. That is enough
for most instruction changes, tool changes, append-only history writes, and
shim-owned state updates.

Internally, the harness still creates one bound session per agent instance. You
only need to think about `BoundShim` when a shim needs private per-agent
in-memory state or custom bind-time setup.

## What A Shim Can Change

The main mutation surface is
[`PreparedTurn`](../../src/agentlane/harness/shims/_types.py).

It gives a shim access to:

1. `run_state`
2. `tools`
3. `model_args`
4. `transient_state`

The most important helpers are:

1. `set_system_instruction(...)`
2. `append_system_instruction(...)`
3. `append_history_item(...)`
4. `append_history_items(...)`

Use them like this:

1. change `tools` or `model_args` in `prepare_turn(...)`,
2. append to the persisted system instruction only when the run needs a
   bootstrap-time behavior change,
3. append visible conversation items to `RunState.history` for ongoing changes,
4. write resumable shim-owned state to `RunState.shim_state`.

`append_history_item(...)` uses the same run-history item contract as the rest
of the harness request builder. Supported items include:

1. canonical message dicts,
2. prior `ModelResponse` values,
3. `PromptSpec` values,
4. user-side content values such as strings, JSON-like values, or Pydantic
   models.

## Shim State

Shims have two state buckets.

### Transient Per-Run State

`PreparedTurn.transient_state` is shared across all turns in one run and is
discarded when the run ends.

It is typed as
[`RunContext`](../../src/agentlane/models/run/_context.py) and uses
[`DefaultRunContext`](../../src/agentlane/models/run/_context.py) by default.

Use it for temporary cached values or other in-memory state that should not
survive a resumed run.

### Persisted Resumable State

`PreparedTurn.run_state.shim_state` is copied into the saved
[`RunState`](../../src/agentlane/harness/_run.py).

It is typed as [`ShimState`](../../src/agentlane/harness/_run.py).

Use it when a shim needs state to survive:

1. later `run(...)` calls on the same `DefaultAgent`,
2. explicit `RunState` resume paths,
3. crash recovery through saved run state.

## Lifecycle

One bound shim session follows this order:

```text
bind once per concrete agent instance
        |
        v
on_run_start(...)
        |
        v
prepare_turn(...)
        |
        v
transform_messages(...)
        |
        v
model call completes
        |
        v
on_model_response(...)
        |
        v
on_run_end(...)
```

The normal mutation points are:

1. `prepare_turn(...)` for instruction updates, tool changes, and history
   appends before the next model call,
2. `on_model_response(...)` for updating shim-owned state from the completed
   response,
3. `transform_messages(...)` only when you need one-call message surgery after
   the runner has already built the canonical request.

## Minimal Example

```python
from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.shims import PreparedTurn, Shim


class ReplyPrefixShim(Shim):
    @property
    def name(self) -> str:
        return "reply-prefix"

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        if turn.run_state.turn_count == 1:
            turn.append_system_instruction(
                "Always start every reply with `Support:`.",
                separator="\n",
            )


agent = DefaultAgent(
    descriptor=AgentDescriptor(
        name="Support",
        model=model,
        instructions="You are a concise support assistant.",
        shims=(ReplyPrefixShim(),),
    )
)

result = await agent.run("My order arrived damaged.")
```

## Advanced Bound Sessions

Use `BoundShim` only when a shim needs private per-agent in-memory state or
custom setup when the shim is attached to one concrete agent instance.

In that case, override `Shim.bind(...)` and return a custom `BoundShim`
session. One `Shim` definition may then be reused across many agents without
leaking mutable state between them.

For a runnable example, see
[examples/harness/default_agent_shims_quickstart](../../examples/harness/default_agent_shims_quickstart/README.md).

## Boundaries

Shims are meant to change run behavior.
[`RunnerHooks`](../../src/agentlane/harness/_hooks.py) are not. Hooks remain
observation-only for tracing, logging, and tests.

Shims also do not change runtime delivery behavior. They operate inside the
harness lifecycle and runner after work has already been accepted by a bound
agent instance.
