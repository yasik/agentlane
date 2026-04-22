# Harness Shims

Shims are the harness extension point for behavior that needs to change what a
run actually does.

Use a shim when you need to:

1. explicitly append to or replace the persisted system instruction,
2. add or adjust visible tools,
3. append visible conversation items to persisted history,
4. update persisted run-owned state, or
5. rewrite the final model request messages before one call.

This is the mechanism intended for future features such as skills, memory, and
context compaction. Those capabilities should build on this seam instead of
adding new special-case fields to the core harness types.

The first major first-party example of that pattern is
[Harness Skills](./skills.md).

## Import Path

```python
from agentlane.harness.shims import (
    PreparedTurn,
    Shim,
)
```

## The Common Shim Shape

Most shims should be one class.

Subclass `Shim` and override the lifecycle callbacks you need. That is enough
for instruction changes, tool changes, message rewrites, and persisted shim
state in most cases.

Internally, the harness still creates one per-agent bound session for each
shim. You only need to think about that lower-level `BoundShim` contract when
your shim needs private in-memory state for one concrete agent instance or
custom bind-time setup.

## Where Shims Are Declared

Shims are attached to [`AgentDescriptor`](../../src/agentlane/harness/_lifecycle.py):

```python
descriptor = AgentDescriptor(
    name="Support",
    model=model,
    instructions="You are a support assistant.",
    shims=(ReplyPrefixShim(), TurnCounterShim()),
)
```

The descriptor order is the execution order. Later shims see the results of
earlier ones.

## What A Shim Can Change

The main mutation surface is [`PreparedTurn`](../../src/agentlane/harness/shims/_types.py).

It gives a shim access to:

1. `run_state`
2. `tools`
3. `model_args`
4. `transient_state`

The intended use is:

1. change `tools` or `model_args` in `prepare_turn(...)`,
2. update the persisted system instruction explicitly with
   `set_system_instruction(...)` or `append_system_instruction(...)`,
3. append visible conversation items with `append_history_item(...)` or
   `append_history_items(...)`,
4. update persisted state in `run_state.shim_state`,
5. use `transform_messages(...)` only when you need one-call message surgery
   after the runner has already built the canonical message list.

The system instruction is a single persisted value stored at
[`RunState.instructions`](../../src/agentlane/harness/_run.py). It is not split
into multiple system messages.

The visible conversation is one append-only list stored at
[`RunState.history`](../../src/agentlane/harness/_run.py).

`append_history_item(...)` uses the same run-history item contract as the rest
of the harness request builder. Supported items include:

1. canonical message dicts,
2. prior `ModelResponse` values,
3. `PromptSpec` values,
4. user-side content values such as strings, JSON-like values, or Pydantic
   models.

This means the normal ongoing shim pattern is:

1. bootstrap the system instruction before the first model turn if needed,
2. keep later changes append-only by writing conversation items into history,
3. keep temporary in-memory values in `transient_state`.

Avoid repeated system-instruction mutation during long-running sessions unless
you explicitly need that stronger escape hatch.

## Shim State

Shims have two different places to keep state.

### Per-Run Transient State

`PreparedTurn.transient_state` is shared across all turns in one run and is
discarded when the run ends.

It is typed as
[`RunContext`](../../src/agentlane/models/run/_context.py) and uses
[`DefaultRunContext`](../../src/agentlane/models/run/_context.py) as the
default implementation. This is the right place for temporary cached values or
other in-memory state that should not survive a resumed run.

### Persisted Resumable State

`PreparedTurn.run_state.shim_state` is copied into the saved
[`RunState`](../../src/agentlane/harness/_run.py).

It is typed as [`ShimState`](../../src/agentlane/harness/_run.py), a persisted
mapping-backed state container for shim-owned resumable state.

Use this when a shim needs state to survive:

1. later `run(...)` calls on the same `DefaultAgent`,
2. explicit `RunState` resume paths, or
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

This gives a shim one clean place to prepare a turn, one clean place to
observe the completed model response, and one clean place to finalize per-run
work.

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
session. One `Shim` definition may then be safely reused across many agents
without leaking mutable state between them.

For a runnable example, see
[examples/harness/default_agent_shims_quickstart](../../examples/harness/default_agent_shims_quickstart/README.md).

## Boundaries

Shims are meant to change run behavior. [`RunnerHooks`](../../src/agentlane/harness/_hooks.py)
are not. Hooks stay observation-only for tracing, logging, and tests.

Shims also do not change runtime delivery behavior. They operate inside the
harness lifecycle and runner once work has already been accepted by a bound
agent instance.
