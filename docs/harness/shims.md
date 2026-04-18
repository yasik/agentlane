# Harness Shims

Shims are the harness extension point for behavior that needs to change what a
run actually does.

Use a shim when you need to:

1. append or replace effective instructions,
2. add or adjust visible tools,
3. inject temporary context for the next model call,
4. update persisted run-owned state, or
5. rewrite the final model request messages before one call.

This is the mechanism intended for future features such as skills, memory, and
context compaction. Those capabilities should build on this seam instead of
adding new special-case fields to the core harness types.

## Import Path

```python
from agentlane.harness.shims import (
    BoundHarnessShim,
    HarnessShim,
    PreparedTurn,
    ShimBindingContext,
)
```

## The Two Shim Types

There are two related contracts.

`HarnessShim` is the definition-time object. It carries static configuration and
binds itself once for each concrete agent instance.

`BoundHarnessShim` is the per-agent session. It receives lifecycle callbacks and
may keep private in-memory state that belongs only to that one bound agent.

That split is important because one shim definition may be reused across many
agents without leaking mutable state between them.

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

1. `instructions`
2. `tools`
3. `model_args`
4. `run_state`
5. `context_items`
6. `transient_state`

The intended use is:

1. change `instructions`, `tools`, `model_args`, or `context_items` in
   `prepare_turn(...)`,
2. update persisted state in `run_state.shim_state`,
3. use `transform_messages(...)` only when you need one-call message surgery
   after the runner has already built the canonical message list.

`context_items` uses the same run-history item contract as the rest of the
harness request builder. Supported items include:

1. canonical message dicts,
2. prior `ModelResponse` values,
3. `PromptSpec` values,
4. user-side content values such as strings, JSON-like values, or Pydantic
   models.

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
from agentlane.harness.shims import BoundHarnessShim, HarnessShim, PreparedTurn


class ReplyPrefixBoundShim(BoundHarnessShim):
    async def prepare_turn(self, turn: PreparedTurn) -> None:
        if isinstance(turn.instructions, str):
            turn.instructions = (
                f"{turn.instructions}\nAlways start every reply with `Support:`."
            )


class ReplyPrefixShim(HarnessShim):
    @property
    def name(self) -> str:
        return "reply-prefix"

    async def bind(self, context: ShimBindingContext) -> BoundHarnessShim:
        del context
        return ReplyPrefixBoundShim()


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

For a runnable example, see
[examples/harness/default_agent_shims_quickstart](../../examples/harness/default_agent_shims_quickstart/README.md).

## Boundaries

Shims are meant to change run behavior. [`RunnerHooks`](../../src/agentlane/harness/_hooks.py)
are not. Hooks stay observation-only for tracing, logging, and tests.

Shims also do not change runtime delivery behavior. They operate inside the
harness lifecycle and runner once work has already been accepted by a bound
agent instance.
