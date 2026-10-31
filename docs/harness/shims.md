# Shims

Shims are the harness extension point for behavior that needs to change what a
run actually does.

Use a shim when you need to:

1. adjust the single persisted system instruction,
2. add or adjust visible tools,
3. append visible items to the persisted conversation history,
4. update shim-owned persisted state, or
5. perform one-call message rewriting as an advanced escape hatch.

This is the mechanism intended for features such as first-party tools, skills,
memory, and context compaction. Those features should build on this seam instead
of adding new special-case fields to the core harness types.

First-party examples include [Tools](./tools/README.md) and
[Harness Skills](./skills.md). For the history-rewrite pattern used by
conversation compaction, see [Harness Compaction](./compaction.md).

## Import Path

```python
from agentlane.harness.shims import (
    BoundShim,
    DelegatingBoundShim,
    DelegatingShim,
    PreparedTurn,
    Shim,
    ShimBindingContext,
    ToolNameCollisionError,
    ToolSourceBinding,
)
```

The public surface is `Shim`, `BoundShim`, `DelegatingBoundShim`,
`DelegatingShim`, `PreparedTurn`, `ShimBindingContext`, `ToolNameCollisionError`,
and `ToolSourceBinding`.

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
    name="Clinical Assistant",
    model=model,
    instructions="You are a clinical workflow assistant.",
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
5. `replace_history(...)`
6. `exclude_tools(...)`
7. `add_tools(...)`

Use them like this:

1. change `tools` or `model_args` in `prepare_turn(...)`,
2. append to the persisted system instruction only when the run needs a
   bootstrap-time behavior change,
3. append visible conversation items to `RunState.history` for ongoing changes,
4. replace the whole history only for persistent rewrite behavior such as
   conversation compaction,
5. write resumable shim-owned state to `RunState.shim_state`.

Use `exclude_tools(...)` to remove names after all shims prepare the turn.
Directly filtering `turn.tools` affects only the tools present at that step.
Exclusions preserve the tool settings during preparation, including when no
tools remain, so later contributions keep those settings.

Use `add_tools(...)` to merge tool definitions while preserving tool settings
and collision checks. Local contributions keep first-wins precedence. A
source that requires unambiguous names, such as MCP, passes
`require_unique_names=True`.
A conflicting contribution then raises `ToolNameCollisionError` in either
shim order, even if another shim filtered the earlier tool out of the current
configuration. Custom shims that assign `turn.tools` directly bypass this
registration check.

The runner applies tool-call limits and round-trip limits after all shims
prepare the turn. If no tools remain, the model receives no tool configuration,
including a previously required tool choice.

`append_history_item(...)` uses the same run-history item contract as the rest
of the harness request builder. Supported items include:

1. canonical message dicts,
2. prior `ModelResponse` values,
3. `PromptSpec` values,
4. user-side content values such as strings, JSON-like values, or Pydantic
   models.

`replace_history(...)` uses the same item contract and copies the supplied
items before installing them. It is the sanctioned whole-history rewrite point
for compaction and similar persisted history transformations.

### Internal context messages

Mark host-generated context with
[`internal_message()`](../models/overview.md#internal-message-metadata) before
appending it to history. Use `is_internal_message(item, origin=...)` to find
context owned by your shim. Keep provider fields such as `name` for their
provider-defined purpose.

To preserve a cached request prefix, append a context message only when its
contents change or compaction removes it. Keep earlier messages and system
instructions unchanged. When appending from `transform_messages(...)`, append
the marked message to both `turn` history and the returned message list.
The metadata helpers do not inject or deduplicate context.

The model receives the context contents. Stored history, live hooks, and
snapshots retain its runtime metadata. Event `to_dict()` and `dumps()` and
result `to_dict()` omit internal messages automatically, including messages
in nested history and model requests. For details, see
[Internal messages](event-serialization.md#internal-messages).

### Run-scoped context

Use an ephemeral message (`internal_message(..., ephemeral=True)`) for context
that is not correct after the run, such as the current UI view. The model
receives it on each model call of the run. The harness removes it from the
state that the run commits. See
[Ephemeral context](../models/overview.md#ephemeral-context).

A host can send the message after the user input in list input:
`agent.run([user_text, view])`. A shim can append it to `state.history` in
`on_run_start(...)`. The message then follows the run input:

```python
from agentlane.harness.shims import Shim
from agentlane.models import internal_message


class ViewContextShim(Shim):
    @property
    def name(self) -> str:
        return "ui-view"

    async def on_run_start(self, state, transient_state) -> None:
        state.history.append(
            internal_message(
                {"role": "user", "content": "The user views report.csv."},
                origin="ui_view",
                ephemeral=True,
            )
        )
```

Do not append ephemeral context in `prepare_turn(...)`. That hook runs on each
turn, so the history then contains one copy for each turn.

Prompt caching: during the run, the message stays at the same position, so
each model call of the run can use the cached request prefix. The next run
does not contain the message, so the request prefix changes at its old
position. The provider then does not use the cache for the messages after
that position. For context that changes slowly, use a persistent internal
message and append it only when its contents change.

## Shim State

Shims have two state buckets.

### Transient Per-Run State

`PreparedTurn.transient_state` is shared across all turns in one run and is
discarded when the run ends.

It is typed as
[`RunContext[Any]`](../../src/agentlane/models/run/_context.py) and uses
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
validate_request(...)
        |
        v
on_request_prepared(...)
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

1. `on_run_start(...)` for per-run setup that mutates the working state before
   the first turn. It receives the working `RunState` and `transient_state`.
2. `prepare_turn(...)` for instruction updates, tool changes, and history
   appends before the next model call,
3. `on_model_response(...)` for updating shim-owned state from the completed
   response,
4. `transform_messages(...)` only when you need one-call message surgery after
   the runner has already built the canonical request.

`validate_request(turn, messages)` runs after all message transforms. Use it to
check the complete request and raise an error before the model call. Validators
must not mutate the request. If preparation, a message transform, or validation
fails, the shim manager restores the history from before turn preparation.
This rollback covers history; it does not undo external side effects or changes
to instructions and shim state.

`on_request_prepared(turn, messages, error)` observes the result after all
validators succeed or after history rollback on failure. `error` is `None` on
success and contains the original exception on failure. Use this callback for
request diagnostics. It must not change the request. Callback errors are
ignored, so diagnostics cannot replace the request result or its original
error.

`on_run_end(...)` is the teardown callback. It receives the final
`RunResult | None` and the same `transient_state`, and runs once after the
last turn. It also runs if that session's `on_run_start(...)` began but did not
finish. Cleanup follows shim order and continues if one callback fails. A
cleanup failure does not replace the original startup or run error or
cancellation.

Cancelling the run's `CancellationToken` interrupts awaited shim startup and
turn preparation. Cleanup still runs for each session whose startup began.
The cancelled run token does not interrupt cleanup, so teardown callbacks can
await resource release.

If there is no earlier error or cancellation, cancellation during cleanup
is propagated as cancellation. Other cleanup failures are reported together
in an exception group. See [run cancellation](./runner.md#stream-cancellation-and-closure)
for the public result conventions.

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
                "Always start every reply with `Care note:`.",
                separator="\n",
            )


agent = DefaultAgent(
    descriptor=AgentDescriptor(
        name="Clinical Assistant",
        model=model,
        instructions="You are a concise clinical workflow assistant.",
        shims=(ReplyPrefixShim(),),
    )
)

result = await agent.run("Draft follow-up guidance for a patient with new bruising.")
```

## Advanced Bound Sessions

Use `BoundShim` only when a shim needs private per-agent in-memory state or
custom setup when the shim is attached to one concrete agent instance.

In that case, override `Shim.bind(...)` and return a custom `BoundShim`
session. One `Shim` definition may then be reused across many agents without
leaking mutable state between them. The bind signature is:

```python
async def bind(self, context: ShimBindingContext) -> BoundShim: ...
```

`ShimBindingContext.task` identifies the harness task or agent that owns the
session. `tool_source_bindings` contains restricted factories for inherited
dynamic tool sources and defaults to an empty tuple. Wrappers must forward
the complete context to the inner definition's `bind(...)` method.
When this tuple is nonempty, it is the source allowlist for that definition.
Dynamic sources without an identity match must stay inactive.

Shims may also contribute additional runner hooks through `runner_hooks()`.
This is useful when a shim should register tracing, logging, database writes,
script execution, or other lifecycle-triggered actions automatically with the
bound agent runtime. A plain `Shim` subclass can override `runner_hooks()`
directly; the default bound session forwards it automatically. Reserve a custom
`BoundShim` for cases that also need private per-agent in-memory state.

A bound dynamic tool source can implement `inherit_tools(names)` to bind its
tools to child agents. `names` contains the parent's visible tool names allowed
by the child policy. Return a tuple of `ToolSourceBinding` values, an empty
tuple to suppress inheritance of this definition, or `None` for normal shim
inheritance.

Each `ToolSourceBinding` has two fields:

- `source`: the original source definition, matched by object identity.
- `bind`: an async factory that accepts `ShimBindingContext` and returns one
  fresh `BoundShim` limited to the inherited names. It must create child-owned
  state and must not copy the parent's leases or executable tool handlers.

The source definition checks `context.tool_source_bindings` in `bind(...)`.
When `binding.source is self`, it calls `await binding.bind(context)`. A
matching binding takes precedence over the definition's default source
configuration, so that configuration cannot broaden the inherited tool set.
If the tuple is nonempty but has no match, return an inactive `BoundShim()`.
This also prevents a combined definition from restoring an excluded source.
The factory constructs the session with its restrictions already applied.

The harness calls the original outer definition's `bind(...)` for each child.
Its wrapper chain creates fresh bound wrapper state around the restricted
source. `bind(...)` continues to return one `BoundShim`.

For a runnable example, see
[examples/harness/default_agent_shims_quickstart](../../examples/harness/default_agent_shims_quickstart/README.md).

## Wrapping Another Shim

Composition — wrapping an existing shim to add behavior around it — should not
require re-forwarding every callback by hand. A hand-written delegate is
silently incomplete the moment the framework adds a callback: the new callback
falls through to the inert `BoundShim`/`Shim` default and never reaches the
wrapped shim.

Two delegate-by-default bases remove that hazard.

### `DelegatingBoundShim`

Wrap a bound session and override only the callbacks you change. Every other
callback, including callbacks added in future releases, forwards to the inner
bound shim automatically.

```python
from agentlane.harness.shims import BoundShim, DelegatingBoundShim, PreparedTurn


class RunStateInjectingBoundShim(DelegatingBoundShim):
    def __init__(self, inner: BoundShim, consumer: RunStateConsumer) -> None:
        super().__init__(inner)
        self._consumer = consumer

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        self._consumer.set_run_state(turn.run_state)
        await super().prepare_turn(turn)
```

The wrapped session is available as `self.inner`. Call `await super().<callback>`
inside an override to run your behavior and then forward to the inner shim.

### `DelegatingShim`

`DelegatingShim` is the definition-level analog. It forwards `name`, `bind`, and
every lifecycle callback to an inner `Shim`. The common pattern is to override
`bind(...)` only, wrapping the inner bound session in a `DelegatingBoundShim`
subclass:

```python
from agentlane.harness.shims import (
    BoundShim,
    DelegatingShim,
    ShimBindingContext,
)


class RunStateInjectingShim(DelegatingShim):
    async def bind(self, context: ShimBindingContext) -> BoundShim:
        inner = await self.inner.bind(context)
        return RunStateInjectingBoundShim(inner, self._consumer)
```

This pattern also preserves wrappers around inherited dynamic tool sources.
The outer `bind(...)` runs again for the child and forwards the restricted
source bindings through `context`. Construct a new wrapper session on each
call to keep parent and child state separate.

Both bases are reflection-tested against the full `BoundShim`/`Shim` callback
surface, so a newly added callback cannot silently bypass delegation.

## Boundaries

Shims and hooks serve different roles.

[`RunnerHooks`](../../src/agentlane/harness/_hooks.py) are general lifecycle
callback points. Hook authors may perform whatever application-defined work is
appropriate at those boundaries, including external actions.

Shims are the dedicated seam for direct harness integration such as prepared
turn mutation, message transformation, tool augmentation, and persisted
shim-owned state updates.

That distinction still holds when a shim contributes hooks automatically. A
shim may register additional hook callbacks, but direct in-harness turn and
request shaping still belongs on the shim contract itself.

Shims also do not change runtime delivery behavior. They operate inside the
harness lifecycle and runner after work has already been accepted by a bound
agent instance.
