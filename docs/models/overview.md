# Models Overview

The models layer gives the rest of AgentLane a stable way to describe an LLM
interaction without baking provider details into application code. This is
where prompts are shaped, tools become model-visible, and structured outputs
are defined.

In day-to-day code, that usually means defining a
[`PromptTemplate`](../../src/agentlane/models/_prompts.py) or
[`MultiPartPromptTemplate`](../../src/agentlane/models/_prompts.py), binding
run-specific values with [`PromptSpec`](../../src/agentlane/models/_prompts.py),
describing tool access with [`Tools`](../../src/agentlane/models/_interface.py)
and [`@as_tool`](../../src/agentlane/models/_tool.py), and declaring the
expected response shape with
[`OutputSchema`](../../src/agentlane/models/_output_schema.py). Provider
packages then implement the shared
[`Model`](../../src/agentlane/models/_interface.py) and
[`Config`](../../src/agentlane/models/_interface.py) boundary instead of
inventing a different request shape for every integration.

## Prompts, Tools, And Output

The three pieces fit together:

1. prompt templates describe the input you want to send
2. tools describe the actions the model may call during a run
3. output schemas describe the shape you expect back

In practice, a typical flow looks like this:

1. build a prompt from typed values
2. expose a small tool set
3. decide whether the result should be plain text or structured output

That is the reason the models layer feels lower-level than the harness but
higher-level than a provider SDK. It does not run an agent loop. It gives the
rest of the framework a stable language for talking to models.

## Prompt Templating

Prompt templating is described in more detail in
[Models: Prompt Templating](./prompt-templating.md), but the short version is
that templates keep prompt structure separate from prompt values.

That separation matters because it lets you:

1. reuse the same prompt shape across runs
2. keep prompt values typed and explicit
3. hand prompts to the harness without flattening them into raw message dicts

## Tool Policy

The common tool paths are intentionally lightweight. Use
[`@as_tool`](../../src/agentlane/models/_tool.py) when a function should read
as a tool definition at the declaration site. Use
[`Tool.from_function(...)`](../../src/agentlane/models/_tool.py) when you want
an explicit tool value in code.

[`Tools(...)`](../../src/agentlane/models/_interface.py) is the higher-level
piece. It combines the visible tool set with request policy such as
`tool_choice`, `parallel_tool_calls`, per-tool call limits, per-call timeout
and retry-on-timeout (`tool_call_timeout`, `tool_call_max_retries`), and the
overall round-trip safety limit. That policy belongs in the models layer
because it describes what the model is allowed to ask for. The harness then
decides how those tool calls are executed inside a run.

### Wrapping And Copying Tools

To adapt an existing tool without rebuilding it field by field, use the copy
API on [`Tool`](../../src/agentlane/models/_tool.py):

- `tool.replace(**overrides)` returns a copy with selected fields changed and
  every other field preserved by construction (mirroring
  `dataclasses.replace`). Unspecified fields — including a custom `formatter`
  and an explicit `parameters_schema` — are carried through unchanged, so a
  copy that only meant to change one thing cannot silently drop another.
- `tool.with_handler(wrapper)` returns a copy whose handler is
  `wrapper(tool.handler)`. This is the standard shape for argument
  interception, result post-processing, sandboxing, or redaction: the wrapper
  receives the original handler and returns the replacement. `tool.handler` and
  `tool.formatter` are also exposed as read-only properties for composition.

Prefer these over reconstructing `Tool(name=..., handler=..., ...)`: rebuilding
from parts must enumerate every field, and a future `Tool` field would be lost
by callers that forgot to forward it.

## Provider Boundary

This layer is also where provider-specific behavior is contained.

The shared [`Model`](../../src/agentlane/models/_interface.py) contract defines
what the harness expects from a model client. Integration modules such as
`agentlane_openai` and `agentlane_litellm` adapt concrete providers to that
contract instead of pushing provider-specific request shapes up into the rest of
the framework. Install LiteLLM support with `agentlane[litellm]`.

That is why `Config` stays narrow. It covers shared networking and control-plane
concerns. Model-specific options are passed through per-call arguments instead
of being forced into one large global settings object.

## Streaming

The shared [`Model`](../../src/agentlane/models/_interface.py) contract now
supports both:

1. `get_response(...)` for terminal-response workflows
2. `stream_response(...)` for provider-grounded event streams

`stream_response(...)` yields
[`ModelStreamEvent`](../../src/agentlane/models/_streaming.py) items. The
normalized event kinds stay small on purpose:

1. `text_delta`
2. `tool_call_arguments_delta`
3. `reasoning`
4. `completed`
5. `error`
6. `provider`

That split matters because providers do not all stream the same way.

The base `Model.stream_response(...)` is a non-streaming fallback: it delegates
to `get_response(...)` and emits a single terminal `completed` event (or an
`error` event on failure). Real incremental streaming requires a provider
override.

`agentlane-openai` can preserve semantic Responses API events, including output
deltas and reasoning-related events. `agentlane-litellm` preserves the chunk
shape LiteLLM documents publicly and then assembles a final canonical
`ModelResponse` when the stream completes.

The normalized fields are there for the common framework cases. The original
provider event or chunk is still attached on `ModelStreamEvent.raw`, together
with `provider_event_type`, so provider-specific detail is not lost at the
adapter boundary.

The harness now reuses this same event type directly. High-level agent
streaming through `DefaultAgent.run_stream(...)` yields `ModelStreamEvent`
through a harness `RunStream` handle instead of wrapping the per-event payload.

### Typed accessors for common provider shapes

A few provider-native shapes are only reachable through the preserved raw
payload. Rather than duck-typing them with `getattr`, use the typed accessors
exported from `agentlane.models`:

1. `get_reasoning_phase(event)` reads the OpenAI Responses output-message phase
   off `ModelStreamEvent.raw` for `output_item` events. It returns a
   `ReasoningPhase` (`COMMENTARY` or `FINAL_ANSWER`) or `None`. `None` means
   "no phase reported" — the event is not an output-item event, the item type
   carries no phase, or the provider sent no phase — and never signals a
   failure.
2. `get_usage_totals(response)` returns a typed `UsageTotals` view over
   `ModelResponse.usage`, exposing `prompt_tokens`, `completion_tokens`, and
   `total_tokens`. It returns `None` when the response (or its usage) is
   missing, which callers should treat as missing data rather than zero.

```python
from agentlane.models import (
    ReasoningPhase,
    get_reasoning_phase,
    get_usage_totals,
)

phase = get_reasoning_phase(event)
if phase is ReasoningPhase.FINAL_ANSWER:
    ...

totals = get_usage_totals(response)
if totals is not None:
    track(totals.total_tokens)
```

## Run-Scoped Context

Some model-adjacent helpers live under `agentlane.models.run`.

They are useful when a run needs local state or tracing support that should not
be sent to the model. The main ones are
[`DefaultRunContext`](../../src/agentlane/models/run/_context.py) for
dictionary-backed run-local state and
[`TraceCtxManager`](../../src/agentlane/models/run/_ctx_managers.py) for
ensuring a trace exists when model work begins.

## Tool Execution Context

Tool handlers receive a
[`ToolExecutionContext`](../../src/agentlane/models/_tool.py) carrying
correlation identity (`run_id`, `agent_name`, `tool_call_id`) plus
host-application `metadata`. The harness also attaches a read-only view of the
live run on `context.run_state`, typed as
[`RunStateView`](../../src/agentlane/models/_tool.py), so a tool can observe the
run it is executing inside without an app-built side channel:

- `run_state.task_id` — the stable task identity for the run.
- `run_state.shim_state` — the persisted shim-owned state, read-only.
- `run_state.active_skill_names` — the names of skills active for the run.

`run_state` is `None` when a tool runs outside a runner loop (for example a
direct `tool.run(...)` call). When present, `context.run_id` is guaranteed to
equal `str(run_state.task_id)`, so per-run stores may be keyed by either field
consistently. `RunStateView` is a read surface only; a tool that needs to
persist state should do so through its own shim.

`RunStateView` is a structural protocol defined in the models layer; the
harness supplies the concrete live implementation. A skills shim records active
skill names under a `shim_state` key ending in
`ACTIVE_SKILL_NAMES_STATE_KEY_SUFFIX`
([`agentlane.harness`](../../src/agentlane/harness/_run.py)), which is what
`active_skill_names` reads.

## Example

```python
from typing import TypedDict

from agentlane.models import OutputSchema, PromptSpec, PromptTemplate, Tools, as_tool


class RiskVars(TypedDict):
    portfolio_tier: str


risk_prompt = PromptTemplate[RiskVars, str](
    system_template="You review {{ portfolio_tier }} portfolio risk alerts.",
    output_schema=OutputSchema(str),
)


@as_tool
async def search_risk_policy(question: str) -> str:
    """Search the risk policy library."""
    del question
    return "Single-sector exposure above 35% requires risk review."


instructions = PromptSpec(
    template=risk_prompt,
    values={"portfolio_tier": "institutional"},
)

tools = Tools(tools=[search_risk_policy])
```

## Related Docs

1. [Models: Prompt Templating](./prompt-templating.md)
2. [Harness Architecture](../harness/architecture.md)
3. [Harness Runner](../harness/runner.md)
4. [Transport Serialization](../transport/serialization.md)
