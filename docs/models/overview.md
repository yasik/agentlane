# Overview

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

## Message roles

Use `MessageRole` from `agentlane.models` for shared message roles. It defines
`SYSTEM`, `DEVELOPER`, `USER`, `ASSISTANT`, and `TOOL` with the string values
`system`, `developer`, `user`, `assistant`, and `tool`. The harness and model
adapters use these values for message construction and role checks.

Use `.value` to construct a message with a plain string role:

```python
from agentlane.models import MessageRole

message = {"role": MessageRole.USER.value, "content": "Hello"}
```

Plain string roles remain valid inputs. Message serialization keeps the same
string values, and each provider retains its supported roles and conversion
rules.

The models package also exports shared values for tool and stream contracts:

| Enum | Values |
| --- | --- |
| `ToolChoice` | `auto`, `required`, `none` |
| `ToolErrorKind` | `error`, `timeout`, `cancelled`, `nonzero_exit` |
| `PlanStepState` | `pending`, `in_progress`, `completed` |
| `SteeringStatus` | `submitted`, `accepted`, `applied`, `failed`, `unknown` |
| `ModelContinuation` | `automatic`, `explicit` |

Use each member's `.value` in existing string fields. `ToolError.kind` also
accepts custom strings. The `PlanStepStatus` type and the string types for
tool choice, steering status, and continuation remain unchanged.

Provider adapters share `ESCAPE_SEQUENCE_RETRY_MESSAGE` and
`SCHEMA_NORMALIZATION_ERROR` for model validation feedback.

## Internal message metadata

Use `internal_message()` to mark runtime context that the model must receive
but a public conversation must omit:

```python
from agentlane.models import internal_message, is_internal_message

context = internal_message(
    {"role": "user", "content": "Available files: report.csv"},
    origin="file_inventory",
)
assert is_internal_message(context, origin="file_inventory")
```

The helper returns a copy with JSON-safe visibility and origin metadata in the
reserved `_agentlane` field. The origin must be nonempty. Use
`is_internal_message(message)` to detect context from any origin. Messages
without this metadata remain public by default.

The harness, LiteLLM client, and OpenAI adapter remove runtime metadata before
provider calls. They preserve message contents and order. Custom provider
adapters can call `provider_messages(messages)` from `agentlane.models` at
their request boundary. Keep metadata in stored history and request hooks.

History copies and `AgentSnapshot` JSON restoration retain the metadata.
For public output, apply the
[public message projection](../harness/event-serialization.md#public-message-projection).

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

### Schema and timeout policy

`ToolSpec.strict` defaults to `None`, which keeps each model adapter's default
schema mode. Set `strict=False` to preserve optional fields and open objects
in inferred schemas. An explicit `parameters_schema` is passed through
unchanged; adapters receive the strictness setting with the tool definition.

`Tool.retry_on_timeout` defaults to `True`. Set it to `False` to disable
automatic timeout retries in `ToolExecutor`, even when
`Tools.tool_call_max_retries` allows retries. [MCP tools](../harness/tools/mcp.md)
use `strict=False` and `retry_on_timeout=False`.

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

### OpenAI Responses adapter

`agentlane_openai.ResponsesClient` delegates requests to the native OpenAI SDK.
Pass SDK request options as client keyword arguments or in `extra_call_args`.
The SDK and service validate model IDs, reasoning settings, and service tiers.
The adapter retains the `max_tokens`, `reasoning_effort`, and `verbosity`
aliases. Structured-output schemas merge with your `text` settings.

The result retains the canonical `ModelResponse` interface. Its
`model_extra["openai_response"]` contains the SDK response serialized as JSON,
including ordered output items, annotations, artifacts, detailed usage, and
the returned service tier. Missing usage remains missing. An artifact-only
response can have no canonical text; read its native output to get the artifact.

Assistant messages carry native output in `openai_responses_input`. The
harness renderer and version-1 snapshots preserve this field. On replay,
`ResponsesClient` sends these native items once and skips the canonical text
and tool-call projection for that message. Keep the field when copying history.
LiteLLM rejects messages with this field. Convert history explicitly before
changing providers, and decide which native information your application can
discard.

To supply native input through either the adapter or harness, use a canonical
message with the native input field. For example, to reference a stored item:

```python
messages = [
    {
        "role": "user",
        "openai_responses_input": [
            {"type": "item_reference", "id": "ITEM_ID"},
        ],
    },
]
response = await client.get_response(messages)
```

Replace `ITEM_ID` with the provider item ID. The field accepts SDK Responses
input dictionaries. Direct adapter calls also accept native item dictionaries
in `messages`; use the envelope for harness history. Standard chat image and
file parts are converted to Responses input parts.

Native hosted declarations in `extra_call_args["tools"]` merge with the final
prepared `Tools` set. Local and MCP functions retain their strictness and
execution policies. Duplicate native/local names raise `ValueError`. When
`Tools` is supplied, register function handlers there; raw function or custom
declarations cannot add handlers outside the prepared set. Direct adapter calls
without `Tools` can send native function declarations and receive canonical
function calls for application-managed execution.

An explicit native tool choice is retained when the local choice is `auto`.
An explicit local `required` or `none` choice takes precedence. After exclusions
or budgets remove all local tools, the harness retains an empty `Tools` policy
with choice `none` so provider defaults cannot restore disabled functions.

The supported boundaries are:

| Feature | Adapter and harness behavior |
| --- | --- |
| Reasoning, cache, diagnostics, tier, and text options | Passed to the SDK; native response metadata is retained. |
| Hosted search, image generation, code interpreter, and server tool search | Native declarations pass through; returned artifacts and items are retained. |
| Direct local and MCP function calls | Run through the existing harness executor and prepared tool policies. |
| Compaction options and items | Passed through and retained; they do not coordinate a second history owner. Select one compaction policy. |
| Custom raw-text, computer, local shell, patch, hosted MCP approval, client tool search, and programmatic or namespaced calls | No local execution integration. Unsupported calls raise `ResponseError` with the native response attached. |
| Steering | Set `websocket=True` and use the active run stream's `steer(...)` method. |
| Async local and MCP function calls | Select names with `async_tools`; the harness executes complete calls while model output continues. |

`ResponsesClient.handles_retries` is `True`. The adapter owns HTTP retries;
the SDK and harness do not add retry attempts. Retries honor SDK `Retry-After`
headers, and quota or policy failures are not retried. Streams retry setup
only. A failure after streaming starts is reported without replay.
HTTP SDK exceptions retain their identity. Terminal provider failures raise
`ResponseError`; its `response` field retains the native response or event.
Token-limit and content-filter truncation retain their canonical finish reason.
Payload-free tracing excludes native input and output.

### Steering and async tools

The adapter uses OpenAI SDK `3.31.0`. The LiteLLM extras use the official
development release `1.107.0.dev2`, which accepts SDK 3. HTTP is the default
transport. Steering requires `websocket=True` and a model and account that
support the provider feature.

Run the [steering example](../../examples/harness/openai_steering/) to inspect
receipt transitions and committed history. The
[async-tool example](../../examples/harness/openai_async_tools/) checks native
async calls, overlapping tool execution, model text during execution, and final
tool results. Both examples use real OpenAI requests and fail if the expected
feature behavior is not observed.

Configure the adapter and register async functions through the existing
`Tools` interface:

```python
import os

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.models import Config, Tools
from agentlane_openai import ResponsesClient


async def lookup(topic: str) -> str:
    return f"Recorded topic: {topic}"


model = ResponsesClient(
    Config(api_key=os.environ["OPENAI_API_KEY"], model="gpt-6-astra"),
    websocket=True,
    async_tools=["lookup"],
)
agent = DefaultAgent(
    descriptor=AgentDescriptor(
        name="Research", model=model, tools=Tools([lookup]),
    ),
)
stream = await agent.run_stream("Look up battery storage and explain the result.")
receipt = None
async for event in stream:
    if event.provider_event_type == "response.created" and receipt is None:
        receipt = await stream.steer("Focus on home use.")
    if event.text:
        print(event.text, end="")
result = await stream.result()
```

Call `steer(...)` while a native response is active. The method accepts text
or a list of SDK steering input dictionaries. It returns a live
`SteeringReceipt`. Its status starts as `submitted`, then changes as native
events arrive. `accepted` means the provider queued the input. `applied`
means the provider created the successor response; the harness then adds the
input to history. `failed` retains the provider error. `unknown` means the
run ended before the outcome was confirmed. Event `steering` fields contain
receipt snapshots. Inspect the receipt or events to determine the outcome.

`run_events(...)` exposes the same `steer(...)` method. Both stream types
follow successor responses within one logical run. The adapter creates one
SDK connection per run and closes it during cleanup. Explicit continuations
retain caller settings and submit tool outputs once. Automatic successors
keep the inherited tool handlers; current exclusions, authorization checks,
and budgets still apply. Connection failures stop the run without automatic
reconnect or replay. If a steering send fails or is cancelled, the adapter
closes the connection because delivery is uncertain. Pending receipts become
`unknown`. After a handoff, steering controls the active child run. A handoff
fails while a steering continuation is pending.

`async_tools` selects executable local or MCP function names. It works with
HTTP streaming or WebSocket runs. The harness dispatches a call after the SDK
reports a complete output item, reserves its budget, and retains its call ID
and handler. Ready results continue through the same model loop. Tools use
the existing hooks, timeouts, authorization, and cleanup. Async calls disable
timeout retries; a repeated call ID does not execute twice. Cancellation
stops and joins pending calls before shim cleanup. A handoff fails while async
results are pending. Pending work is local to the run and cannot resume from
a saved snapshot.

WebSocket mode uses harness runs or `stream_response(...)` inside
`async with model.open_run() as active_model`. Direct `get_response(...)`
is not supported in this mode. Live harness runs own `previous_response_id`
and reject separate `conversation`, `context_management`, `background`,
`stream`, and `stream_id` options. A history replacement requires a fresh
request and is rejected while accepted steering is pending or an automatic
successor is active. Use the native SDK for independent conversation or
compaction control.

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
   "no usable phase reported" — the event is not an output-item event, the item
   type carries no phase, the provider sent no phase, or the provider sent a
   phase string this enum does not model. The preserved raw payload is not
   re-validated, so an unmodeled phase degrades to `None` rather than raising;
   it never signals a failure.
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

The harness enforces that read surface at the top-level `shim_state` mapping:
key insertions, replacements, and deletions through `run_state.shim_state` raise
`TypeError`. Values under those keys are still the shim owner's objects, so shims
should store immutable values when nested mutation would be unsafe.

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
