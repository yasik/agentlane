# Run Event Serialization

Every concrete run event exposes `to_dict()` and `dumps()`. These methods are part
of the core package and do not require the process bridge.

```python
from agentlane.harness import RunAgentStartEvent

event = RunAgentStartEvent(task_name="Example", task_id="task-1")
record = event.to_dict()
text = event.dumps()
```

`to_dict()` returns a new `RunEventRecord`, a typed dictionary with two fields:

- `type: str`: the source `RunEventKind` value, `event.kind.value`.
- `payload: dict[str, JsonValue]`: the complete converted event fields with
  their original nesting.

`dumps()` returns the same record as compact JSON text. It preserves Unicode,
escapes embedded newlines, and rejects nonfinite numbers. It adds no line
terminator, SSE framing, or host metadata. Both methods use the same private
conversion code. There is no public standalone serializer function.

Use `to_dict()` when a host adds its own envelope; no JSON decode step is needed.
Use `dumps()` when the complete record can be written directly to a transport.

## Final Results

`RunResult.to_dict()` returns a new `RunResultRecord` with the complete converted
`final_output`, `responses`, `turn_count`, and `run_state` fields. It uses the
same internal conversion code as events. It adds no event kind or transport
envelope. Import both public result types from `agentlane.harness`.

```python
result = await stream.result()
record = result.to_dict()
```

The method converts every field before returning. An unsupported response or
state fails conversion even when the caller only needs `final_output`. The
record is an inspection representation, separate from snapshot restoration.

## Native Event Structure

All run events keep their `RunEventKind` name and all dataclass fields,
including the `kind` field at `payload.kind`. The serializer does not filter
child events or flatten model and approval wrappers.

`RunModelStreamEvent` always has `type="model_stream"`. All inner model fields
stay at `payload.event`, including the model kind at `payload.event.kind`.
Text deltas, reasoning, tool-call argument deltas, provider events, completion,
and errors keep their source kinds. They do not become `assistant_delta`,
`reasoning_delta`, `tool_arguments_delta`, `provider_event`, or a top-level
`error` event. Empty text deltas and model completion events are retained.
A model error does not assert whole-run failure.

`RunToolApprovalEvent` always has `type="tool_approval"`. Its inner event stays
at `payload.event`, its complete record at `payload.event.record`, and its
approval status at `payload.event.record.status`. Neither `pending` nor
`resolved` changes the type to `approval_request` or `approval_resolved`.

The serializer adds no synthetic `scope` or `run_event_kind` fields. Source
fields with these names, when present in supported values, remain unchanged.
Transport envelopes belong to the host.

The methods do not create `run_start`, `run_complete`, or `run_cancelled`
notifications. They do not read the source stream or determine its outcome.
After consuming the stream, the host must check its authoritative `result()`.
An `agent_end` event alone does not prove successful completion.

## Value Conversion

| Source value | JSON representation |
| --- | --- |
| Null, string, boolean, integer, finite float | Unchanged |
| Enum | Converted enum value |
| UUID or path | Complete string |
| Mapping with string keys | Object with converted values |
| List or tuple | Array in source order |
| Dataclass | Object containing all fields |
| Pydantic `BaseModel`, including SDK models derived from it | Object containing raw field values and extras, including nulls |
| Pydantic `RootModel` | Converted root value |
| Exception | Object with `type` and `message` |
| `PromptSpec` | Rendered `messages`, original `values`, and `response_format` |

There are no string or collection limits. Reasoning signatures and raw provider
metadata remain present when supplied. Shared object references are copied into
each JSON position; only ancestor cycles are rejected.

Pydantic conversion uses field iteration, not model JSON serializers. It retains
subclass fields and extras, without applying aliases, exclusion rules, computed
fields, or custom field serializers. A field value must itself be supported.
This avoids losing nested framework values before their conversion is applied.

`ShimState` is both a dataclass and a mapping. Its public mapping contents are
serialized; its runtime locks are not.

Structured tool results remain structured JSON values for UI use and business
logic. The serializer does not replace them with model-facing text.

### Stored Prompts And Model Requests

For `PromptSpec`, the serializer calls the public template methods
`render_messages(values)` and `response_format()`. Its `messages` field is a
rendered view of the stored template and values. It does not necessarily contain
the messages sent to the model client. Template rendering errors propagate to
the caller.

`RunLLMStartEvent.messages`, serialized at `payload.messages`, contains the
messages passed to the model client after harness transformations, including
shim message transformations. Use this field to inspect the model request
messages. A shim can change these messages without changing the stored
`PromptSpec`, so the two rendered message lists can differ. This field does not
describe later provider-specific wire conversion inside the model client.

Serialized run events are **not restorable agent snapshots**. The serializer
does not persist executable template code. Keep snapshot storage and restoration
separate from event serialization.

## Failures and Access

Unsupported values, non-string object keys, bare prompt templates, ancestor
cycles, and nonfinite numbers raise errors. Unsupported values include bytes,
sets, and dates. There is no text fallback, empty substitute, or partial result.
Model kinds and approval statuses are converted as source values, not selected
through a transport-name mapping.

The host must authorize access before it sends these payloads. Source events can
contain instructions, full history, tool data, and provider metadata. Exception
messages are preserved, not sanitized; traceback and runtime attributes are not
included. This serializer is not a redaction boundary.

The host owns SSE or WebSocket framing, session and run IDs, run-error policy,
stream cleanup, and persistence. The methods change neither source objects nor
the set of streams selected by the host.

If serialization fails or the consumer disconnects, close the stream in `finally`
and retrieve its result, handling failure or cancellation. See
[Stream Cancellation And Closure](runner.md#stream-cancellation-and-closure).

## Other Serialization APIs

- `ModelStreamEvent.to_trace_dict()` is for traces. It omits null fields, can omit
  raw payloads, and can use text fallbacks.
- `agentlane.transport` codecs serialize registered message types. They do not
  define the native harness event structure or prompt representation above.
- The process bridge places `RunEvent.to_dict()` unchanged at `run_event.event`.
  It uses `RunResult.to_dict()` after `stream.result()` for completion. Native
  event and result conversion failures cause controlled run errors; the bridge's
  ordinary control-value text fallback never applies to them. See
  [Protocol and Lifecycle](../process-bridge/protocol.md).

New transports can use these methods for complete native inspection records and
add their own framing and failure policy.
