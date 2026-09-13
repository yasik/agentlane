# Protocol and Lifecycle

The session API uses a strict stdio protocol internally. Every command and
event is one JSON object followed by `\n`.

Stdout is reserved for protocol events. Python logging and diagnostics must go
to stderr; `run_stdio(...)` configures that before emitting the ready event.

Every protocol object carries:

1. `protocol_version`
2. `type`

Events also carry `ts`, a Unix timestamp rounded to milliseconds. The protocol
version is `1.0`. Native run events stay nested inside a `run_event` envelope.
Python and TypeScript packages must use this contract together. The previous
flat event contract is incompatible even though the protocol version is unchanged.

Commands:

1. `prompt` with `text`
2. `approve` with `id`, strict boolean `allowed`, and optional `reason`
3. `configure` with opaque object `patch`
4. `cancel`
5. `reset`
6. `shutdown`

Only JSON boolean `true` grants an approval. Values such as `"true"`, `1`, or
`{}` deny.

## Lifecycle

`BridgeBackend` owns one active run at a time. If a prompt arrives while a run
is active, the backend emits a command-scoped `error` and leaves the current run
untouched.

Cancel, reset, shutdown, run failure, EOF, and backend close all clear pending
approvals and close the active `RunEventStream` with the documented AgentLane
pattern:

```python
await stream.aclose()
with contextlib.suppress(asyncio.CancelledError):
    await stream.result()
```

That stops provider-side work and retrieves the result future so asyncio does
not report an unobserved cancellation.

## Event Model

Each source event produces one envelope whose `event` field equals
`RunEvent.to_dict()`. For example, a complete native agent-start record is:

```json
{
  "protocol_version": "1.0",
  "ts": 0,
  "type": "run_event",
  "event": {
    "type": "agent_start",
    "payload": {
      "kind": "agent_start",
      "task_name": "Example",
      "task_id": "task-1",
      "parent_task_id": null,
      "is_root": true
    }
  }
}
```

Native model events stay at `event.payload.event`, with their source `kind`.
All model fields survive, including empty deltas, reasoning signatures, raw
provider data, completion records, and errors. A model completion or error does
not settle a run or command. Only bridge `run_complete`, `run_cancelled`, and
run-scoped `error` events determine the run outcome.

Approval records stay at `event.payload.event.record`. Pending and resolved
records are delivered regardless of display callbacks. The shared Python
approval broker and host `approve` commands control execution.

Lineage remains on the source events that provide it. TypeScript presentation
helpers derive flags from those fields. The bridge adds no model lineage or
predicted turn count. Tool-end source fields retain `ok`, `error`, and the
complete structured `result`.

`BridgeEventType` contains the bridge controls and `run_event`. Command handling
uses `BRIDGE_COMMAND_HANDLERS`; native forwarding needs no Python handler per
event kind. TypeScript validates known native kinds and their required fields,
rejects contradictory known discriminators, and retains extra fields at every
native object level. Unknown native kinds reach `onEvent` intact and have no
presentation effect. Unknown outer event names and malformed records produce
`BridgeDecodeError` values and do not reach app reducers.

## Payload Values

The bridge sends complete content. It does not shorten strings, limit list or
object entries, or create result previews. Apps choose their own display
limits, folding, and pagination.

Native events and final results use the shared strict
[harness conversion](../harness/event-serialization.md#value-conversion).
Tuples become arrays, dataclasses and Pydantic models become objects, and
exceptions retain their type and message. Unsupported values, non-string keys,
cycles, nonfinite numbers, and prompt rendering errors fail conversion. Native
values have no text fallback. Conversion finishes before a record is queued.

The result fields are:

| Event | Field | Value |
| --- | --- | --- |
| `run_event` / `tool_end` | `event.payload.result` | Complete JSON value, including objects and arrays |
| `run_event` / `agent_end`, `handoff_end` | `event.payload.result.final_output` | Complete JSON value or `null`; agent result can also be `null` |
| `run_event` / `llm_end` | `event.payload.response` | Complete model response, including choices and usage |
| `run_complete` | `final_output` | Complete JSON value or `null` |

After event iteration, the backend awaits `stream.result()` and converts the
complete result with `RunResult.to_dict()`. `run_complete` selects its final
output, turn count, response count, and final shim state. An unsupported response
or state fails the run even when that whole field is absent from `run_complete`.
An earlier root `agent_end` does not prove success.

Serialization or delivery failure cancels the run token, denies pending
approvals, closes the stream, and retrieves its result. A writable pipe receives
one run-scoped `error` after cleanup, with no `run_complete`. A failed or
disconnected writer cannot guarantee a terminal frame; the TypeScript process
exit path settles the active run.

Writer failure also wakes a command loop that is waiting for stdin. The backend
can clean up and exit even when the host keeps stdin open but stops reading
stdout. The writer defaults to a 30-second timeout and a 1,024-record queue;
the timeout applies to writes, queue admission, and drain waits. Writer failure
releases producers blocked on queue admission and discards their queued records.
No records remain pending after those producers settle. Later writes and
`drain()` fail immediately with the stored failure. Setting
`write_timeout_seconds=None` disables the timeout.

Synchronous reads and writes use daemon threads so a blocked borrowed stream
does not hold the event loop or prevent interpreter shutdown. Cancelling the
wait does not interrupt the underlying stream call. Each bridge has at most
one pending read and one pending write. Normal process stdout uses direct UTF-8
descriptor writes to avoid a blocked Python output-buffer lock during exit.
The bridge does not close borrowed descriptors. Run cleanup still depends on
the agent's cooperative stream closure; the I/O timeout is not a total run
shutdown deadline.

If close fails after output is already unavailable, the backend logs
`bridge_close_after_dead_client_failed` with `error_type`. This diagnostic
omits traceback locals so queued payloads do not delay shutdown.

Ready metadata and runtime configuration require strict JSON values. Ordinary
bridge-owned payloads retain the legacy text fallback for unsupported values,
cycles, and nonfinite numbers. This includes runtime shim metadata used when the
final result has no run state. That fallback never applies to native events or
the complete final result.

The writer's `verbatim_payload` argument requires JSON-serializable values.
Its bounded queue and write timeouts preserve backpressure. Only nested model
`text_delta`, `reasoning`, `tool_call_arguments_delta`, and `provider` records
can be batched. Other native records and bridge controls flush the queue.

Validate TypeScript `RunResult.finalOutput`, agent `finalOutput`, and tool
results against the app schema before use. These values can contain structured
data. TypeScript text callbacks use string final outputs to complete or correct
streamed text. Structured final outputs are available through run results and
agent activity callbacks. Apps decide how to display them.

## Access and Model Requests

The host app selects the backend factory and owns the child process pipes.
This local transport has no network listener or separate client authorization
layer. Full native records can expose instructions, history, tool results,
prompt values, exception messages, and raw provider data to the host. The bridge
does not redact them. The host must authorize later forwarding or storage.

Stored `PromptSpec` values include a rendered view of their template and values.
Use native `llm_start` at `event.payload.messages` to inspect messages after
harness transformations. Those messages can differ from the stored prompt view.
Provider-specific wire conversion happens later. Native event and result
records are inspection data; snapshot persistence remains separate.

## Low-Level TypeScript Primitives

The TypeScript package also exports protocol, process, and channel helpers:

```ts
import {
  createBridgeChannel,
  spawnBridgeProcess,
} from "@agentlanejs/process-bridge";

const child = spawnBridgeProcess(
  {
    command: "uv",
    args: [
      "run",
      "python",
      "-m",
      "agentlane_process_bridge",
      "--app",
      "my_app.backend:create_backend",
    ],
  },
  {
    onEvent: (event) => {
      console.log(event.type);
    },
    onStderr: (line) => console.error(line),
  },
);

const channel = createBridgeChannel(child);
channel.send({ type: "prompt", text: "Summarize this case." });
```

Use the low-level helpers for tests, custom launchers, or bridge
infrastructure. Consumers that use them directly own ready gating, command
correlation, text buffering, approval resolution, operation settlement, and
lifecycle cleanup.
