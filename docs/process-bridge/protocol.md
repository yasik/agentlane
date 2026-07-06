# Protocol and Lifecycle

The session API uses a strict stdio protocol internally. Every command and
event is one JSON object followed by `\n`.

Stdout is reserved for protocol events. Python logging and diagnostics must go
to stderr; `run_stdio(...)` configures that before emitting the ready event.

Every protocol object carries:

1. `protocol_version`
2. `type`

Events also carry `ts`, a Unix timestamp rounded to milliseconds. Event fields
stay flat for app consumption.

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

The bridge serializes AgentLane's existing `RunEvent` vocabulary:

1. model text, reasoning, tool-argument, provider, and error events
2. agent, LLM, tool, and handoff lifecycle events
3. plan updates
4. tool approval requests and resolutions
5. state snapshots
6. runtime config settlements and ready/reset config announcements
7. run start, completion, cancellation, reset, cancel, shutdown, and errors

`BridgeEventType` is the bridge wire vocabulary. Bridge-only lifecycle values
live in the process-bridge package; values that correspond to AgentLane run
events derive from upstream `RunEventKind` or `HarnessEventType` values so
bridges and apps do not duplicate framework event literals.

Command handling and run-event encoding use explicit handler registries. When
AgentLane adds a new `RunEventKind`, the bridge extension path is one
`RunEventBridgeHandler` implementation that declares the upstream kind, event
class, downstream `BridgeEventType` values, and encoder logic. When the bridge
adds a command, the extension path is one `BridgeCommandHandler` implementation
that declares the command class and owns the side effects. `BridgeBackend` and
`RunEventEncoder` accept explicit handler tuples and default to
`BRIDGE_COMMAND_HANDLERS` and `RUN_EVENT_BRIDGE_HANDLERS`; fixture parity tests
fail until new upstream run-event kinds are covered.

Lineage fields such as `task_id`, `parent_task_id`, `is_root`, and
`is_subagent` are preserved on task-carrying events. `tool_end` events include
the framework-derived `ok` flag and typed `error` payload instead of asking the
app to infer success from result text.

Unhandled AgentLane run-event classes encode as the explicit `run_event`
diagnostic event. Unknown bridge protocol event names and invalid payloads fail
strict TypeScript decoding and are reported as `BridgeDecodeError` values
instead of being delivered to app reducers.

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
