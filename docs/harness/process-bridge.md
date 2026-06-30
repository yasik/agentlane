# Harness Process Bridge

The process bridge is for local apps that want a TypeScript shell around a
Python AgentLane backend. The TypeScript side starts a Python process, writes
newline-delimited JSON commands to stdin, reads newline-delimited JSON events
from stdout, and treats stderr as diagnostics only.

Use it when:

1. the app UI or terminal shell is TypeScript
2. the agent implementation, tools, and model clients live in Python
3. the backend runs as a local child process owned by the app
4. the app wants high-level `run_events(...)` lifecycle data instead of raw
   model deltas only

Do not use it as a distributed runtime replacement. It does not route work to
remote workers, expose a network server, sandbox the Python process, or define
app state. Distributed agents should keep using the runtime and messaging
primitives under `agentlane.runtime` and `agentlane.messaging`.

## Protocol

Every command and event is one JSON object followed by `\n`.

Stdout is reserved for protocol events. Python logging and diagnostics must go
to stderr; `run_stdio(...)` configures that before emitting the ready event.

Every protocol object carries:

1. `protocol_version`
2. `type`

Events also carry `ts`, a Unix timestamp rounded to milliseconds. Event fields
stay flat for easy app consumption.

Commands:

1. `prompt` with `text`
2. `approve` with `id`, strict boolean `allowed`, and optional `reason`
3. `cancel`
4. `reset`
5. `shutdown`

Only JSON boolean `true` grants an approval. Values such as `"true"`, `1`, or
`{}` deny.

## Lifecycle

`BridgeBackend` owns one active run at a time. If a prompt arrives while a run
is active, the backend emits a command-scoped `error` and leaves the current
run untouched.

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
6. run start, completion, cancellation, reset, cancel, shutdown, and errors

`BridgeEventType` is the bridge wire vocabulary. Bridge-only lifecycle values
live in the process-bridge package; values that correspond to AgentLane run
events derive from upstream `RunEventKind` or `HarnessEventType` values so
bridges and apps do not duplicate framework event literals.

Command handling and run-event encoding use explicit handler registries. When
AgentLane adds a new `RunEventKind`, the bridge extension path is one
`RunEventBridgeHandler` implementation that declares the upstream kind, event
class, downstream `BridgeEventType` values, and encoder logic. When the bridge
adds a command, the extension path is one `BridgeCommandHandler`
implementation that declares the command class and owns the side effects.
`BridgeBackend` and `RunEventEncoder` accept explicit handler tuples and
default to `BRIDGE_COMMAND_HANDLERS` and `RUN_EVENT_BRIDGE_HANDLERS`; fixture
parity tests fail until new upstream run-event kinds are covered.

Lineage fields such as `task_id`, `parent_task_id`, `is_root`, and
`is_subagent` are preserved on task-carrying events. `tool_end` events include
the framework-derived `ok` flag and typed `error` payload instead of asking the
app to infer success from result text.

Unknown future run-event classes encode as `run_event`. Unknown future protocol
events decode on the TypeScript side as `unknown_event` with the raw payload.

## TypeScript Consumer

The TypeScript package exports command, decoder, process, and channel helpers:

```ts
import {
  createBridgeChannel,
  spawnBridgeProcess,
} from "@agentlane/process-bridge";

const child = spawnBridgeProcess(
  { command: "uv", args: ["run", "python", "backend.py"] },
  {
    onEvent: (event) => {
      // App reducer owns rendering and state.
      console.log(event.type);
    },
    onStderr: (line) => console.error(line),
  },
);

const channel = createBridgeChannel(child);
channel.send({ type: "prompt", text: "Summarize this case." });
```

See
[`examples/harness/process_bridge_stdio`](../../examples/harness/process_bridge_stdio/)
for a runnable no-model-key smoke example.
