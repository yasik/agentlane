# agentlane-process-bridge

`agentlane-process-bridge` is the Python side of AgentLane's local stdio bridge.
It lets an app run a Python AgentLane backend in a child process while a
TypeScript host sends commands over stdin and receives versioned NDJSON events
over stdout.

Use this package when the application shell is TypeScript but the agent runtime
is Python and local to the same machine. It is not a distributed runtime,
network transport, sandbox, or UI reducer.

The main public entrypoints are:

1. `AgentBackend`
2. `BridgeBackend`
3. `EventWriter`
4. `RunEventEncoder`
5. `BridgeCommandHandler`
6. `RunEventBridgeHandler`
7. `RuntimeConfigStore`
8. `ConfigRejectedError`
9. `serve_stdio`
10. `run_stdio`

The backend accepts one active prompt at a time, streams AgentLane
`RunEvent` values as bridge events, routes diagnostics to stderr, and closes
active streams with AgentLane's `aclose()` then `result()` drain pattern during
cancel, reset, shutdown, and EOF teardown.

App-facing TypeScript consumers should usually launch the backend through:

```bash
python -m agentlane_process_bridge --app my_app.backend:create_backend
```

The referenced factory may return an `AgentBackend`, an awaitable
`AgentBackend`, or a bare `AgentRuntime` for approval-free agents.

When the host wires its agent's tool `approval_callback` to a specific
`ToolApprovalBroker`, it must pass that same broker to `BridgeBackend` /
`run_stdio` via the `approvals` parameter. The agent's pending requests and the
bridge's `approve`/`cancel` commands then resolve against one broker instance;
otherwise interactive approvals never complete. When `approvals` is omitted the
backend creates its own broker (the right default for agents that do not gate
tools on approval).

Apps that expose model or runtime settings can pass a `RuntimeConfigStore` to
`AgentBackend.config`. The store receives opaque top-level JSON patches from
the TypeScript app and returns the full authoritative config document. The
bridge does not interpret config keys; it only guarantees that `ready`, `reset`,
and `config` settlement events announce the document without truncation or fail
loudly if the document cannot be emitted safely.

For the full bridge-scoped model-settings path, including how `ready.metadata`
differs from runtime config and how a store applies selections onto
`AgentDescriptor.model` / `model_args`, see
[Process Bridge: Runtime Configuration](../../docs/process-bridge/runtime-configuration.md).

`BridgeEventType` is the bridge wire vocabulary. Bridge-only lifecycle values
live in this package; values that correspond to AgentLane run events derive
from upstream `RunEventKind` or `HarnessEventType` values so downstream code
does not duplicate framework event literals.

Command handling and run-event encoding are both registry-based.
`BridgeBackend` accepts an explicit command-handler tuple and defaults to
`BRIDGE_COMMAND_HANDLERS`; each command handler declares the command class it
handles and owns that command's side effects. `RunEventEncoder` accepts an
explicit run-event-handler tuple and defaults to `RUN_EVENT_BRIDGE_HANDLERS`;
each run-event handler declares the upstream `RunEventKind`, the event class,
the emitted `BridgeEventType` values, and its encoder implementation.

## Developer Workflow

See [Process Bridge Development](../../docs/process-bridge/development.md) for
the command, config, run-event, and bridge-only event extension steps.
