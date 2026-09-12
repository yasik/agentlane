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
4. `BridgeCommandHandler`
5. `RuntimeConfigStore`
6. `ConfigRejectedError`
7. `serve_stdio`
8. `run_stdio`

The backend accepts one active prompt at a time, streams AgentLane
`RunEvent.to_dict()` records inside `run_event.event`, routes diagnostics to
stderr, and closes active streams with AgentLane's `aclose()` then `result()`
drain pattern during cancel, reset, shutdown, and EOF teardown.

The bridge sends complete content and preserves JSON-compatible result
structure. Apps own display limits. Native events and complete final results
use strict harness conversion. Unsupported values, cycles, nonfinite numbers,
and prompt rendering failures cause a controlled run error. Ordinary
bridge-owned metadata retains its existing fallback; ready metadata and runtime
config remain strict JSON.
See [Payload Values](../../docs/process-bridge/protocol.md#payload-values) for
result fields and serialization behavior.

App-facing TypeScript consumers should usually launch the backend through:

```bash
uv run python -m agentlane_process_bridge --app my_app.backend:create_backend
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

`BridgeEventType` contains bridge controls and the native `run_event` envelope.
The complete native record retains its source kind, nesting, and lineage.
Model completion and errors remain native model records; only the authoritative
stream result determines run success.

Command handling is registry-based.
`BridgeBackend` accepts an explicit command-handler tuple and defaults to
`BRIDGE_COMMAND_HANDLERS`; each command handler declares the command class it
handles and owns that command's side effects. Native delivery uses no per-event
encoder or handler registry.

The host app receives full instructions, history, tool results, prompt values,
and provider data over its child process pipe. The bridge does not redact these
fields. The host must authorize any forwarding or storage.

Use Python and TypeScript packages with the same native-event contract. Protocol
version `1.0` is unchanged, but the previous flat event contract is incompatible.

## Developer Workflow

See [Process Bridge Development](../../docs/process-bridge/development.md) for
the command, config, run-event, and bridge-only event extension steps.
