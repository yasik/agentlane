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
7. `serve_stdio`
8. `run_stdio`

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

### Add a Command

1. Add the command name to `CommandType` in
   `src/agentlane_process_bridge/_protocol.py`.
2. Add a `COMMAND_*` constant for the new command name in
   `src/agentlane_process_bridge/_protocol.py`.
3. Add the command constant to `COMMAND_TYPES`.
4. Add a frozen command dataclass with a literal `type` field in
   `src/agentlane_process_bridge/_protocol.py`.
5. Add the dataclass to the `BridgeCommand` union.
6. Add one `BridgeCommandParser` implementation in
   `src/agentlane_process_bridge/_protocol.py`.
7. Add the parser instance to `COMMAND_PARSERS`.
8. Add one `BridgeCommandHandler` implementation in
   `src/agentlane_process_bridge/_backend.py`.
9. Add the handler instance to `BRIDGE_COMMAND_HANDLERS`.
10. Re-export the command dataclass and any public handler type from
   `src/agentlane_process_bridge/__init__.py` when apps should import it.
11. Add the matching command name to `KNOWN_COMMAND_TYPES` and the command
    shape to `BridgeCommand` in `packages/process_bridge_ts/src/protocol.ts`.
12. Add or update Python protocol/backend tests and TypeScript protocol/channel
    tests.
13. Run:

    ```bash
    uv run pytest packages/process_bridge/tests -q
    /usr/bin/make lint-ts
    /usr/bin/make test-ts
    ```

### Add Run-Event Handling

1. Add the upstream run event to `agentlane.harness.RunEventKind` and the
   concrete run-event dataclass in AgentLane core first.
2. Add any new typed wire event to `BridgeEventType` in
   `src/agentlane_process_bridge/_protocol.py`. Derive the value from
   `RunEventKind` or `HarnessEventType`; do not repeat event string literals
   downstream.
3. Add one `RunEventBridgeHandler` implementation in
   `src/agentlane_process_bridge/_events.py`.
4. In that handler, declare the upstream `RunEventKind`, the upstream event
   class, every downstream `BridgeEventType` it can emit, and the encoder logic.
5. Add the handler instance to `RUN_EVENT_BRIDGE_HANDLERS`.
6. If the TypeScript app should treat the event as known, add the event type and
   strict schema entry to `BRIDGE_EVENT_SCHEMAS` in
   `packages/process_bridge_ts/src/protocol.ts`.
7. Add a representative event object to
   `fixtures/protocol/events.json`.
8. Add or update Python encoding/fixture tests and TypeScript decoder/parity
    tests.
9. Run:

    ```bash
    uv run pytest packages/process_bridge/tests -q
    /usr/bin/make lint-ts
    /usr/bin/make test-ts
    ```

### Add Bridge-Only Event Handling

1. Add the wire event name to `BridgeEventType` in
   `src/agentlane_process_bridge/_protocol.py`.
2. Emit it from the command handler or backend operation that owns the side
   effect.
3. Add the TypeScript event shape to `packages/process_bridge_ts/src/protocol.ts`.
4. Add the matching strict schema entry to `BRIDGE_EVENT_SCHEMAS` in
   `packages/process_bridge_ts/src/protocol.ts`.
5. Add a representative event object to `fixtures/protocol/events.json`.
6. Add or update Python protocol/backend tests and TypeScript decoder/parity
   tests.

## Developer Experience Reflection

The extension path is explicit and code-native. A command has one parser, one
backend handler, and one TypeScript command shape. A run event has one encoder
handler, one downstream decoder shape, and one representative fixture.

The remaining manual work is intentional: Python dataclasses and TypeScript
types stay hand-authored because they are small and readable. The parity tests
compare parser registries, event registries, fixtures, and TypeScript schema
keys so missing command or event updates fail with concrete missing/extra names.
TypeScript process wiring reports malformed frames as `BridgeDecodeError`
values without delivering them to app reducers, so schema drift is visible
instead of silently turning into default state.
