# Process Bridge Development

Use this page when extending the bridge protocol itself. Application developers
building a TypeScript shell usually need only the
[process bridge overview](./README.md) and
[runtime configuration](./runtime-configuration.md).

Command handling and run-event encoding are registry-based.
`BridgeBackend` accepts an explicit command-handler tuple and defaults to
`BRIDGE_COMMAND_HANDLERS`; each command handler declares the command class it
handles and owns that command's side effects. `RunEventEncoder` accepts an
explicit run-event-handler tuple and defaults to `RUN_EVENT_BRIDGE_HANDLERS`;
each run-event handler declares the upstream `RunEventKind`, the event class,
the emitted `BridgeEventType` values, and its encoder implementation.

## Add a Command

1. Add the command name to `CommandType` in
   `packages/process_bridge/src/agentlane_process_bridge/_protocol.py`.
2. Add a `COMMAND_*` constant for the new command name in
   `packages/process_bridge/src/agentlane_process_bridge/_protocol.py`.
3. Add the command constant to `COMMAND_TYPES`.
4. Add a frozen command dataclass with a literal `type` field in
   `packages/process_bridge/src/agentlane_process_bridge/_protocol.py`.
5. Add the dataclass to the `BridgeCommand` union.
6. Add one `BridgeCommandParser` implementation in
   `packages/process_bridge/src/agentlane_process_bridge/_protocol.py`.
7. Add the parser instance to `COMMAND_PARSERS`.
8. Add one `BridgeCommandHandler` implementation in
   `packages/process_bridge/src/agentlane_process_bridge/_backend.py`.
9. Add the handler instance to `BRIDGE_COMMAND_HANDLERS`.
10. Re-export the command dataclass and any public handler type from
    `packages/process_bridge/src/agentlane_process_bridge/__init__.py` when
    apps should import it.
11. Add the matching command name to `KNOWN_COMMAND_TYPES` and the command shape
    to `BridgeCommand` in `packages/process_bridge_ts/src/protocol.ts`.
12. Add or update Python protocol/backend tests and TypeScript protocol/channel
    tests.
13. Run:

    ```bash
    uv run pytest packages/process_bridge/tests -q
    /usr/bin/make lint-ts
    /usr/bin/make test-ts
    ```

## Add Runtime Config Handling

1. Define the app-owned config document shape in Python and TypeScript. Keep it
   small; every announced document must be JSON-serializable and fit under the
   bridge contract payload cap.
2. Validate raw patches into a named app patch type at the Python boundary. Do
   not thread generic dict lookups through the store's application logic.
3. Implement `RuntimeConfigStore.snapshot()` to return the full current
   document.
4. Implement `RuntimeConfigStore.apply(patch)` to validate the whole patch
   before mutating state, then return the full applied document.
5. Raise `ConfigRejectedError` for user-fixable problems such as unknown model
   ids, attributes, or options. Let unexpected exceptions raise normally; the
   bridge reports them as internal failures with a fresh snapshot.
6. Pass the store as `AgentBackend(config=store)` or `run_stdio(config=store)`.
7. In TypeScript, call `createAgentSession<TConfig, TConfigPatch>({
   decodeConfig, ... })` when the patch shape differs from backend truth, and
   render from `session.config` plus `onConfigChanged`.
8. Apply changes through `await session.configure(patch)`. Do not predict local
   state; the resolved document and callback are backend truth.
9. Add Python store/handler tests and TypeScript session tests for success,
   rejection, reset re-announcement, and bad config decoding.

## Add Run-Event Handling

1. Add the upstream run event to `agentlane.harness.RunEventKind` and the
   concrete run-event dataclass in AgentLane core first.
2. Add any new typed wire event to `BridgeEventType` in
   `packages/process_bridge/src/agentlane_process_bridge/_protocol.py`. Derive
   the value from `RunEventKind` or `HarnessEventType`; do not repeat event
   string literals downstream.
3. Add one `RunEventBridgeHandler` implementation in
   `packages/process_bridge/src/agentlane_process_bridge/_events.py`.
4. In that handler, declare the upstream `RunEventKind`, the upstream event
   class, every downstream `BridgeEventType` it can emit, and the encoder logic.
5. Add the handler instance to `RUN_EVENT_BRIDGE_HANDLERS`.
6. If the TypeScript app should treat the event as known, add the event type and
   strict schema entry to `BRIDGE_EVENT_SCHEMAS` in
   `packages/process_bridge_ts/src/protocol.ts`.
7. Add a representative event object to
   `packages/process_bridge/fixtures/protocol/events.json`.
8. Add or update Python encoding/fixture tests and TypeScript decoder/parity
   tests.
9. Run:

    ```bash
    uv run pytest packages/process_bridge/tests -q
    /usr/bin/make lint-ts
    /usr/bin/make test-ts
    ```

## Add Bridge-Only Event Handling

1. Add the wire event name to `BridgeEventType` in
   `packages/process_bridge/src/agentlane_process_bridge/_protocol.py`.
2. Emit it from the command handler or backend operation that owns the side
   effect.
3. Add the TypeScript event shape to `packages/process_bridge_ts/src/protocol.ts`.
4. Add the matching strict schema entry to `BRIDGE_EVENT_SCHEMAS` in
   `packages/process_bridge_ts/src/protocol.ts`.
5. Add a representative event object to
   `packages/process_bridge/fixtures/protocol/events.json`.
6. Add or update Python protocol/backend tests and TypeScript decoder/parity
   tests.

## Developer Experience

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
