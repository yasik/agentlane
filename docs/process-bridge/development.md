# Process Bridge Development

Use this page when extending the bridge protocol itself. Application developers
building a TypeScript shell usually need only the
[process bridge overview](./README.md) and
[runtime configuration](./runtime-configuration.md).

Command handling is registry-based.
`BridgeBackend` accepts an explicit command-handler tuple and defaults to
`BRIDGE_COMMAND_HANDLERS`; each command handler declares the command class it
handles and owns that command's side effects. Native events use
`RunEvent.to_dict()` directly. They need no separate Python encoder registry.

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

1. Define the app-owned config document shape in Python and TypeScript. Every
   announced document must be JSON-serializable. The app owns schema validation;
   the bridge sends the complete document without a size cap.
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
2. Verify that `to_dict()` preserves every source field and rejects unsupported
   values. The bridge delivers the record inside `run_event.event` automatically.
3. Add typed payload validation to `NATIVE_EVENT_SCHEMAS` in
   `packages/process_bridge_ts/src/protocol-native.ts`. Retain extra fields at
   every nested object level. Unknown native kinds already reach `onEvent`.
4. Add session presentation only when a callback needs it. Keep the raw record
   intact. Derive display fields from source data and preserve lineage.
5. Update Python-generated fixtures in
   `packages/process_bridge/fixtures/protocol/events.json` and TypeScript parity
   tests. Assert exact field retention across the language boundary.
6. Add lifecycle tests if the event affects approval or presentation state.
   Model completion and errors must not settle the run or command promises.
7. Run:

    ```bash
    uv run pytest packages/process_bridge/tests -q
    /usr/bin/make lint-ts
    /usr/bin/make test-ts
    ```

Regenerate the shared protocol fixture from native Python records before the
parity tests:

```bash
uv run python -m packages.process_bridge.tests.native_fixtures
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

A command has one parser, one backend handler, and one TypeScript command shape.
A native event uses the core serializer, a TypeScript payload schema for known
fields, and a representative fixture. Unknown native kinds remain available to
raw consumers before presentation support is added.

Parity tests compare Python records and TypeScript-decoded values. TypeScript
process wiring reports malformed frames as `BridgeDecodeError` values without
delivering them to app reducers. Update both packages and their callers together:
protocol `1.0` does not distinguish the previous flat contract from native events.

## Synthetic Payload Measurement

Run from the repository root:

```bash
uv run python -m packages.process_bridge.tests.measure_native_events
```

One local sample on 2026-09-12 used 30 repetitions per case. Delivery includes
conversion, `EventWriter`, and queue drain into a `StringIO` sink. This measures
neither OS pipe throughput nor provider latency. Frame sizes can vary slightly
with the timestamp.

| Synthetic record | Frame bytes | Median conversion (ms) | Median delivery (ms) |
| --- | ---: | ---: | ---: |
| Text delta | 391 | 0.007 | 0.143 |
| Agent end, 10 history entries | 4,224 | 0.022 | 0.166 |
| Agent end, 100 history entries | 38,694 | 0.101 | 0.437 |
| Agent end, 1,000 history entries | 384,294 | 0.874 | 2.951 |

Full history increases payload size and conversion cost. The protocol retains
all fields; apps control display limits. Measure representative app workloads
before choosing downstream storage and forwarding policies.
