# @agentlane/process-bridge

`@agentlane/process-bridge` is the TypeScript companion for AgentLane's local
process bridge. It provides command encoding, event decoding, child-process
wiring, and shutdown helpers for apps that host a Python AgentLane backend.

The package is intentionally UI-framework agnostic. Apps own reducers,
rendering, audit panels, and domain state.

The main public entrypoints are:

1. `encodeBridgeCommand`
2. `decodeBridgeEventLine`
3. `spawnBridgeProcess`
4. `wireBridgeProcess`
5. `createBridgeChannel`

The package is currently private to the repository while npm publication
policy is decided.

## Development

Install dependencies from the repository root:

```bash
/usr/bin/make sync
```

Run the TypeScript package checks directly:

```bash
bun run format
bun run lint
bun run typecheck
bun run test
```

The root `make format`, `make lint`, `make typecheck`, and `make tests` targets
also run this package's TypeScript gates.
