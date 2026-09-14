# Process Bridge Stdio

This example shows the app-facing process bridge API. A TypeScript harness app
starts a Python AgentLane backend with `createAgentSession`, receives typed
session callbacks, sends one prompt, and closes the session. It does not require
`OPENAI_API_KEY`.

The Python side exposes one factory:

```python
def create_backend() -> AgentBackend:
    ...
```

The TypeScript side references that factory:

```ts
const session = await createAgentSession({
  backend: {
    app: "examples.harness.process_bridge_stdio.backend:create_backend",
    projectDir: repoRoot,
  },
});
```

Build the checkout's TypeScript package, then run the client. Start from the
repository root:

```bash
/usr/bin/make sync
bun run --cwd packages/process_bridge_ts build
bun install --cwd examples/harness/process_bridge_stdio
bun run examples/harness/process_bridge_stdio/client.ts
```

The example uses a local `file:` dependency for TypeScript and the same checkout
for Python. Rebuild after TypeScript library edits. No package publication is
needed. Both sides must use this native-event contract; the older flat events
are incompatible despite the unchanged protocol version `1.0`.

Expected output includes `ready`, `run_start`, `run_event: model_stream`,
`run_complete`, `shutdown`, and the final assistant text. Each native record is
available at `event.event` in `onEvent`; text callbacks provide the display view.
