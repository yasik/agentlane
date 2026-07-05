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

Run the client from the repository root:

```bash
cd examples/harness/process_bridge_stdio
bun install
bun run client.ts
```

Expected output includes lifecycle events such as `ready`, `run_start`,
`assistant_delta`, `run_complete`, `shutdown`, and the final assistant text.
