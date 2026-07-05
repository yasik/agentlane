# @agentlane/process-bridge

`@agentlane/process-bridge` is the TypeScript companion for local AgentLane
harness apps. The primary API is `createAgentSession`: it starts a Python
AgentLane backend, waits for readiness, streams typed session callbacks, and
settles run, cancel, reset, and close operations.

The package is UI-framework agnostic. Apps own rendering, conversation state,
audit panels, and domain-specific reducers.

The package is currently private to the repository while npm publication policy
is decided.

## Quickstart

Create a Python backend factory:

```python
from agentlane_process_bridge import AgentBackend
from my_app.agents import build_agent

def create_backend() -> AgentBackend:
    return AgentBackend(agent=build_agent())
```

Start it from TypeScript:

```ts
import { createAgentSession } from "@agentlane/process-bridge";

const session = await createAgentSession({
  backend: { app: "my_app.backend:create_backend", projectDir: "." },
  onAssistantText: ({ delta }) => process.stdout.write(delta),
  onToolActivity: (activity) => app.tools.apply(activity),
  onPlan: (plan) => app.plan.replace(plan),
});

await session.run("Summarize this workspace.");
await session.close();
```

For interactive approvals, provide one policy function:

```ts
const session = await createAgentSession({
  backend: { app: "my_app.backend:create_backend", projectDir: "." },
  approvals: async ({ request, signal }) => {
    return ui.confirmToolUse(request, { signal });
  },
});
```

The Python factory owns agent construction, tools, model settings, sub-agents,
and approval broker wiring. The TypeScript app sends prompts and control
commands, then consumes session callbacks and run results.

## Session API

The main public entrypoints are:

1. `createAgentSession`
2. `AgentSession`
3. `AgentSessionOptions`
4. `RunResult`
5. `RunError`
6. `SessionStartError`
7. `SessionClosedError`
8. `SessionStateError`

`createAgentSession(options)` resolves after the backend emits `ready`. The
returned session handle supports one active `run()` at a time plus `cancel()`,
`reset()`, and idempotent `close()`.

Session callbacks provide balanced semantic events:

1. `onAssistantText` and `onReasoningText` stream text chunks with a final
   `done: true` chunk per segment.
2. `onToolActivity` emits `start`, `end`, and synthesized `cancelled` phases.
3. `onAgentActivity` emits balanced root-agent and sub-agent task phases.
4. `onPlan` emits normalized plan snapshots.
5. `onApprovalResolved` reports backend-confirmed approval outcomes.
6. `onEvent` is the raw strict `BridgeEvent` tap for apps that need protocol
   details such as LLM spans, handoffs, provider events, or state snapshots.

## Low-Level Building Blocks

The session API is built on the exported protocol and process primitives. Use
these when writing tests, custom launchers, or bridge infrastructure:

1. `encodeBridgeCommand`
2. `decodeBridgeEventLine`
3. `spawnBridgeProcess`
4. `wireBridgeProcess`
5. `createBridgeChannel`
6. `BridgeEvent`
7. `BridgeDecodeError`

Low-level consumers are responsible for ready gating, command correlation,
approval decisions, text coalescing, operation settlement, and lifecycle
cleanup.

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
