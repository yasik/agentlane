# @agentlanejs/process-bridge

`@agentlanejs/process-bridge` is the TypeScript companion for local AgentLane
harness apps. The primary API is `createAgentSession`: it starts a Python
AgentLane backend, waits for readiness, streams typed session callbacks, and
settles run, configure, cancel, reset, and close operations.

The package is UI-framework agnostic. Apps own rendering, conversation state,
audit panels, and domain-specific reducers.

The bridge delivers complete content and structured JSON results. Apps own
previews and display limits. See
[Payload Values](../../docs/process-bridge/protocol.md#payload-values) for
result fields and serialization behavior.

The package is published to npm with the same version as the Python `agentlane`
package. Python and TypeScript must use the same native-event contract. Protocol
version `1.0` is unchanged, but the previous flat event contract is incompatible.

## Install

Add the package to the TypeScript app:

```bash
bun add @agentlanejs/process-bridge
```

For npm-based apps:

```bash
npm install @agentlanejs/process-bridge
```

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
import { createAgentSession } from "@agentlanejs/process-bridge";

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

## Runtime Configuration

Use `session.configure(patch)` for backend runtime settings that belong to the
agent instance, such as a model selection. The Python backend owns validation
through `RuntimeConfigStore`; the TypeScript side owns rendering and optional
runtime decoding through `decodeConfig`.

For the Python store shape, model-settings propagation, and the difference
between static `ready.metadata` and mutable `session.config`, see
[Process Bridge: Runtime Configuration](../../docs/process-bridge/runtime-configuration.md).

```ts
import { createAgentSession } from "@agentlanejs/process-bridge";
import { z } from "zod";

const configSchema = z.object({
  model: z.string(),
  attributes: z.record(z.string(), z.string()).default({}),
});

type ModelConfig = z.infer<typeof configSchema>;
type ModelConfigPatch = { model?: string };

const session = await createAgentSession<ModelConfig, ModelConfigPatch>({
  backend: { app: "my_app.backend:create_backend", projectDir: "." },
  decodeConfig: (raw) => configSchema.parse(raw),
  onConfigChanged: (config) => {
    modelPicker.setSelected(config.model);
    attributePanel.render(config.attributes);
  },
});

if (session.config) {
  modelPicker.setSelected(session.config.model);
  attributePanel.render(session.config.attributes);
}

await session.configure({ model: "anthropic/claude-opus-4-8" });
```

`configure()` resolves with the full applied document on success. It rejects
with `ConfigureError` when the backend reports `invalid`, `unsupported`,
`rejected`, or `internal`; if the failure includes a truth snapshot, the session
cache is updated before the promise rejects. The initial `ready.config` does
not fire `onConfigChanged`; read `session.config` after startup so app setup has
one clear initialization point. Top-level patch values must not be `undefined`;
use an explicit app-defined value when a setting needs a reset or disabled
state.

## Session API

The main public entrypoints are:

1. `createAgentSession`
2. `AgentSession`
3. `AgentSessionOptions`
4. `RunResult`
5. `RunError`
6. `ConfigureError`
7. `SessionStartError`
8. `SessionClosedError`
9. `SessionStateError`

`createAgentSession(options)` resolves after the backend emits `ready`. The
returned session handle supports one active `run()` at a time plus
`configure()`, `cancel()`, `reset()`, and idempotent `close()`.

Session callbacks provide balanced semantic events:

1. `onAssistantText` and `onReasoningText` stream text chunks with a final
   `done: true` chunk per segment.
2. `onToolActivity` emits `start`, `end`, and synthesized `cancelled` phases.
3. `onAgentActivity` emits balanced root-agent and sub-agent task phases.
4. `onPlan` emits normalized plan snapshots.
5. `onApprovalResolved` reports backend-confirmed approval outcomes.
6. `onConfigChanged` reports authoritative runtime config announcements after
   startup.
7. `onEvent` is the raw strict `BridgeEvent` tap for apps that need protocol
   details such as LLM spans, handoffs, provider events, or state snapshots.

`RunResult.finalOutput` and `AgentActivity.finalOutput` in the `end` phase have
type `unknown` because final outputs can be any JSON value. Tool end results
also preserve their JSON structure. Validate these values against your app's
schema before use.

Text callbacks complete or correct streamed text when the final output is a
string. Structured final outputs are delivered through the run result and
agent activity callbacks. Apps decide how to display them.

## Native Events

`onEvent` receives each complete source record inside `run_event.event`. Fields
and extras retain their native nesting. Use the exported `isNativeEvent` helper
to narrow known payloads without rebuilding them:

```ts
import { isNativeEvent } from "@agentlanejs/process-bridge";

function inspect(event: import("@agentlanejs/process-bridge").BridgeEvent) {
  if (event.type === "run_event" && isNativeEvent(event.event, "llm_start")) {
    console.log(event.event.payload.messages);
  }
}
```

`NativeRunEvent`, `NativePayload`, and `NativeModelEvent` describe native values.
Known records validate required fields and retain all extras. Unknown native
kinds reach `onEvent` and have no presentation effect. Unknown outer event names
and malformed known records fail decoding.

Model events stay at `event.payload.event`. Empty deltas and provider completion
records remain visible in `onEvent`, even when they produce no text callback.
Model completion and errors do not settle runs or unrelated commands. Approval
records stay at `event.payload.event.record`; policies run independently of
display callbacks. Presentation uses source lineage. Consumers can read turn
counts from state snapshots or the final run result; the bridge predicts none.

Use `llm_start` messages to inspect requests after harness transformations.
Stored prompt values contain a rendered template view that can differ from the
actual request. Full native records can include instructions, history, tool
results, prompt values, exception messages, and provider data. The local host
must authorize any forwarding or storage; the bridge does not redact fields.

Native events and complete final results use strict Python conversion.
Unsupported values, cycles, nonfinite numbers, and prompt rendering failures
produce a run error. A disconnected backend is settled through process exit.

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
bun run build
```

The root `make format`, `make lint`, `make typecheck`, and `make tests` targets
also run this package's TypeScript gates. `bun run check` is the package-level
release gate and includes linting, static analysis, tests, and the npm build.
