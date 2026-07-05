# Harness Process Bridge

The process bridge is for local apps that want a TypeScript harness around a
Python AgentLane backend. The app starts a local Python process, sends prompts
and control commands, and receives typed session callbacks for text, tools,
plans, approvals, lifecycle events, and diagnostics.

Use it when:

1. the app UI or terminal shell is TypeScript
2. the agent implementation, tools, model clients, and sub-agents live in Python
3. the backend runs as a local child process owned by the app
4. the app wants high-level `run_events(...)` lifecycle data

For distributed execution, keep using the runtime and messaging primitives
under `agentlane.runtime` and `agentlane.messaging`.

## TypeScript App API

Use `createAgentSession` from `@agentlane/process-bridge` as the app-facing
entrypoint:

```ts
import { createAgentSession } from "@agentlane/process-bridge";

const session = await createAgentSession({
  backend: { app: "my_app.backend:create_backend", projectDir: "." },
  onAssistantText: ({ delta }) => process.stdout.write(delta),
  onToolActivity: (activity) => app.tools.apply(activity),
  onPlan: (plan) => app.plan.replace(plan),
});

await session.run("Summarize this case.");
await session.close();
```

`createAgentSession` resolves after the backend emits `ready`. The returned
handle supports one active `run()` at a time plus `cancel()`, `reset()`, and
idempotent `close()`.

Session callbacks are balanced by the package:

1. text chunks receive exactly one final `done: true` chunk per segment
2. tool calls receive `start`, `end`, or synthesized `cancelled`
3. agent and sub-agent tasks receive `start`, `end`, or synthesized `cancelled`
4. approval policies are called once per request and receive an abort signal
5. operation promises settle on completion, cancellation, backend exit, send
   failure, or protocol failure

Apps that need raw protocol details can subscribe to `onEvent`. That callback
receives the strict `BridgeEvent` union before semantic processing.

## Python Backend Factory

The Python side owns agent construction, model settings, tools, sub-agents, and
approval broker wiring. Expose one factory that returns `AgentBackend`:

```python
from agentlane.harness.tools import ToolApprovalBroker
from agentlane_process_bridge import AgentBackend
from my_app.agents import build_agent

def create_backend() -> AgentBackend:
    broker = ToolApprovalBroker()
    return AgentBackend(
        agent=build_agent(approval_callback=broker.callback),
        approvals=broker,
    )
```

The TypeScript backend spec:

```ts
{ app: "my_app.backend:create_backend", projectDir: "." }
```

launches:

```bash
uv run --project . python -m agentlane_process_bridge --app my_app.backend:create_backend
```

Approval-gated agents must share one `ToolApprovalBroker` between the agent
tool callbacks and `AgentBackend.approvals`. Agents that do not gate tools can
return `AgentBackend(agent=agent)` or the bare `AgentRuntime`.

See
[`examples/harness/process_bridge_stdio`](../../examples/harness/process_bridge_stdio/)
for a runnable no-model-key smoke example.

## Protocol

The session API uses a strict stdio protocol internally. Every command and
event is one JSON object followed by `\n`.

Stdout is reserved for protocol events. Python logging and diagnostics must go
to stderr; `run_stdio(...)` configures that before emitting the ready event.

Every protocol object carries:

1. `protocol_version`
2. `type`

Events also carry `ts`, a Unix timestamp rounded to milliseconds. Event fields
stay flat for app consumption.

Commands:

1. `prompt` with `text`
2. `approve` with `id`, strict boolean `allowed`, and optional `reason`
3. `cancel`
4. `reset`
5. `shutdown`

Only JSON boolean `true` grants an approval. Values such as `"true"`, `1`, or
`{}` deny.

## Lifecycle

`BridgeBackend` owns one active run at a time. If a prompt arrives while a run
is active, the backend emits a command-scoped `error` and leaves the current run
untouched.

Cancel, reset, shutdown, run failure, EOF, and backend close all clear pending
approvals and close the active `RunEventStream` with the documented AgentLane
pattern:

```python
await stream.aclose()
with contextlib.suppress(asyncio.CancelledError):
    await stream.result()
```

That stops provider-side work and retrieves the result future so asyncio does
not report an unobserved cancellation.

## Event Model

The bridge serializes AgentLane's existing `RunEvent` vocabulary:

1. model text, reasoning, tool-argument, provider, and error events
2. agent, LLM, tool, and handoff lifecycle events
3. plan updates
4. tool approval requests and resolutions
5. state snapshots
6. run start, completion, cancellation, reset, cancel, shutdown, and errors

`BridgeEventType` is the bridge wire vocabulary. Bridge-only lifecycle values
live in the process-bridge package; values that correspond to AgentLane run
events derive from upstream `RunEventKind` or `HarnessEventType` values so
bridges and apps do not duplicate framework event literals.

Command handling and run-event encoding use explicit handler registries. When
AgentLane adds a new `RunEventKind`, the bridge extension path is one
`RunEventBridgeHandler` implementation that declares the upstream kind, event
class, downstream `BridgeEventType` values, and encoder logic. When the bridge
adds a command, the extension path is one `BridgeCommandHandler`
implementation that declares the command class and owns the side effects.
`BridgeBackend` and `RunEventEncoder` accept explicit handler tuples and default
to `BRIDGE_COMMAND_HANDLERS` and `RUN_EVENT_BRIDGE_HANDLERS`; fixture parity
tests fail until new upstream run-event kinds are covered.

Lineage fields such as `task_id`, `parent_task_id`, `is_root`, and
`is_subagent` are preserved on task-carrying events. `tool_end` events include
the framework-derived `ok` flag and typed `error` payload instead of asking the
app to infer success from result text.

Unhandled AgentLane run-event classes encode as the explicit `run_event`
diagnostic event. Unknown bridge protocol event names and invalid payloads fail
strict TypeScript decoding and are reported as `BridgeDecodeError` values
instead of being delivered to app reducers.

## Low-Level TypeScript Primitives

The TypeScript package also exports protocol, process, and channel helpers:

```ts
import {
  createBridgeChannel,
  spawnBridgeProcess,
} from "@agentlane/process-bridge";

const child = spawnBridgeProcess(
  {
    command: "uv",
    args: [
      "run",
      "python",
      "-m",
      "agentlane_process_bridge",
      "--app",
      "my_app.backend:create_backend",
    ],
  },
  {
    onEvent: (event) => {
      console.log(event.type);
    },
    onStderr: (line) => console.error(line),
  },
);

const channel = createBridgeChannel(child);
channel.send({ type: "prompt", text: "Summarize this case." });
```

Use the low-level helpers for tests, custom launchers, or bridge
infrastructure. Consumers that use them directly own ready gating, command
correlation, text buffering, approval resolution, operation settlement, and
lifecycle cleanup.
