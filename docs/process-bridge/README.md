# Process Bridge

The process bridge is for local apps that want a TypeScript shell around a
Python AgentLane backend. The app starts a local Python process, sends prompts
and control commands, and receives typed session callbacks for text, tools,
plans, approvals, lifecycle events, and diagnostics.

Use it when:

1. the app UI or terminal shell is TypeScript
2. the agent implementation, tools, model clients, and sub-agents live in Python
3. the backend runs as a local child process owned by the app
4. the app wants high-level `run_events(...)` lifecycle data

For pure Python harness apps, use the harness docs directly. For distributed
execution, use the runtime and messaging primitives under `agentlane.runtime`
and `agentlane.messaging`.

## Pages

1. [Runtime Configuration](./runtime-configuration.md): `ready.metadata`,
   `session.config`, `configure()`, model picker state, and model-settings
   propagation.
2. [Protocol and Lifecycle](./protocol.md): commands, events, lifecycle,
   low-level TypeScript primitives, and strict decoding behavior.
3. [Development](./development.md): how to add commands, runtime config
   handling, run-event handling, and bridge-only events.

## TypeScript App API

Use `createAgentSession` from `@agentlanejs/process-bridge` as the app-facing
entrypoint:

```bash
bun add @agentlanejs/process-bridge
```

```ts
import { createAgentSession } from "@agentlanejs/process-bridge";

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
idempotent `close()`. Backends that expose runtime settings also support
`configure()`.

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
