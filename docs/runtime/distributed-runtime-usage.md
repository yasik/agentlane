# Runtime: Distributed Runtime Usage

Distributed runtime support is easiest to approach in two modes:

1. use the managed entrypoint when you want the normal runtime API and one
   primary worker behind it
2. use explicit hosts and workers when you need to control topology yourself

Most applications should start with
[`distributed_runtime(...)`](../../src/agentlane/runtime/_context.py), which
yields a managed
[`DistributedRuntimeEngine`](../../src/agentlane/runtime/_worker_runtime.py).
Reach for
[`WorkerAgentRuntime`](../../src/agentlane/runtime/_worker_runtime.py) and
[`WorkerAgentRuntimeHost`](../../src/agentlane/runtime/_worker_runtime_host.py)
directly only when worker placement or topology needs to be explicit.

## Start With The Managed Entry Point

Use [`distributed_runtime()`](../../src/agentlane/runtime/_context.py) when you
want to keep the runtime mental model unchanged. You still call
`send_message(...)`, `publish_message(...)`, `register_factory(...)`, and the
other normal runtime APIs. The difference is that a host and a primary worker
are created for you behind the scenes.

```python
from agentlane.messaging import AgentId, MessageContext
from agentlane.runtime import BaseAgent, Engine, distributed_runtime, on_message


class CounterAgent(BaseAgent):
    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)
        self._count = 0

    @on_message
    async def handle(self, payload: str, context: MessageContext) -> object:
        _ = payload
        _ = context
        self._count += 1
        return {"count": self._count}


async with distributed_runtime() as runtime:
    runtime.register_factory("counter", CounterAgent)

    first = await runtime.send_message(
        "one",
        recipient=AgentId.from_values("counter", "session-1"),
    )
    second = await runtime.send_message(
        "two",
        recipient=AgentId.from_values("counter", "session-1"),
    )

    assert first.status.value == "delivered"
    assert second.response_payload == {"count": 2}
```

This is the right default when:

1. you want distributed execution without designing your own topology
2. one host plus one primary worker is enough
3. you want the same mental model as the in-process runtime

## Move To Explicit Hosts And Workers When Placement Matters

Use explicit
[`WorkerAgentRuntimeHost`](../../src/agentlane/runtime/_worker_runtime_host.py)
and [`WorkerAgentRuntime`](../../src/agentlane/runtime/_worker_runtime.py)
instances when you want different agent types to live on different workers.

The important design idea is that the host chooses the destination worker, but
the chosen worker still runs the delivery locally with its own registry,
scheduler, and agent instances.

### Direct Send Across Workers

```python
from agentlane.messaging import AgentId, MessageContext
from agentlane.runtime import (
    BaseAgent,
    WorkerAgentRuntime,
    WorkerAgentRuntimeHost,
    on_message,
)


class EchoAgent(BaseAgent):
    @on_message
    async def handle(self, payload: str, context: MessageContext) -> object:
        _ = context
        return {"agent_id": self.id.key.value, "echo": payload}


host = WorkerAgentRuntimeHost(address="127.0.0.1:0")
await host.start()

caller = WorkerAgentRuntime(host_address=host.address, address="127.0.0.1:0")
echo_worker = WorkerAgentRuntime(host_address=host.address, address="127.0.0.1:0")

echo_worker.register_factory("echo", EchoAgent)

await caller.start()
await echo_worker.start()

try:
    outcome = await caller.send_message(
        "ping",
        recipient=AgentId.from_values("echo", "session-1"),
    )
    assert outcome.status.value == "delivered"
finally:
    await caller.stop_when_idle()
    await echo_worker.stop_when_idle()
    await host.stop_when_idle()
```

The caller does not need to host the target agent type. The host resolves the
destination worker and forwards the delivery there.

### Publish Across Workers

Publish follows the same high-level pattern, but the host resolves all matching
subscriptions before it forwards concrete deliveries to workers.

```python
from agentlane.messaging import MessageContext, TopicId
from agentlane.runtime import (
    BaseAgent,
    WorkerAgentRuntime,
    WorkerAgentRuntimeHost,
    on_message,
)


class ListenerAgent(BaseAgent):
    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        _ = context
        print(f"listener received: {payload}")
        return None


host = WorkerAgentRuntimeHost(address="127.0.0.1:0")
await host.start()

publisher = WorkerAgentRuntime(host_address=host.address, address="127.0.0.1:0")
listener = WorkerAgentRuntime(host_address=host.address, address="127.0.0.1:0")

listener.register_factory("listener", ListenerAgent)
listener.subscribe_exact(topic_type="alerts", agent_type="listener")

await publisher.start()
await listener.start()

try:
    ack = await publisher.publish_message(
        {"event": "ready"},
        topic=TopicId.from_values(type_value="alerts", route_key="session-1"),
    )
    assert ack.enqueued_recipient_count == 1
finally:
    await publisher.stop_when_idle()
    await listener.stop_when_idle()
    await host.stop_when_idle()
```

## What To Keep In Mind

The current distributed implementation is intentionally narrow:

1. `distributed_runtime()` manages one in-process host plus one primary worker
2. a given `AgentType` is owned by one worker at a time
3. `PublishAck` still means enqueue only, not handler completion
4. cancellation does not yet propagate across the process boundary

Those are current implementation facts, not just usage advice.

## Related Docs

1. [Runtime: Engine and Execution](./engine-and-execution.md)
2. [Runtime: Distributed Runtime Architecture](./distributed-runtime-architecture.md)
