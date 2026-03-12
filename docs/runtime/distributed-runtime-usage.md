# Runtime: Distributed Runtime Usage

This guide shows how to use the distributed runtime that is implemented in core today.

## When To Use Which Entry Point

Use `distributed_runtime()` when you want the same runtime API as the
single-threaded engine, but backed by an in-process host plus one primary worker
with no network setup.

Use `WorkerAgentRuntimeHost` and `WorkerAgentRuntime` directly when you want to
model an explicit multi-worker topology yourself.

## Zero-Config Example

`distributed_runtime()` manages the host lifecycle for you and yields a
`DistributedRuntimeEngine`.

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

What this does:

1. Starts an in-process `WorkerAgentRuntimeHost`.
2. Starts one primary `WorkerAgentRuntime`.
3. Runs your registered factories on that worker.
4. Stops the worker and host automatically when the context exits.

## Explicit Direct RPC Across Workers

Use the explicit primitives when you want to place different agent types on
different workers and route direct RPCs through the host.

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
    assert outcome.response_payload == {
        "agent_id": "session-1",
        "echo": "ping",
    }
finally:
    await caller.stop_when_idle()
    await echo_worker.stop_when_idle()
    await host.stop_when_idle()
```

The important point is that `caller` does not host the `echo` factory. The
host resolves `AgentType("echo")` to `echo_worker`, then that worker executes
the agent locally.

## Explicit Publish Across Workers

Use the explicit primitives when you want publish fan-out resolved centrally by
the host and executed on whichever worker owns the subscribed agent type.

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

## Lifecycle Notes

Host lifecycle:

1. Start the host before starting explicit workers.
2. `host.stop()` fails in-flight direct RPC sessions immediately.
3. `host.stop_when_idle()` currently waits for direct RPC sessions only.

Worker lifecycle:

1. A worker must know its `host_address` before `start()`.
2. Worker startup binds the local gRPC endpoint first, then registers with the host.
3. Worker shutdown deregisters from the host before transport teardown so new
   cross-worker traffic stops first.
4. Registering factories and subscriptions before `start()` is fine. The worker
   pushes a full catalog snapshot during startup and after later local changes.

## Routing Semantics

Current v1 placement and routing rules:

1. One worker owns a given `AgentType`.
2. All distributed `send_message` and `publish_message` calls go through the host.
3. The host keeps the authoritative pending-session table for direct RPCs.
4. The host mirrors worker subscriptions and computes concrete publish deliveries.
5. The target worker directly enqueues the concrete deliveries it receives from
   the host; it does not rerun global publish subscription matching.

## Current Limitations

These are current implementation facts, not future promises:

1. `distributed_runtime()` manages one in-process host plus one primary worker.
   Extra workers must be created explicitly with `WorkerAgentRuntime`.
2. Placement is exclusive per `AgentType`, so different keys of the same type are
   not sharded across multiple workers yet.
3. `cancellation_token` is part of the API surface but is not propagated across
   the process boundary yet.
4. `PublishAck` means enqueue only, never downstream handler completion.

## Runnable Examples

See these runnable distributed runtime demos:

1. [distributed_publish_fan_in](../../examples/runtime/distributed_publish_fan_in/README.md):
   explicit host/workers with publish fan-out and one stateful aggregator agent
   for fan-in.
2. [distributed_scatter_gather](../../examples/runtime/distributed_scatter_gather/README.md):
   explicit host/workers with one coordinator agent aggregating multiple direct
   RPC responses.
3. [simple](../../examples/runtime/simple/README.md):
   stripped-down starter versions of both patterns with plain `print(...)`
   output and single-file scripts.

For design background and tradeoffs, see
[Runtime: Distributed Host/Worker Architecture](./distributed-runtime-architecture.md).
