# Runtime: Engine and Execution

This document explains how AgentLane runtime executes message deliveries, including ordering, concurrency, lifecycle, and extension points.

## TL;DR

1. `RuntimeEngine` is the orchestration entrypoint (`send_message`, `publish_message`).
2. `SingleThreadedRuntimeEngine` is the current production path in v1.
3. Scheduling guarantees in-order execution per `AgentId`.
4. Concurrency happens across different `AgentId` values, bounded by worker count.

## Runtime Components

A delivery flows through:

1. `RuntimeEngine` (API + envelope construction),
2. `RoutingEngine` (publish route resolution),
3. `PerAgentMailboxScheduler` (queueing and fairness),
4. `Dispatcher` (instance lookup + handler invocation),
5. `AgentRegistry` (factory/instance lifecycle).

## Agent Lifecycle

### Registration Modes

1. `register_factory(agent_type, factory)` for lazy creation.
2. `register_instance(agent_id, instance)` for explicit pre-created stateful instance.

Factory contract:

1. Engine is always passed in: `factory(engine)`.
2. Factory may be sync or async.
3. Created instances are bound to their `AgentId` by runtime.

### Reuse Behavior

1. Same `AgentId` resolves to same cached instance.
2. Different `AgentId` values produce isolated instances.
3. Publish `STATELESS` mode uses unique transient recipient keys per delivery.

## Scheduling and Concurrency

Default scheduler behavior:

1. one FIFO mailbox per recipient `AgentId`,
2. fair round-robin across non-empty mailboxes,
3. at most one active task per recipient at a time.

Practical implication:

1. two parallel sends to same `AgentId` are serialized,
2. two sends to different `AgentId` values can run in parallel,
3. increase `worker_count` to raise cross-recipient concurrency.

## Delivery APIs

### `send_message`

1. builds one RPC envelope,
2. enqueues one task,
3. waits for terminal `DeliveryOutcome`.

Use this for request/response paths where caller needs completion status and optional response payload.

### `publish_message`

1. builds one base event envelope,
2. resolves matching subscriptions,
3. enqueues one task per resolved route,
4. returns `PublishAck` after enqueue.

Use this for event fan-out where caller only needs enqueue confirmation.

## Lifecycle Management

`RuntimeEngine` lifecycle methods are idempotent:

1. `start()`
2. `stop()` (immediate shutdown, cancels in-flight/queued work)
3. `stop_when_idle()` (graceful drain)

Async context helpers:

1. `single_threaded_runtime(...)`
2. `distributed_runtime(...)`
3. `runtime_scope(...)`

These helpers start runtime on entry and stop/stop-when-idle on exit depending on success/failure path.

## Distributed Mode Status

`DistributedRuntimeEngine` exists as a contract placeholder in v1 and currently rejects `_submit(...)` with `NotImplementedError`.

This keeps API compatibility while transport + placement implementations are introduced incrementally.

## Minimal Example

```python
from dataclasses import dataclass

from agentlane.messaging import AgentId, MessageContext
from agentlane.runtime import (
    BaseAgent,
    SingleThreadedRuntimeEngine,
    on_message,
)


@dataclass(slots=True)
class Ping:
    value: str


class WorkerAgent(BaseAgent):
    @on_message
    async def handle(self, payload: Ping, context: MessageContext) -> object:
        _ = context
        return {"reply": payload.value.upper()}


runtime = SingleThreadedRuntimeEngine(worker_count=4)
runtime.register_factory("worker", WorkerAgent)

result = await runtime.send_message(
    Ping(value="hello"),
    recipient=AgentId.from_values("worker", "session-1"),
)
assert result.status.value == "delivered"
assert result.response_payload == {"reply": "HELLO"}
```
