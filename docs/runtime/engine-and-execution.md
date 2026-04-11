# Runtime: Engine and Execution

The runtime is where AgentLane decides what delivery means. It is responsible
for creating handlers, preserving ordering, and defining what a caller can
expect after sending or publishing a message.

Most applications meet that behavior through
[`SingleThreadedRuntimeEngine`](../../src/agentlane/runtime/_runtime.py). The
distributed path keeps the same public contract through
[`DistributedRuntimeEngine`](../../src/agentlane/runtime/_worker_runtime.py).
Both build on
[`RuntimeEngine`](../../src/agentlane/runtime/_runtime.py) and rely on
[`AgentRegistry`](../../src/agentlane/runtime/_registry.py) plus
[`PerAgentMailboxScheduler`](../../src/agentlane/runtime/_scheduler.py) to
reuse instances and preserve fair FIFO execution.

## The Delivery Model

There are two caller-facing paths:

1. `send_message(...)` sends work to one recipient and waits for a terminal
   [`DeliveryOutcome`](../../src/agentlane/messaging/_outcome.py)
2. `publish_message(...)` fans one event out to all matching subscriptions and
   returns a [`PublishAck`](../../src/agentlane/messaging/_outcome.py) once the
   deliveries are enqueued

That difference shapes most other runtime behavior. Direct sends are about
completion. Publish is about fan-out and enqueue.

## Identity And Instance Reuse

The runtime addresses work by
[`AgentId`](../../src/agentlane/messaging/_identity.py). If the same `AgentId`
is used again, the runtime reuses the same cached agent instance. If a
different `AgentId` is used, the runtime creates or resolves a different
instance.

That is why statefulness is explicit rather than hidden:

1. reuse the same `AgentId` when you want one long-lived instance
2. use different `AgentId` values when you want isolated work

Registration follows the same idea. `register_factory(...)` is the normal path
when the runtime should create instances lazily. `register_instance(...)` is
useful when you want to bind an already-created stateful instance to one
identity.

## Ordering And Concurrency

The runtime guarantees FIFO delivery per recipient. Two deliveries for the same
`AgentId` run one after the other. Deliveries for different `AgentId` values
may run concurrently.

In practice, that means:

1. one mailbox per recipient
2. one active handler per recipient at a time
3. fair scheduling across different recipients that have queued work

This is why concurrency in AgentLane is easier to reason about than a general
task pool. Parallelism comes from using different recipients, not from
re-entering the same one.

## Lifecycle And Shutdown

[`RuntimeEngine`](../../src/agentlane/runtime/_runtime.py) has three lifecycle
operations:

1. `start()` to make the runtime ready for new work
2. `stop()` to cancel in-flight and queued work immediately
3. `stop_when_idle()` to stop after queued work has drained

Most application code uses the async context helpers
`single_threaded_runtime(...)`, `distributed_runtime(...)`, or
`runtime_scope(...)` instead of calling lifecycle methods directly.

## Where Distributed Mode Fits

Distributed mode keeps the same public runtime API while changing where work is
executed. The convenience entrypoint still behaves like one runtime from the
caller's point of view, but behind the scenes a host chooses a worker and that
worker runs the delivery using the same local scheduler and registry model.

Read [Runtime: Distributed Runtime Usage](./distributed-runtime-usage.md) when
you need the practical setup. Read
[Runtime: Distributed Runtime Architecture](./distributed-runtime-architecture.md)
when you want the host and worker responsibilities in more detail.

## Example

```python
from dataclasses import dataclass

from agentlane.messaging import AgentId, MessageContext
from agentlane.runtime import BaseAgent, SingleThreadedRuntimeEngine, on_message


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
