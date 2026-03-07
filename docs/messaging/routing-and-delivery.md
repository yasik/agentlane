# Messaging: Routing and Delivery

This document describes AgentLane messaging primitives and how `send_message` / `publish_message` are routed and delivered.

## TL;DR

1. Use `send_message` for one recipient and terminal `DeliveryOutcome`.
2. Use `publish_message` for fan-out and enqueue-only `PublishAck`.
3. Subscriptions map topic matches to recipients by `route_key` (`topic.source`).
4. `DeliveryMode.STATEFUL` reuses recipient instances; `STATELESS` creates per-delivery recipients.

## Core Primitives

### Identity

1. `AgentId = (AgentType, AgentKey)` identifies one runtime target.
2. `TopicId = (type, source)` where `source` is the route-key dimension.
3. `CorrelationId` ties one logical workflow chain across hops.
4. `MessageId` uniquely identifies one envelope.

### Envelope

`MessageEnvelope` carries:

1. message kind (`RPC_REQUEST`, `PUBLISH_EVENT`, ...),
2. sender and recipient/topic,
3. payload metadata (`schema_name`, `content_type`, `format`),
4. optional `idempotency_key`.

## Direct Send Semantics

`send_message(...)` does:

1. recipient resolution (`AgentId` directly, or type-only resolution),
2. single RPC envelope creation,
3. one task enqueue,
4. wait for terminal `DeliveryOutcome`.

If recipient cannot be resolved, runtime returns `POLICY_REJECTED` or `UNDELIVERABLE` outcome (depending on failure stage), not a partial success.

## Publish Semantics

`publish_message(...)` does:

1. build one base publish envelope,
2. resolve all matching subscriptions into `PublishRoute` values,
3. enqueue one task per route,
4. return `PublishAck` once enqueued.

Important:
`PublishAck` confirms enqueue only, not downstream handler completion.

## Subscriptions and Topic Matching

Use runtime convenience APIs:

1. `subscribe_exact(topic_type=..., agent_type=..., delivery_mode=...)`
2. `subscribe_prefix(topic_prefix=..., agent_type=..., delivery_mode=...)`

Matching strategies:

1. `TYPE_EXACT`: `topic.type == topic_pattern`
2. `TYPE_PREFIX`: `topic.type.startswith(topic_pattern)`

## Delivery Modes

### Stateful

`DeliveryMode.STATEFUL` maps recipient key from topic route key:

1. recipient key = `topic.source`,
2. same `(agent_type, route_key)` reuses same cached instance,
3. preserves state across deliveries for that key.

### Stateless

`DeliveryMode.STATELESS` uses a unique per-delivery recipient key:

1. no instance reuse guarantee,
2. parallel-friendly for workflow fan-out,
3. still keeps deterministic routing per subscription.

## Ordering Guarantees

1. Per `AgentId` delivery order is FIFO.
2. Different `AgentId` values can run concurrently (subject to worker count).
3. Publish fan-out order is deterministic after routing dedup/sort.

## Correlation and Idempotency

1. Preserve `correlation_id` when forwarding work to keep one traceable chain.
2. Use `idempotency_key` for retry-safe dedup semantics at transport/runtime boundaries.

## Minimal Example

```python
from agentlane.messaging import AgentId, DeliveryMode, TopicId
from agentlane.runtime import SingleThreadedRuntimeEngine


runtime = SingleThreadedRuntimeEngine()
runtime.subscribe_exact(
    topic_type="workflow.plan_ready",
    agent_type="worker",
    delivery_mode=DeliveryMode.STATEFUL,
)

ack = await runtime.publish_message(
    {"plan_id": "p-1"},
    topic=TopicId.from_values(
        type_value="workflow.plan_ready",
        route_key="session-42",
    ),
)
assert ack.enqueued_recipient_count >= 0

result = await runtime.send_message(
    {"task_id": "t-1"},
    recipient=AgentId.from_values("worker", "session-42"),
)
assert result.status.value in {
    "delivered",
    "handler_error",
    "policy_rejected",
    "undeliverable",
}
```
