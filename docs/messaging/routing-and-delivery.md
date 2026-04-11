# Messaging: Routing and Delivery

Messaging is the vocabulary the rest of AgentLane builds on. Before there are
tools, handoffs, or runs, there are only recipients, topics, envelopes, and
delivery outcomes.

That vocabulary is represented directly in code:
[`AgentId`](../../src/agentlane/messaging/_identity.py) names one recipient,
[`TopicId`](../../src/agentlane/messaging/_identity.py) names a publish target,
[`MessageEnvelope`](../../src/agentlane/messaging/_envelope.py) carries the
work, and [`DeliveryOutcome`](../../src/agentlane/messaging/_outcome.py) or
[`PublishAck`](../../src/agentlane/messaging/_outcome.py) tell the caller what
happened. Subscription matching is governed by
[`DeliveryMode`](../../src/agentlane/messaging/_subscription.py).

## Two Delivery Patterns

There are two caller-facing ways to move work:

1. `send_message(...)` for one recipient and a terminal outcome
2. `publish_message(...)` for topic-based fan-out and enqueue confirmation

Use send when the caller needs completion. Use publish when the caller needs to
announce an event and let matching subscribers process it independently.

## Identities And Topics

An [`AgentId`](../../src/agentlane/messaging/_identity.py) points at one
runtime recipient. It is made from an `AgentType` and an `AgentKey`.

A [`TopicId`](../../src/agentlane/messaging/_identity.py) describes a publish
target instead. It has a topic type and a route key. The route key is what
later lets publish deliveries preserve stateful affinity for the same logical
stream of work.

Once a delivery is created, it travels as a
[`MessageEnvelope`](../../src/agentlane/messaging/_envelope.py). The envelope
holds the sender, recipient or topic, payload metadata, and correlation data
that must survive transport.

## Direct Send

`send_message(...)` resolves one recipient, enqueues one delivery, and waits for
one terminal
[`DeliveryOutcome`](../../src/agentlane/messaging/_outcome.py).

That outcome is where the caller learns whether the message was delivered,
rejected by policy, failed in the handler, or could not be delivered at all.

## Publish

`publish_message(...)` starts from one topic and expands it into one or more
concrete deliveries. The publish side returns a
[`PublishAck`](../../src/agentlane/messaging/_outcome.py), which tells you how
many recipients were enqueued. It does not tell you whether those recipients
finished their handlers successfully.

That distinction matters when you design workflows. Publish is a fan-out
mechanism, not a multi-recipient RPC.

## Subscriptions And Delivery Modes

Subscriptions map a topic match to an agent type. The main choice is whether
publish deliveries should reuse a stateful recipient or create a fresh one.

[`DeliveryMode.STATEFUL`](../../src/agentlane/messaging/_subscription.py) uses
the topic route key to derive a stable recipient key. That means repeated events
for the same route key reach the same cached agent instance.

[`DeliveryMode.STATELESS`](../../src/agentlane/messaging/_subscription.py)
creates a unique recipient key per delivery. That is useful for fan-out work
where instance reuse is not part of the contract.

## Ordering And Correlation

Ordering is guaranteed per recipient. If multiple deliveries target the same
`AgentId`, they are processed FIFO. Different recipients may run concurrently.

Correlation is a separate concern. Preserve `correlation_id` when work should be
traceable across multiple hops. Use `idempotency_key` when retries need
deduplication semantics at the transport or runtime boundary.

## Example

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

result = await runtime.send_message(
    {"task_id": "t-1"},
    recipient=AgentId.from_values("worker", "session-42"),
)

assert ack.enqueued_recipient_count >= 0
assert result.status.value in {
    "delivered",
    "handler_error",
    "policy_rejected",
    "undeliverable",
}
```
