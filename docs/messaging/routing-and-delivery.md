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
[`SubscriptionKind`](../../src/agentlane/messaging/_subscription.py)
(`TYPE_EXACT` / `TYPE_PREFIX`), and the post-match delivery lifecycle is
governed by [`DeliveryMode`](../../src/agentlane/messaging/_subscription.py).

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
target instead. It stores a topic `type` and a `source`, where `route_key` is
exposed as a read-only property aliasing `source`. The route key is what later
lets publish deliveries preserve stateful affinity for the same logical stream
of work. Build one with `TopicId.from_values(type_value=..., route_key=...)` or
the [`Topics`](../../src/agentlane/messaging/_identity.py) convenience
constructor `Topics.id(type_value, route_key)`.

Once a delivery is created, it travels as a
[`MessageEnvelope`](../../src/agentlane/messaging/_envelope.py). The envelope
holds the sender, recipient or topic, the
[`Payload`](../../src/agentlane/messaging/_envelope.py), and correlation data
that must survive transport. A `Payload` carries a `schema_name`,
`content_type`, a [`PayloadFormat`](../../src/agentlane/messaging/_envelope.py)
(`JSON` / `PROTOBUF` / `BYTES`), and the application `data`. The envelope's
[`MessageKind`](../../src/agentlane/messaging/_envelope.py)
(`RPC_REQUEST` / `RPC_RESPONSE` / `PUBLISH_EVENT`) records whether it is a
direct send or a published event.

Correlation and identity primitives are typed value wrappers:
[`CorrelationId`](../../src/agentlane/messaging/_identity.py),
[`IdempotencyKey`](../../src/agentlane/messaging/_identity.py), and
[`MessageId`](../../src/agentlane/messaging/_identity.py) each expose a `.new()`
factory that mints a fresh identifier.

## Direct Send

`send_message(...)` resolves one recipient, enqueues one delivery, and waits for
one terminal
[`DeliveryOutcome`](../../src/agentlane/messaging/_outcome.py).

That outcome is where the caller learns how the message resolved. The status is
a [`DeliveryStatus`](../../src/agentlane/messaging/_outcome.py), which has eight
terminal states:

- `DELIVERED` — handler completed successfully.
- `DROPPED` — message was intentionally discarded by runtime policy.
- `UNDELIVERABLE` — recipient could not be resolved/created, so dispatch never
  reached a handler.
- `TIMEOUT` — delivery exceeded the configured processing deadline.
- `CANCELED` — delivery was canceled (for example during shutdown or cooperative
  cancellation).
- `HANDLER_ERROR` — handler resolution or execution raised an exception.
- `SERIALIZATION_ERROR` — payload serialization/deserialization failed before
  the handler ran.
- `POLICY_REJECTED` — runtime policy rejected dispatch before execution.

## Publish

`publish_message(...)` starts from one topic and expands it into one or more
concrete deliveries. The publish side returns a
[`PublishAck`](../../src/agentlane/messaging/_outcome.py), which tells you how
many recipients were enqueued. It does not tell you whether those recipients
finished their handlers successfully.

That distinction matters when you design workflows. Publish is a fan-out
mechanism, not a multi-recipient RPC.

## Subscriptions And Delivery Modes

A [`Subscription`](../../src/agentlane/messaging/_subscription.py) maps a topic
match to an agent type. Two decisions shape it: how a topic is matched, and how
matched deliveries reuse recipients.

### Matching: SubscriptionKind

[`SubscriptionKind`](../../src/agentlane/messaging/_subscription.py) selects the
matching strategy:

- `TYPE_EXACT` matches only when `topic.type` exactly equals the pattern.
- `TYPE_PREFIX` matches when `topic.type` starts with the pattern.

Create subscriptions through the convenience constructors `Subscription.exact`
and `Subscription.prefix`, or register them directly on the runtime with
`runtime.subscribe_exact(topic_type=..., agent_type=...)` and
`runtime.subscribe_prefix(topic_prefix=..., agent_type=...)`.

### Delivery: DeliveryMode

After a subscription matches, `DeliveryMode` controls the delivery lifecycle.
The main choice is whether publish deliveries should reuse a stateful recipient
or create a fresh one.

[`DeliveryMode.STATEFUL`](../../src/agentlane/messaging/_subscription.py) uses
the topic route key to derive a stable recipient key. That means repeated events
for the same route key reach the same cached agent instance.

[`DeliveryMode.STATELESS`](../../src/agentlane/messaging/_subscription.py)
creates a unique recipient key per delivery. That is useful for fan-out work
where instance reuse is not part of the contract.

### Routing Policy

The [`RoutingEngine`](../../src/agentlane/messaging/_routing.py) evaluates active
subscriptions against a published topic through a
[`RoutingPolicy`](../../src/agentlane/messaging/_routing_policy.py). The default
[`SourceKeyAffinityRoutingPolicy`](../../src/agentlane/messaging/_routing_policy.py)
maps each matched topic's route key onto the recipient agent key, then dedups
and orders the resulting `PublishRoute` list deterministically: stateful routes
are deduped by concrete recipient id, stateless routes by subscription plus
recipient type, and both are stably sorted so fan-out is reproducible across
runs.

## Ordering And Correlation

Ordering is guaranteed per recipient. If multiple deliveries target the same
`AgentId`, they are processed FIFO. Different recipients may run concurrently.

Correlation is a separate concern. Preserve `correlation_id` when work should be
traceable across multiple hops. Use `idempotency_key` when retries need
deduplication semantics at the transport or runtime boundary.

By default the runtime generates the envelope `message_id`. Pass an explicit
`message_id` to `send_message(...)` when the sender must know the id before
delivery — for example to bridge trace context across a hop, where the sender
keys a snapshot under the id and the receiver looks the same id up via
`MessageContext.message_id`.

## Example

This snippet shows the API shape. It registers a subscription but no agent
factory for the `worker` type, so the stateful-affinity behavior described above
is not exercised here — a real run requires a registered agent factory for the
recipient type. The status assertion therefore lists a representative subset of
the `DeliveryStatus` states rather than asserting a single result.

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
