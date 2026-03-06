import pytest

from agentlane.messaging import (
    AgentId,
    AgentType,
    DeliveryMode,
    IdempotencyKey,
    MessageEnvelope,
    Payload,
    PayloadFormat,
    RoutingEngine,
    Subscription,
    TopicId,
    Topics,
)


def test_router_resolve_publish_recipients_with_source_key_affinity() -> None:
    router = RoutingEngine()
    router.add_subscription(
        Subscription.exact(
            topic_type="jobs",
            agent_type=AgentType("worker"),
        )
    )
    envelope = MessageEnvelope.new_publish_event(
        sender=None,
        topic=TopicId.from_values(type_value="jobs", route_key="tenant-1"),
        payload=Payload(
            schema_name="dict",
            content_type="application/python-object",
            format=PayloadFormat.JSON,
            data={"task": 1},
        ),
    )

    recipients = router.resolve_publish_recipients(envelope)
    assert recipients == [AgentId.from_values("worker", "tenant-1")]


def test_router_resolve_publish_recipients_empty_when_no_match() -> None:
    router = RoutingEngine()
    envelope = MessageEnvelope.new_publish_event(
        sender=None,
        topic=TopicId.from_values(type_value="unknown", route_key="tenant-1"),
        payload=Payload(
            schema_name="str",
            content_type="application/python-object",
            format=PayloadFormat.JSON,
            data="hello",
        ),
    )

    assert router.resolve_publish_recipients(envelope) == []


def test_router_raises_for_missing_rpc_recipient() -> None:
    router = RoutingEngine()
    envelope = MessageEnvelope.new_publish_event(
        sender=None,
        topic=TopicId.from_values(type_value="jobs", route_key="tenant-1"),
        payload=Payload(
            schema_name="str",
            content_type="application/python-object",
            format=PayloadFormat.JSON,
            data="hello",
        ),
    )

    with pytest.raises(LookupError):
        router.resolve_rpc_recipient(envelope)


def test_router_resolve_publish_routes_includes_delivery_mode() -> None:
    router = RoutingEngine()
    router.add_subscription(
        Subscription.exact(
            topic_type="jobs",
            agent_type=AgentType("stateful-worker"),
            delivery_mode=DeliveryMode.STATEFUL,
        )
    )
    router.add_subscription(
        Subscription.exact(
            topic_type="jobs",
            agent_type=AgentType("stateless-worker"),
            delivery_mode=DeliveryMode.STATELESS,
        )
    )
    envelope = MessageEnvelope.new_publish_event(
        sender=None,
        topic=TopicId.from_values(type_value="jobs", route_key="tenant-1"),
        payload=Payload(
            schema_name="dict",
            content_type="application/python-object",
            format=PayloadFormat.JSON,
            data={"task": 1},
        ),
    )

    routes = router.resolve_publish_routes(envelope)
    route_modes = {
        (route.recipient.type.value, route.delivery_mode.value) for route in routes
    }
    assert route_modes == {
        ("stateful-worker", "stateful"),
        ("stateless-worker", "stateless"),
    }


def test_topic_helpers_build_route_key_alias() -> None:
    topic = TopicId.from_values(type_value="jobs", route_key="tenant-1")
    assert topic.type == "jobs"
    assert topic.source == "tenant-1"
    assert topic.route_key == "tenant-1"

    from_topics = Topics.id(type_value="jobs", route_key="tenant-1")
    assert from_topics == topic


def test_message_envelope_uses_typed_idempotency_key() -> None:
    key = IdempotencyKey.new()
    rpc_envelope = MessageEnvelope.new_rpc_request(
        sender=None,
        recipient=AgentId.from_values("worker", "tenant-1"),
        payload=Payload(
            schema_name="str",
            content_type="application/python-object",
            format=PayloadFormat.JSON,
            data="hello",
        ),
        idempotency_key=key,
    )
    publish_envelope = MessageEnvelope.new_publish_event(
        sender=None,
        topic=TopicId.from_values(type_value="jobs", route_key="tenant-1"),
        payload=Payload(
            schema_name="str",
            content_type="application/python-object",
            format=PayloadFormat.JSON,
            data="hello",
        ),
        idempotency_key=key,
    )

    assert rpc_envelope.idempotency_key == key
    assert publish_envelope.idempotency_key == key
