import pytest

from agentlane.messaging import (
    AgentId,
    AgentType,
    MessageEnvelope,
    Payload,
    PayloadFormat,
    RoutingEngine,
    Subscription,
    SubscriptionKind,
    TopicId,
)


def test_router_resolve_publish_recipients_with_source_key_affinity() -> None:
    router = RoutingEngine()
    router.add_subscription(
        Subscription(
            kind=SubscriptionKind.TYPE_EXACT,
            agent_type=AgentType("worker"),
            topic_pattern="jobs",
        )
    )
    envelope = MessageEnvelope.new_publish_event(
        sender=None,
        topic=TopicId(type="jobs", source="tenant-1"),
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
        topic=TopicId(type="unknown", source="tenant-1"),
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
        topic=TopicId(type="jobs", source="tenant-1"),
        payload=Payload(
            schema_name="str",
            content_type="application/python-object",
            format=PayloadFormat.JSON,
            data="hello",
        ),
    )

    with pytest.raises(LookupError):
        router.resolve_rpc_recipient(envelope)
