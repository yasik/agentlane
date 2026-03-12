from typing import Any, cast

from google.protobuf import wrappers_pb2
from google.protobuf.message import Message as ProtobufMessage

from agentlane.messaging import (
    AgentId,
    CorrelationId,
    DeliveryError,
    DeliveryOutcome,
    DeliveryStatus,
    IdempotencyKey,
    MessageEnvelope,
    MessageId,
    Payload,
    PayloadFormat,
    PublishAck,
    Subscription,
    TopicId,
)
from agentlane.runtime._distributed_wire import (
    WireDeliveryOutcome,
    WireEnvelope,
    WirePayloadData,
    deserialize_agent_types,
    deserialize_subscriptions,
    serialize_agent_types,
    serialize_subscriptions,
)
from agentlane.transport import (
    ProtobufSerializer,
    SerializerRegistry,
    WireEncoding,
    WirePayload,
    create_default_serializer_registry,
    infer_content_type_for_value,
    infer_schema_id_for_value,
)

StringValue = cast(type[ProtobufMessage], wrappers_pb2.__dict__["StringValue"])
"""Wrapper protobuf message type used for distributed wire tests."""


def _new_string_value(value: str) -> ProtobufMessage:
    message = StringValue()
    cast(Any, message).value = value
    return message


def _payload_for(value: object) -> Payload:
    return Payload(
        schema_name=infer_schema_id_for_value(value).value,
        content_type=infer_content_type_for_value(value).value,
        format=(
            PayloadFormat.PROTOBUF
            if isinstance(value, ProtobufMessage)
            else PayloadFormat.BYTES if isinstance(value, bytes) else PayloadFormat.JSON
        ),
        data=value,
    )


def test_wire_payload_data_roundtrip_preserves_json_payload_shape_and_bytes() -> None:
    registry = create_default_serializer_registry()
    payload = _payload_for({"event": "ready"})

    wire_payload = WirePayloadData.from_payload(
        payload,
        serializer_registry=registry,
    )
    wire_json = wire_payload.to_json()
    restored = WirePayloadData.from_json(wire_payload.to_json()).to_payload(
        serializer_registry=registry
    )

    assert wire_json["schema_id"] == payload.schema_name
    assert wire_json["content_type"] == payload.content_type
    assert restored == payload


def test_wire_payload_data_metadata_payload_keeps_body_opaque() -> None:
    registry = create_default_serializer_registry()
    payload = _payload_for({"event": "ready"})

    wire_payload = WirePayloadData.from_payload(
        payload,
        serializer_registry=registry,
    )
    metadata_payload = wire_payload.to_metadata_payload()

    assert metadata_payload.schema_name == payload.schema_name
    assert metadata_payload.content_type == payload.content_type
    assert metadata_payload.format == payload.format
    assert isinstance(metadata_payload.data, bytes)
    assert metadata_payload.data == wire_payload.to_wire_payload().body


def test_wire_envelope_roundtrip_restores_runtime_envelope() -> None:
    registry = create_default_serializer_registry()
    sender = AgentId.from_values("publisher", "sender-1")
    recipient = AgentId.from_values("worker", "session-1")
    envelope = MessageEnvelope.new_rpc_request(
        sender=sender,
        recipient=recipient,
        payload=_payload_for({"event": "ready"}),
        correlation_id=CorrelationId("corr-1"),
        deadline_ms=123456,
        trace_id="trace-1",
        idempotency_key=IdempotencyKey("idem-1"),
    )

    wire_envelope = WireEnvelope.from_envelope(
        envelope,
        serializer_registry=registry,
    )
    restored = WireEnvelope.from_json(wire_envelope.to_json()).to_envelope(
        serializer_registry=registry
    )

    assert restored == envelope


def test_wire_envelope_metadata_path_preserves_core_type_parity() -> None:
    registry = create_default_serializer_registry()
    sender = AgentId.from_values("publisher", "sender-1")
    topic = TopicId.from_values("alerts", "session-1")
    envelope = MessageEnvelope.new_publish_event(
        sender=sender,
        topic=topic,
        payload=_payload_for({"event": "ready"}),
        correlation_id=CorrelationId("corr-1"),
        deadline_ms=123456,
        trace_id="trace-1",
        idempotency_key=IdempotencyKey("idem-1"),
    )

    wire_envelope = WireEnvelope.from_envelope(
        envelope,
        serializer_registry=registry,
    )
    wire_json = wire_envelope.to_json()
    metadata_envelope = wire_envelope.to_metadata_envelope()

    assert wire_json["sender"] == sender.to_json()
    assert wire_json["topic"] == topic.to_json()
    assert metadata_envelope.sender == sender
    assert metadata_envelope.topic == topic
    assert metadata_envelope.payload.format == PayloadFormat.JSON
    assert isinstance(metadata_envelope.payload.data, bytes)
    assert (
        metadata_envelope.payload.data == wire_envelope.payload.to_wire_payload().body
    )


def test_wire_envelope_json_parity_for_publish_event_topic_and_null_optionals() -> None:
    registry = create_default_serializer_registry()
    topic = TopicId.from_values("alerts", "session-1")
    envelope = MessageEnvelope.new_publish_event(
        sender=None,
        topic=topic,
        payload=_payload_for({"event": "ready"}),
        correlation_id=None,
        deadline_ms=None,
        trace_id=None,
        idempotency_key=None,
    )

    wire_json = WireEnvelope.from_envelope(
        envelope,
        serializer_registry=registry,
    ).to_json()
    restored = WireEnvelope.from_json(wire_json)

    assert wire_json["sender"] is None
    assert wire_json["recipient"] is None
    assert wire_json["correlation_id"] is None
    assert wire_json["deadline_ms"] is None
    assert wire_json["trace_id"] is None
    assert wire_json["idempotency_key"] is None
    assert wire_json["topic"] == topic.to_json()
    assert restored.sender is None
    assert restored.recipient is None
    assert restored.correlation_id is None
    assert restored.deadline_ms is None
    assert restored.trace_id is None
    assert restored.idempotency_key is None
    assert restored.topic == topic


def test_wire_delivery_outcome_roundtrip_restores_json_response() -> None:
    registry = create_default_serializer_registry()
    outcome = DeliveryOutcome(
        status=DeliveryStatus.DELIVERED,
        message_id=MessageId("msg-1"),
        correlation_id=CorrelationId("corr-1"),
        response_payload={"count": 2},
        error=None,
        started_at_ms=10,
        finished_at_ms=20,
    )

    wire_outcome = WireDeliveryOutcome.from_outcome(
        outcome,
        serializer_registry=registry,
    )
    restored = WireDeliveryOutcome.from_json(wire_outcome.to_json()).to_outcome(
        serializer_registry=registry
    )

    assert restored == outcome


def test_wire_delivery_outcome_roundtrip_normalizes_bytes_like_response() -> None:
    registry = create_default_serializer_registry()
    outcome = DeliveryOutcome(
        status=DeliveryStatus.DELIVERED,
        message_id=MessageId("msg-1"),
        correlation_id=CorrelationId("corr-1"),
        response_payload=memoryview(b"payload-bytes"),
        error=None,
        started_at_ms=10,
        finished_at_ms=20,
    )

    wire_outcome = WireDeliveryOutcome.from_outcome(
        outcome,
        serializer_registry=registry,
    )
    restored = WireDeliveryOutcome.from_json(wire_outcome.to_json()).to_outcome(
        serializer_registry=registry
    )

    assert restored.response_payload == b"payload-bytes"
    assert restored.status == outcome.status
    assert restored.message_id == outcome.message_id
    assert restored.correlation_id == outcome.correlation_id


def test_wire_delivery_outcome_roundtrip_restores_protobuf_response() -> None:
    registry = SerializerRegistry()
    response_payload = _new_string_value("ready")
    registry.register(
        ProtobufSerializer(
            schema_id=infer_schema_id_for_value(response_payload),
            message_type=StringValue,
        )
    )
    outcome = DeliveryOutcome(
        status=DeliveryStatus.DELIVERED,
        message_id=MessageId("msg-1"),
        correlation_id=CorrelationId("corr-1"),
        response_payload=response_payload,
        error=None,
        started_at_ms=10,
        finished_at_ms=20,
    )

    wire_outcome = WireDeliveryOutcome.from_outcome(
        outcome,
        serializer_registry=registry,
    )
    restored = WireDeliveryOutcome.from_json(wire_outcome.to_json()).to_outcome(
        serializer_registry=registry
    )

    assert isinstance(restored.response_payload, StringValue)
    assert cast(Any, restored.response_payload).value == "ready"
    assert restored.status == outcome.status
    assert restored.message_id == outcome.message_id
    assert restored.correlation_id == outcome.correlation_id
    assert restored.started_at_ms == outcome.started_at_ms
    assert restored.finished_at_ms == outcome.finished_at_ms


def test_wire_delivery_outcome_json_matches_core_error_shape() -> None:
    registry = create_default_serializer_registry()
    error = DeliveryError(
        code=DeliveryStatus.TIMEOUT,
        message="worker timed out",
        retryable=True,
    )
    outcome = DeliveryOutcome(
        status=DeliveryStatus.TIMEOUT,
        message_id=MessageId("msg-1"),
        correlation_id=CorrelationId("corr-1"),
        response_payload=None,
        error=error,
        started_at_ms=10,
        finished_at_ms=20,
    )

    wire_json = WireDeliveryOutcome.from_outcome(
        outcome,
        serializer_registry=registry,
    ).to_json()

    assert wire_json["error"] == error.to_json()


def test_core_types_roundtrip_with_wire_parity_json_shapes() -> None:
    agent_id = AgentId.from_values("worker", "session-1")
    topic = TopicId.from_values("alerts", "session-1")
    subscription = Subscription.prefix(
        topic_prefix="alerts.",
        agent_type="worker",
    )
    publish_ack = PublishAck(
        message_id=MessageId("msg-1"),
        correlation_id=CorrelationId("corr-1"),
        enqueued_recipient_count=3,
        enqueued_at_ms=100,
    )
    delivery_error = DeliveryError(
        code=DeliveryStatus.UNDELIVERABLE,
        message="missing worker",
        retryable=False,
    )

    assert AgentId.from_json(agent_id.to_json()) == agent_id
    assert TopicId.from_json(topic.to_json()) == topic
    assert Subscription.from_json(subscription.to_json()) == subscription
    assert PublishAck.from_json(publish_ack.to_json()) == publish_ack
    assert DeliveryError.from_json(delivery_error.to_json()) == delivery_error


def test_catalog_snapshot_helpers_match_core_type_shapes() -> None:
    exact_subscription = Subscription.exact(topic_type="alerts", agent_type="worker")
    exact_subscription.id = "sub-alerts"
    prefix_subscription = Subscription.prefix(
        topic_prefix="jobs.",
        agent_type="listener",
    )
    prefix_subscription.id = "sub-jobs"
    subscriptions = (exact_subscription, prefix_subscription)
    agent_types = {subscription.agent_type for subscription in subscriptions}

    serialized_agent_types = serialize_agent_types(agent_types)
    serialized_subscriptions = serialize_subscriptions(subscriptions)

    assert serialized_agent_types == sorted(
        subscription.agent_type.value for subscription in subscriptions
    )
    assert serialized_subscriptions == [
        subscription.to_json() for subscription in subscriptions
    ]
    assert deserialize_agent_types(serialized_agent_types) == agent_types
    assert deserialize_subscriptions(serialized_subscriptions) == subscriptions


def test_wire_payload_data_roundtrip_preserves_explicit_non_text_wire_payload() -> None:
    wire_payload = WirePayload(
        schema_id=infer_schema_id_for_value(b"\x00\xffpayload"),
        content_type=infer_content_type_for_value(b"\x00\xffpayload"),
        encoding=WireEncoding.BYTES,
        body=b"\x00\xffpayload",
    )

    wire_json = WirePayloadData.from_wire_payload(wire_payload).to_json()
    restored = WirePayloadData.from_json(wire_json).to_wire_payload()

    assert wire_json["body"] == "AP9wYXlsb2Fk"
    assert restored == wire_payload
