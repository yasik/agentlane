"""Private runtime message normalization helpers."""

from google.protobuf.message import Message as ProtobufMessage

from agentlane.messaging import (
    AgentId,
    AgentKey,
    DeliveryMode,
    MessageEnvelope,
    Payload,
    PayloadFormat,
    PublishRoute,
)
from agentlane.transport import (
    infer_content_type_for_value,
    infer_schema_id_for_value,
)


def payload_from_value(value: object) -> Payload:
    """Wrap one application value into the canonical `Payload` container."""
    payload_data = value
    payload_format = PayloadFormat.JSON
    schema_source = value
    if isinstance(value, bytes):
        payload_format = PayloadFormat.BYTES
    elif isinstance(value, bytearray):
        payload_data = bytes(value)
        schema_source = payload_data
        payload_format = PayloadFormat.BYTES
    elif isinstance(value, memoryview):
        payload_data = value.tobytes()
        schema_source = payload_data
        payload_format = PayloadFormat.BYTES
    elif isinstance(value, ProtobufMessage):
        payload_format = PayloadFormat.PROTOBUF

    schema_id = infer_schema_id_for_value(schema_source)
    content_type = infer_content_type_for_value(schema_source)
    return Payload(
        schema_name=schema_id.value,
        content_type=content_type.value,
        format=payload_format,
        data=payload_data,
    )


def recipient_for_publish_route(
    *,
    route: PublishRoute,
    publish_envelope: MessageEnvelope,
) -> AgentId:
    """Resolve the concrete runtime recipient for one publish route."""
    if route.delivery_mode == DeliveryMode.STATEFUL:
        return route.recipient

    transient_key = AgentKey(
        f"stateless:{route.subscription_id}:{publish_envelope.message_id.value}"
    )
    return AgentId(type=route.recipient.type, key=transient_key)
