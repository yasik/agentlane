"""JSON-safe wire models for distributed runtime transport.

These helpers let the host and worker exchange runtime metadata and opaque payload
bytes without introducing generated protobuf message types into the core runtime
package. The host can inspect routing metadata without deserializing application
payloads, while workers can fully decode payloads once a delivery reaches local
execution.
"""

from base64 import b64decode, b64encode
from dataclasses import dataclass
from typing import cast

from agentlane.messaging import (
    AgentId,
    AgentType,
    CorrelationId,
    DeliveryError,
    DeliveryOutcome,
    DeliveryStatus,
    IdempotencyKey,
    MessageEnvelope,
    MessageId,
    MessageKind,
    Payload,
    Subscription,
    TopicId,
)
from agentlane.transport import (
    ContentType,
    SchemaId,
    SerializerRegistry,
    WireEncoding,
    WirePayload,
    payload_format_for_wire_encoding,
    payload_to_wire_payload,
    wire_payload_to_payload,
)

from ._message_helpers import payload_from_value

type JsonObject = dict[str, object]
"""JSON-compatible object used by distributed runtime transport."""


def _expect_mapping(value: object, *, context: str) -> JsonObject:
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object for {context}.")
    return cast(JsonObject, value)


def _expect_str(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected string for {context}.")
    return value


def _optional_str(value: object, *, context: str) -> str | None:
    if value is None:
        return None
    return _expect_str(value, context=context)


def _expect_int(value: object, *, context: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"Expected integer for {context}.")
    return value


def _optional_int(value: object, *, context: str) -> int | None:
    if value is None:
        return None
    return _expect_int(value, context=context)


def _expect_list(value: object, *, context: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"Expected list for {context}.")
    return cast(list[object], value)


@dataclass(frozen=True, slots=True)
class WirePayloadData:
    """JSON-safe representation of `WirePayload`."""

    schema_id: str
    content_type: str
    encoding: str
    body_base64: str

    @classmethod
    def from_wire_payload(cls, payload: WirePayload) -> "WirePayloadData":
        return cls(
            schema_id=payload.schema_id.value,
            content_type=payload.content_type.value,
            encoding=payload.encoding.value,
            body_base64=b64encode(payload.body).decode("ascii"),
        )

    @classmethod
    def from_payload(
        cls,
        payload: Payload,
        *,
        serializer_registry: SerializerRegistry,
    ) -> "WirePayloadData":
        wire_payload = payload_to_wire_payload(payload, registry=serializer_registry)
        return cls.from_wire_payload(wire_payload)

    @classmethod
    def from_json(cls, data: object) -> "WirePayloadData":
        mapping = _expect_mapping(data, context="wire payload")
        return cls(
            schema_id=_expect_str(mapping["schema_id"], context="wire payload schema"),
            content_type=_expect_str(
                mapping["content_type"], context="wire payload content type"
            ),
            encoding=_expect_str(mapping["encoding"], context="wire payload encoding"),
            body_base64=_expect_str(mapping["body"], context="wire payload body"),
        )

    def to_wire_payload(self) -> WirePayload:
        return WirePayload(
            schema_id=SchemaId(self.schema_id),
            content_type=ContentType(self.content_type),
            encoding=WireEncoding(self.encoding),
            body=b64decode(self.body_base64.encode("ascii")),
        )

    def to_metadata_payload(self) -> Payload:
        """Return routing metadata while leaving the payload body opaque.

        The host may inspect schema/content metadata for transport bookkeeping, but
        it must not deserialize the application payload during routing.
        """
        wire_payload = self.to_wire_payload()
        return Payload(
            schema_name=wire_payload.schema_id.value,
            content_type=wire_payload.content_type.value,
            format=payload_format_for_wire_encoding(wire_payload.encoding),
            data=wire_payload.body,
        )

    def to_payload(self, *, serializer_registry: SerializerRegistry) -> Payload:
        """Decode the wire payload into the runtime `Payload` model."""
        return wire_payload_to_payload(
            self.to_wire_payload(),
            registry=serializer_registry,
        )

    def to_json(self) -> JsonObject:
        return {
            "schema_id": self.schema_id,
            "content_type": self.content_type,
            "encoding": self.encoding,
            "body": self.body_base64,
        }


@dataclass(frozen=True, slots=True)
class WireEnvelope:
    """JSON-safe `MessageEnvelope` that keeps payload bytes opaque in transit."""

    message_id: str
    correlation_id: str | None
    kind: str
    sender: AgentId | None
    recipient: AgentId | None
    topic: TopicId | None
    payload: WirePayloadData
    created_at_ms: int
    deadline_ms: int | None
    trace_id: str | None
    idempotency_key: str | None

    @classmethod
    def from_envelope(
        cls,
        envelope: MessageEnvelope,
        *,
        serializer_registry: SerializerRegistry,
    ) -> "WireEnvelope":
        """Serialize one runtime envelope for host/worker transport."""
        return cls(
            message_id=envelope.message_id.value,
            correlation_id=(
                envelope.correlation_id.value
                if envelope.correlation_id is not None
                else None
            ),
            kind=envelope.kind.value,
            sender=envelope.sender,
            recipient=envelope.recipient,
            topic=envelope.topic,
            payload=WirePayloadData.from_payload(
                envelope.payload,
                serializer_registry=serializer_registry,
            ),
            created_at_ms=envelope.created_at_ms,
            deadline_ms=envelope.deadline_ms,
            trace_id=envelope.trace_id,
            idempotency_key=(
                envelope.idempotency_key.value
                if envelope.idempotency_key is not None
                else None
            ),
        )

    @classmethod
    def from_json(cls, data: object) -> "WireEnvelope":
        """Parse one wire envelope from JSON-safe transport data."""
        mapping = _expect_mapping(data, context="wire envelope")
        sender = mapping.get("sender")
        recipient = mapping.get("recipient")
        topic = mapping.get("topic")
        return cls(
            message_id=_expect_str(mapping["message_id"], context="message id"),
            correlation_id=_optional_str(
                mapping.get("correlation_id"),
                context="correlation id",
            ),
            kind=_expect_str(mapping["kind"], context="message kind"),
            sender=AgentId.from_json(sender) if sender is not None else None,
            recipient=AgentId.from_json(recipient) if recipient is not None else None,
            topic=TopicId.from_json(topic) if topic is not None else None,
            payload=WirePayloadData.from_json(mapping["payload"]),
            created_at_ms=_expect_int(
                mapping["created_at_ms"], context="created_at_ms"
            ),
            deadline_ms=_optional_int(
                mapping.get("deadline_ms"),
                context="deadline_ms",
            ),
            trace_id=_optional_str(mapping.get("trace_id"), context="trace_id"),
            idempotency_key=_optional_str(
                mapping.get("idempotency_key"),
                context="idempotency_key",
            ),
        )

    def to_envelope(
        self, *, serializer_registry: SerializerRegistry
    ) -> MessageEnvelope:
        """Decode one wire envelope into the runtime envelope model."""
        return MessageEnvelope(
            message_id=MessageId(self.message_id),
            correlation_id=(
                CorrelationId(self.correlation_id)
                if self.correlation_id is not None
                else None
            ),
            kind=MessageKind(self.kind),
            sender=self.sender,
            recipient=self.recipient,
            topic=self.topic,
            payload=self.payload.to_payload(serializer_registry=serializer_registry),
            created_at_ms=self.created_at_ms,
            deadline_ms=self.deadline_ms,
            trace_id=self.trace_id,
            idempotency_key=(
                IdempotencyKey(self.idempotency_key)
                if self.idempotency_key is not None
                else None
            ),
        )

    def to_metadata_envelope(self) -> MessageEnvelope:
        """Build one envelope for routing without decoding application payloads.

        This is the host-side path for publish routing: inspect topic and delivery
        metadata only, then hand the original opaque bytes back to the destination
        worker for real deserialization.
        """
        return MessageEnvelope(
            message_id=MessageId(self.message_id),
            correlation_id=(
                CorrelationId(self.correlation_id)
                if self.correlation_id is not None
                else None
            ),
            kind=MessageKind(self.kind),
            sender=self.sender,
            recipient=self.recipient,
            topic=self.topic,
            payload=self.payload.to_metadata_payload(),
            created_at_ms=self.created_at_ms,
            deadline_ms=self.deadline_ms,
            trace_id=self.trace_id,
            idempotency_key=(
                IdempotencyKey(self.idempotency_key)
                if self.idempotency_key is not None
                else None
            ),
        )

    def to_json(self) -> JsonObject:
        return {
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "kind": self.kind,
            "sender": self.sender.to_json() if self.sender is not None else None,
            "recipient": (
                self.recipient.to_json() if self.recipient is not None else None
            ),
            "topic": self.topic.to_json() if self.topic is not None else None,
            "payload": self.payload.to_json(),
            "created_at_ms": self.created_at_ms,
            "deadline_ms": self.deadline_ms,
            "trace_id": self.trace_id,
            "idempotency_key": self.idempotency_key,
        }


@dataclass(frozen=True, slots=True)
class WireDeliveryOutcome:
    """JSON-safe representation of `DeliveryOutcome`.

    Response payloads stay in transport form on the wire and are decoded back into
    the caller-facing `.response_payload` value at the receiving edge.
    """

    status: str
    message_id: str
    correlation_id: str | None
    response_payload: WirePayloadData | None
    error: DeliveryError | None
    started_at_ms: int
    finished_at_ms: int

    @classmethod
    def from_outcome(
        cls,
        outcome: DeliveryOutcome,
        *,
        serializer_registry: SerializerRegistry,
    ) -> "WireDeliveryOutcome":
        response_payload = None
        if outcome.response_payload is not None:
            response_payload = WirePayloadData.from_payload(
                payload_from_value(outcome.response_payload),
                serializer_registry=serializer_registry,
            )
        return cls(
            status=outcome.status.value,
            message_id=outcome.message_id.value,
            correlation_id=(
                outcome.correlation_id.value
                if outcome.correlation_id is not None
                else None
            ),
            response_payload=response_payload,
            error=outcome.error,
            started_at_ms=outcome.started_at_ms,
            finished_at_ms=outcome.finished_at_ms,
        )

    @classmethod
    def from_json(cls, data: object) -> "WireDeliveryOutcome":
        mapping = _expect_mapping(data, context="delivery outcome")
        response_payload = mapping.get("response_payload")
        error = mapping.get("error")
        return cls(
            status=_expect_str(mapping["status"], context="delivery outcome status"),
            message_id=_expect_str(
                mapping["message_id"],
                context="delivery outcome message_id",
            ),
            correlation_id=_optional_str(
                mapping.get("correlation_id"),
                context="delivery outcome correlation_id",
            ),
            response_payload=(
                WirePayloadData.from_json(response_payload)
                if response_payload is not None
                else None
            ),
            error=DeliveryError.from_json(error) if error is not None else None,
            started_at_ms=_expect_int(
                mapping["started_at_ms"],
                context="delivery outcome started_at_ms",
            ),
            finished_at_ms=_expect_int(
                mapping["finished_at_ms"],
                context="delivery outcome finished_at_ms",
            ),
        )

    def to_outcome(self, *, serializer_registry: SerializerRegistry) -> DeliveryOutcome:
        """Decode one wire delivery outcome into the runtime result model.

        The returned `DeliveryOutcome` exposes the decoded payload `.data`, not a
        transport `Payload` wrapper.
        """
        response_payload: object | None = None
        if self.response_payload is not None:
            response_payload = self.response_payload.to_payload(
                serializer_registry=serializer_registry
            ).data
        return DeliveryOutcome(
            status=DeliveryStatus(self.status),
            message_id=MessageId(self.message_id),
            correlation_id=(
                CorrelationId(self.correlation_id)
                if self.correlation_id is not None
                else None
            ),
            response_payload=response_payload,
            error=self.error,
            started_at_ms=self.started_at_ms,
            finished_at_ms=self.finished_at_ms,
        )

    def to_json(self) -> JsonObject:
        return {
            "status": self.status,
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "response_payload": (
                self.response_payload.to_json()
                if self.response_payload is not None
                else None
            ),
            "error": self.error.to_json() if self.error is not None else None,
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
        }


def serialize_agent_types(agent_types: set[AgentType]) -> list[str]:
    """Serialize agent types for catalog sync."""
    return sorted(agent_type.value for agent_type in agent_types)


def deserialize_agent_types(data: object) -> set[AgentType]:
    """Deserialize agent types from a catalog sync payload."""
    return {
        AgentType(_expect_str(item, context="agent type"))
        for item in _expect_list(data, context="agent types")
    }


def serialize_subscriptions(
    subscriptions: tuple[Subscription, ...],
) -> list[JsonObject]:
    """Serialize subscription snapshots for catalog sync."""
    return [subscription.to_json() for subscription in subscriptions]


def deserialize_subscriptions(data: object) -> tuple[Subscription, ...]:
    """Deserialize subscription snapshots from catalog sync payload."""
    return tuple(
        Subscription.from_json(item)
        for item in _expect_list(data, context="subscriptions")
    )
