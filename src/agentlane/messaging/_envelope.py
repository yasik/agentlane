"""Envelope and payload models."""

import enum
from dataclasses import dataclass
from time import time
from typing import Self

from ._identity import (
    AgentId,
    CorrelationId,
    IdempotencyKey,
    MessageId,
    TopicId,
)


def utc_now_ms() -> int:
    """Return current UTC epoch milliseconds."""
    return int(time() * 1000)


class PayloadFormat(enum.StrEnum):
    """Payload representation format."""

    JSON = "json"
    PROTOBUF = "protobuf"
    BYTES = "bytes"


@dataclass(slots=True)
class Payload:
    """Canonical payload container."""

    schema_name: str
    """Logical schema name for payload decoding."""

    content_type: str
    """MIME-like content type."""

    format: PayloadFormat
    """Serialization format marker."""

    data: object
    """Application payload object."""


class MessageKind(enum.StrEnum):
    """Message envelope kind."""

    RPC_REQUEST = "rpc_request"
    RPC_RESPONSE = "rpc_response"
    PUBLISH_EVENT = "publish_event"


@dataclass(slots=True)
class MessageEnvelope:
    """Canonical message envelope."""

    message_id: MessageId
    """Unique identifier for this envelope."""

    correlation_id: CorrelationId | None
    """Correlation chain identifier across related messages."""

    kind: MessageKind
    """Envelope kind (RPC request/response or publish event)."""

    sender: AgentId | None
    """Optional sender identity."""

    recipient: AgentId | None
    """Direct recipient for RPC-like messages."""

    topic: TopicId | None
    """Publish topic for event-like messages."""

    payload: Payload
    """Serialized payload container."""

    created_at_ms: int
    """Envelope creation timestamp in epoch milliseconds."""

    deadline_ms: int | None = None
    """Optional absolute deadline in epoch milliseconds."""

    trace_id: str | None = None
    """Optional tracing identifier."""

    idempotency_key: IdempotencyKey | None = None
    """Optional deduplication key for retry-safe delivery paths."""

    @classmethod
    def new_rpc_request(
        cls,
        *,
        sender: AgentId | None,
        recipient: AgentId,
        payload: Payload,
        correlation_id: CorrelationId | None = None,
        deadline_ms: int | None = None,
        trace_id: str | None = None,
        idempotency_key: IdempotencyKey | None = None,
    ) -> Self:
        """Create a normalized RPC request envelope."""
        return cls(
            message_id=MessageId.new(),
            correlation_id=correlation_id,
            kind=MessageKind.RPC_REQUEST,
            sender=sender,
            recipient=recipient,
            topic=None,
            payload=payload,
            created_at_ms=utc_now_ms(),
            deadline_ms=deadline_ms,
            trace_id=trace_id,
            idempotency_key=idempotency_key,
        )

    @classmethod
    def new_publish_event(
        cls,
        *,
        sender: AgentId | None,
        topic: TopicId,
        payload: Payload,
        correlation_id: CorrelationId | None = None,
        deadline_ms: int | None = None,
        trace_id: str | None = None,
        idempotency_key: IdempotencyKey | None = None,
    ) -> Self:
        """Create a normalized publish event envelope."""
        return cls(
            message_id=MessageId.new(),
            correlation_id=correlation_id,
            kind=MessageKind.PUBLISH_EVENT,
            sender=sender,
            recipient=None,
            topic=topic,
            payload=payload,
            created_at_ms=utc_now_ms(),
            deadline_ms=deadline_ms,
            trace_id=trace_id,
            idempotency_key=idempotency_key,
        )
