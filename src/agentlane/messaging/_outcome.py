"""Delivery outcomes and error primitives."""

import enum
from dataclasses import dataclass
from typing import Self, cast

from agentlane.util import utc_now_ms

from ._identity import CorrelationId, MessageId


class DeliveryStatus(enum.StrEnum):
    """Terminal delivery states."""

    DELIVERED = "delivered"
    """Handler completed successfully and produced a terminal success outcome."""

    DROPPED = "dropped"
    """Message was intentionally discarded by runtime policy."""

    UNDELIVERABLE = "undeliverable"
    """Recipient could not be resolved/created, so dispatch never reached a handler."""

    TIMEOUT = "timeout"
    """Delivery exceeded configured processing deadline before completion."""

    CANCELED = "canceled"
    """Delivery was canceled, typically during runtime shutdown or cooperative cancellation."""

    HANDLER_ERROR = "handler_error"
    """Agent handler resolution or execution raised an exception during dispatch."""

    SERIALIZATION_ERROR = "serialization_error"
    """Payload serialization/deserialization failed before handler execution."""

    POLICY_REJECTED = "policy_rejected"
    """Runtime policy rejected dispatch before execution (e.g., routing/scheduling policy)."""


@dataclass(slots=True)
class DeliveryError:
    """Structured error details for unsuccessful deliveries."""

    code: DeliveryStatus
    """Canonical status code associated with the error."""

    message: str
    """Human-readable error message."""

    retryable: bool
    """True if delivery may be retried safely."""

    @classmethod
    def from_json(cls, data: object) -> Self:
        """Construct a DeliveryError from a JSON-safe mapping."""
        if not isinstance(data, dict):
            raise TypeError("Expected JSON object for delivery error.")

        mapping = cast(dict[str, object], data)
        code = mapping.get("code")
        message = mapping.get("message")
        retryable = mapping.get("retryable")

        if not isinstance(code, str):
            raise TypeError("Expected string for delivery error code.")
        if not isinstance(message, str):
            raise TypeError("Expected string for delivery error message.")
        if not isinstance(retryable, bool):
            raise TypeError("Expected bool for delivery error retryable.")
        return cls(
            code=DeliveryStatus(code),
            message=message,
            retryable=retryable,
        )

    def to_json(self) -> dict[str, object]:
        """Return a JSON-safe mapping for this delivery error."""
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
        }


@dataclass(slots=True)
class DeliveryOutcome:
    """Structured delivery result."""

    status: DeliveryStatus
    """Terminal delivery status."""

    message_id: MessageId
    """Envelope message id."""

    correlation_id: CorrelationId | None
    """Correlation chain identifier."""

    response_payload: object | None
    """Optional response payload for RPC-style delivery."""

    error: DeliveryError | None
    """Structured error details for failed delivery."""

    started_at_ms: int
    """Dispatch start timestamp in epoch milliseconds."""

    finished_at_ms: int
    """Dispatch completion timestamp in epoch milliseconds."""

    @classmethod
    def delivered(
        cls,
        *,
        message_id: MessageId,
        correlation_id: CorrelationId | None,
        response_payload: object | None,
        started_at_ms: int | None = None,
    ) -> Self:
        """Create a successful delivery outcome.

        Args:
            message_id: Envelope message id.
            correlation_id: Optional correlation id for the causal chain.
            response_payload: Optional handler response payload.
            started_at_ms: Optional dispatch start timestamp override.

        Returns:
            Self: Successful delivery outcome.
        """
        started = started_at_ms if started_at_ms is not None else utc_now_ms()
        return cls(
            status=DeliveryStatus.DELIVERED,
            message_id=message_id,
            correlation_id=correlation_id,
            response_payload=response_payload,
            error=None,
            started_at_ms=started,
            finished_at_ms=utc_now_ms(),
        )

    @classmethod
    def failed(
        cls,
        *,
        status: DeliveryStatus,
        message_id: MessageId,
        correlation_id: CorrelationId | None,
        message: str,
        retryable: bool,
        started_at_ms: int | None = None,
    ) -> Self:
        """Create a failed delivery outcome.

        Args:
            status: Failure status code.
            message_id: Envelope message id.
            correlation_id: Optional correlation id for the causal chain.
            message: Human-readable error detail.
            retryable: Whether caller can safely retry this delivery.
            started_at_ms: Optional dispatch start timestamp override.

        Returns:
            Self: Failed delivery outcome.
        """
        started = started_at_ms if started_at_ms is not None else utc_now_ms()
        return cls(
            status=status,
            message_id=message_id,
            correlation_id=correlation_id,
            response_payload=None,
            error=DeliveryError(
                code=status,
                message=message,
                retryable=retryable,
            ),
            started_at_ms=started,
            finished_at_ms=utc_now_ms(),
        )


@dataclass(slots=True)
class PublishAck:
    """Acknowledgment returned for publish enqueue."""

    message_id: MessageId
    """Publish envelope id."""

    correlation_id: CorrelationId | None
    """Correlation id for the published causal chain."""

    enqueued_recipient_count: int
    """Number of recipients enqueued by routing."""

    enqueued_at_ms: int
    """Enqueue acknowledgment timestamp in epoch milliseconds."""

    @classmethod
    def from_json(cls, data: object) -> Self:
        """Construct a PublishAck from a JSON-safe mapping."""
        if not isinstance(data, dict):
            raise TypeError("Expected JSON object for publish ack.")

        mapping = cast(dict[str, object], data)
        message_id = mapping.get("message_id")
        correlation_id = mapping.get("correlation_id")
        count = mapping.get("enqueued_recipient_count")
        enqueued_at_ms = mapping.get("enqueued_at_ms")

        if not isinstance(message_id, str):
            raise TypeError("Expected string for publish ack id.")
        if correlation_id is not None and not isinstance(correlation_id, str):
            raise TypeError("Expected string for publish ack correlation id.")
        if not isinstance(count, int):
            raise TypeError("Expected integer for publish ack count.")
        if not isinstance(enqueued_at_ms, int):
            raise TypeError("Expected integer for publish ack timestamp.")
        return cls(
            message_id=MessageId(message_id),
            correlation_id=(
                CorrelationId(correlation_id) if correlation_id is not None else None
            ),
            enqueued_recipient_count=count,
            enqueued_at_ms=enqueued_at_ms,
        )

    def to_json(self) -> dict[str, object]:
        """Return a JSON-safe mapping for this publish acknowledgment."""
        return {
            "message_id": self.message_id.value,
            "correlation_id": (
                self.correlation_id.value if self.correlation_id is not None else None
            ),
            "enqueued_recipient_count": self.enqueued_recipient_count,
            "enqueued_at_ms": self.enqueued_at_ms,
        }
