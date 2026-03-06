"""Delivery outcomes and error primitives."""

import enum
from dataclasses import dataclass, field
from time import time
from typing import Self

from ._identity import CorrelationId, MessageId


def utc_now_ms() -> int:
    """Return current UTC epoch milliseconds."""
    return int(time() * 1000)


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

    details: dict[str, str] = field(default_factory=dict)
    """Structured extra metadata for debugging and triage."""


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
        """Create a successful delivery outcome."""
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
        details: dict[str, str] | None = None,
        started_at_ms: int | None = None,
    ) -> Self:
        """Create a failed delivery outcome."""
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
                details=details or {},
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
