"""Messaging primitives for envelopes, routing, and outcomes."""

from ._cancellation import CancellationToken
from ._context import MessageContext
from ._envelope import (
    MessageEnvelope,
    MessageKind,
    Payload,
    PayloadFormat,
)
from ._identity import (
    AgentId,
    AgentKey,
    AgentType,
    CorrelationId,
    MessageId,
    TopicId,
)
from ._outcome import (
    DeliveryError,
    DeliveryOutcome,
    DeliveryStatus,
    PublishAck,
)
from ._routing import RoutingEngine
from ._routing_policy import (
    RoutingPolicy,
    SourceKeyAffinityRoutingPolicy,
)
from ._subscription import Subscription, SubscriptionKind

__all__ = [
    "AgentId",
    "AgentKey",
    "AgentType",
    "CancellationToken",
    "CorrelationId",
    "DeliveryError",
    "DeliveryOutcome",
    "DeliveryStatus",
    "MessageContext",
    "MessageEnvelope",
    "MessageId",
    "MessageKind",
    "Payload",
    "PayloadFormat",
    "PublishAck",
    "RoutingEngine",
    "RoutingPolicy",
    "SourceKeyAffinityRoutingPolicy",
    "Subscription",
    "SubscriptionKind",
    "TopicId",
]
