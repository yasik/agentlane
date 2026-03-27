"""Messaging primitives for envelopes, routing, and outcomes."""

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
    IdempotencyKey,
    MessageId,
    TopicId,
    Topics,
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
from ._subscription import (
    DeliveryMode,
    PublishRoute,
    Subscription,
    SubscriptionKind,
)

__all__ = [
    "AgentId",
    "AgentKey",
    "AgentType",
    "CorrelationId",
    "DeliveryMode",
    "DeliveryError",
    "DeliveryOutcome",
    "DeliveryStatus",
    "IdempotencyKey",
    "MessageContext",
    "MessageEnvelope",
    "MessageId",
    "MessageKind",
    "Payload",
    "PayloadFormat",
    "PublishAck",
    "PublishRoute",
    "RoutingEngine",
    "RoutingPolicy",
    "SourceKeyAffinityRoutingPolicy",
    "Subscription",
    "SubscriptionKind",
    "Topics",
    "TopicId",
]
