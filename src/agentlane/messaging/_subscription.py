"""Subscription models used by publish routing."""

import enum
from dataclasses import dataclass, field
from uuid import uuid4

from ._identity import AgentId, AgentKey, AgentType, TopicId


class SubscriptionKind(enum.StrEnum):
    """Supported subscription matching strategies."""

    TYPE_EXACT = "type_exact"
    """Match only when `topic.type` exactly equals `topic_pattern`."""

    TYPE_PREFIX = "type_prefix"
    """Match when `topic.type` starts with `topic_pattern`."""


class DeliveryMode(enum.StrEnum):
    """Delivery lifecycle mode for matching subscriptions."""

    STATEFUL = "stateful"
    """Reuse cached agent instance by route-key-derived recipient id."""

    STATELESS = "stateless"
    """Use unique per-delivery recipient ids so instance reuse is not guaranteed."""


@dataclass(slots=True)
class Subscription:
    """Route publish events to agent types based on topic patterns."""

    kind: SubscriptionKind
    """Matching strategy for topic resolution."""

    agent_type: AgentType
    """Recipient agent type for matched topics."""

    topic_pattern: str
    """Topic pattern used by the selected matching strategy."""

    delivery_mode: DeliveryMode = DeliveryMode.STATEFUL
    """Whether delivery reuses keyed instances or creates transient instances."""

    id: str = field(default_factory=lambda: str(uuid4()))
    """Stable subscription identifier."""

    @classmethod
    def exact(
        cls,
        *,
        topic_type: str,
        agent_type: AgentType | str,
        delivery_mode: DeliveryMode = DeliveryMode.STATEFUL,
    ) -> "Subscription":
        """Create a type-exact subscription."""
        normalized_agent_type = (
            agent_type if isinstance(agent_type, AgentType) else AgentType(agent_type)
        )
        return cls(
            kind=SubscriptionKind.TYPE_EXACT,
            agent_type=normalized_agent_type,
            topic_pattern=topic_type,
            delivery_mode=delivery_mode,
        )

    @classmethod
    def prefix(
        cls,
        *,
        topic_prefix: str,
        agent_type: AgentType | str,
        delivery_mode: DeliveryMode = DeliveryMode.STATEFUL,
    ) -> "Subscription":
        """Create a type-prefix subscription."""
        normalized_agent_type = (
            agent_type if isinstance(agent_type, AgentType) else AgentType(agent_type)
        )
        return cls(
            kind=SubscriptionKind.TYPE_PREFIX,
            agent_type=normalized_agent_type,
            topic_pattern=topic_prefix,
            delivery_mode=delivery_mode,
        )

    def is_match(self, topic_id: TopicId) -> bool:
        """Return whether this subscription matches a topic."""
        if self.kind == SubscriptionKind.TYPE_EXACT:
            return topic_id.type == self.topic_pattern
        if self.kind == SubscriptionKind.TYPE_PREFIX:
            return topic_id.type.startswith(self.topic_pattern)
        raise ValueError(f"Unsupported subscription kind: {self.kind}")

    def map_to_agent(self, topic_id: TopicId) -> AgentId:
        """Map a matching topic to an agent id using source-key affinity."""
        return AgentId(type=self.agent_type, key=AgentKey(topic_id.source))


@dataclass(frozen=True, slots=True)
class PublishRoute:
    """Resolved publish route containing recipient and delivery behavior."""

    subscription_id: str
    """Subscription id that produced this route."""

    recipient: AgentId
    """Target agent id for this route."""

    delivery_mode: DeliveryMode
    """Delivery lifecycle mode for this routed recipient."""
