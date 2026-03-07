"""Subscription models used by publish routing.

These primitives define how published topics are matched to agent types and
whether deliveries should favor instance reuse (`stateful`) or per-delivery
isolation (`stateless`).
"""

import enum
from dataclasses import dataclass, field
from uuid import uuid4

from ._identity import AgentId, AgentKey, AgentType, TopicId


class SubscriptionKind(enum.StrEnum):
    """Supported topic matching strategies for subscriptions."""

    TYPE_EXACT = "type_exact"
    """Match only when `topic.type` exactly equals `topic_pattern`."""

    TYPE_PREFIX = "type_prefix"
    """Match when `topic.type` starts with `topic_pattern`."""


class DeliveryMode(enum.StrEnum):
    """Delivery lifecycle mode applied after a subscription matches."""

    STATEFUL = "stateful"
    """Reuse cached agent instance by route-key-derived recipient id."""

    STATELESS = "stateless"
    """Use unique per-delivery recipient ids so instance reuse is not guaranteed."""


@dataclass(slots=True)
class Subscription:
    """Route publish events to agent types based on topic patterns.

    Typical usage is through convenience constructors:

    ```python
    stateful = Subscription.exact(
        topic_type="jobs.created",
        agent_type="worker",
        delivery_mode=DeliveryMode.STATEFUL,
    )

    stateless = Subscription.prefix(
        topic_prefix="jobs.",
        agent_type="worker",
        delivery_mode=DeliveryMode.STATELESS,
    )
    ```
    """

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
        """Create a subscription that matches one exact topic type.

        Args:
            topic_type: Topic type string requiring exact equality.
            agent_type: Recipient agent type (string or typed wrapper).
            delivery_mode: Delivery lifecycle behavior for matched routes.

        Returns:
            Subscription: Exact-match subscription instance.
        """
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
        """Create a subscription that matches topic types by prefix.

        Args:
            topic_prefix: Prefix used for `topic.type.startswith(...)` matching.
            agent_type: Recipient agent type (string or typed wrapper).
            delivery_mode: Delivery lifecycle behavior for matched routes.

        Returns:
            Subscription: Prefix-match subscription instance.
        """
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
        """Return whether this subscription matches a topic identifier.

        Args:
            topic_id: Topic identifier to evaluate against this subscription.

        Returns:
            bool: True when subscription matches the topic.

        Raises:
            ValueError: If subscription kind is unsupported.
        """
        if self.kind == SubscriptionKind.TYPE_EXACT:
            return topic_id.type == self.topic_pattern
        if self.kind == SubscriptionKind.TYPE_PREFIX:
            return topic_id.type.startswith(self.topic_pattern)
        raise ValueError(f"Unsupported subscription kind: {self.kind}")

    def map_to_agent(self, topic_id: TopicId) -> AgentId:
        """Map a matching topic to a recipient using route-key affinity.

        The mapped key is `topic_id.source` (publicly treated as `route_key`).

        Args:
            topic_id: Matching topic id used to derive recipient key.

        Returns:
            AgentId: Recipient id resolved from subscription + topic route key.
        """
        return AgentId(type=self.agent_type, key=AgentKey(topic_id.source))


@dataclass(frozen=True, slots=True)
class PublishRoute:
    """Resolved publish route produced by routing policy evaluation.

    A route is one publish target after subscription matching and dedup rules.
    Runtime uses this object to derive final task recipient identity and apply
    delivery lifecycle behavior.
    """

    subscription_id: str
    """Subscription id that produced this route."""

    recipient: AgentId
    """Target agent id for this route."""

    delivery_mode: DeliveryMode
    """Delivery lifecycle mode for this routed recipient."""
