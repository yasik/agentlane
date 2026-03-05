"""Subscription models used by publish routing."""

import enum
from dataclasses import dataclass, field
from uuid import uuid4

from ._identity import AgentId, AgentKey, AgentType, TopicId


class SubscriptionKind(enum.StrEnum):
    """Supported subscription matching strategies."""

    TYPE_EXACT = "type_exact"
    TYPE_PREFIX = "type_prefix"


@dataclass(slots=True)
class Subscription:
    """Route publish events to agent types based on topic patterns."""

    kind: SubscriptionKind
    """Matching strategy for topic resolution."""

    agent_type: AgentType
    """Recipient agent type for matched topics."""

    topic_pattern: str
    """Topic pattern used by the selected matching strategy."""

    priority: int = 0
    """Priority used for deterministic ordering when needed."""

    metadata: dict[str, str] = field(default_factory=dict)
    """Additional subscription metadata."""

    id: str = field(default_factory=lambda: str(uuid4()))
    """Stable subscription identifier."""

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
