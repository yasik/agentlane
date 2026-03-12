"""Identity primitives for agents, topics, and messages."""

from dataclasses import dataclass
from typing import Self, cast
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class AgentType:
    """Logical agent type identifier."""

    value: str
    """Human-readable type name (globally unique per runtime scope)."""

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("AgentType value must be non-empty.")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class AgentKey:
    """Logical agent instance key."""

    value: str
    """Instance key used to identify stateful agent instances."""

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("AgentKey value must be non-empty.")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class AgentId:
    """Full agent identity."""

    type: AgentType
    """Logical agent type."""

    key: AgentKey
    """Logical instance key."""

    @classmethod
    def from_values(cls, type_value: str, key_value: str) -> Self:
        """Construct an AgentId from plain string values.

        Args:
            type_value: Logical agent type name.
            key_value: Logical agent instance key.

        Returns:
            Self: Normalized agent id.
        """
        return cls(type=AgentType(type_value), key=AgentKey(key_value))

    @classmethod
    def from_json(cls, data: object) -> Self:
        """Construct an AgentId from a JSON-safe mapping."""
        if not isinstance(data, dict):
            raise TypeError("Expected JSON object for agent id.")
        mapping = cast(dict[str, object], data)
        type_value = mapping.get("type")
        key_value = mapping.get("key")
        if not isinstance(type_value, str):
            raise TypeError("Expected string for agent id type.")
        if not isinstance(key_value, str):
            raise TypeError("Expected string for agent id key.")
        return cls.from_values(type_value=type_value, key_value=key_value)

    def to_json(self) -> dict[str, object]:
        """Return a JSON-safe mapping for this agent id."""
        return {
            "type": self.type.value,
            "key": self.key.value,
        }

    def __str__(self) -> str:
        return f"{self.type.value}:{self.key.value}"


@dataclass(frozen=True, slots=True)
class TopicId:
    """Topic identifier used for publish routing."""

    type: str
    """Topic type used for subscription matching."""

    source: str
    """Source dimension used for source-key affinity routing."""

    def __post_init__(self) -> None:
        if not self.type:
            raise ValueError("TopicId type must be non-empty.")
        if not self.source:
            raise ValueError("TopicId source must be non-empty.")

    @classmethod
    def from_values(cls, type_value: str, route_key: str) -> Self:
        """Construct a TopicId from type and route key values.

        Args:
            type_value: Topic type used by subscription matching.
            route_key: Route-key value mapped onto the `source` dimension.

        Returns:
            Self: Normalized topic id.
        """
        return cls(type=type_value, source=route_key)

    @classmethod
    def from_json(cls, data: object) -> Self:
        """Construct a TopicId from a JSON-safe mapping."""
        if not isinstance(data, dict):
            raise TypeError("Expected JSON object for topic id.")
        mapping = cast(dict[str, object], data)
        type_value = mapping.get("type")
        source_value = mapping.get("source")
        if not isinstance(type_value, str):
            raise TypeError("Expected string for topic id type.")
        if not isinstance(source_value, str):
            raise TypeError("Expected string for topic id source.")
        return cls(type=type_value, source=source_value)

    def to_json(self) -> dict[str, object]:
        """Return a JSON-safe mapping for this topic id."""
        return {
            "type": self.type,
            "source": self.source,
        }

    @property
    def route_key(self) -> str:
        """Return publisher-provided route key alias for source.

        Returns:
            str: Route key value copied from `source`.
        """
        return self.source


class Topics:
    """Convenience constructors for topic identifiers."""

    @staticmethod
    def id(type_value: str, route_key: str) -> TopicId:
        """Build a topic id from type and route key values.

        Args:
            type_value: Topic type used by subscription matching.
            route_key: Route-key value for source-affinity routing.

        Returns:
            TopicId: New topic identifier instance.
        """
        return TopicId.from_values(type_value=type_value, route_key=route_key)


@dataclass(frozen=True, slots=True)
class MessageId:
    """Unique envelope identifier."""

    value: str
    """Globally unique message identifier."""

    @classmethod
    def new(cls) -> Self:
        """Create a new message identifier.

        Args:
            cls: MessageId class.

        Returns:
            Self: Newly generated message id.
        """
        return cls(value=str(uuid4()))


@dataclass(frozen=True, slots=True)
class CorrelationId:
    """Causal chain identifier."""

    value: str
    """Identifier tying a causal message chain together."""

    @classmethod
    def new(cls) -> Self:
        """Create a new correlation identifier.

        Args:
            cls: CorrelationId class.

        Returns:
            Self: Newly generated correlation id.
        """
        return cls(value=str(uuid4()))


@dataclass(frozen=True, slots=True)
class IdempotencyKey:
    """Deduplication key for at-least-once delivery paths."""

    value: str
    """Caller-provided key used to deduplicate retried deliveries."""

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("IdempotencyKey value must be non-empty.")

    @classmethod
    def new(cls) -> Self:
        """Create a new idempotency key.

        Args:
            cls: IdempotencyKey class.

        Returns:
            Self: Newly generated idempotency key.
        """
        return cls(value=str(uuid4()))
