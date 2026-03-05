"""Identity primitives for agents, topics, and messages."""

from dataclasses import dataclass
from typing import Self
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
        """Construct an AgentId from plain string values."""
        return cls(type=AgentType(type_value), key=AgentKey(key_value))

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


@dataclass(frozen=True, slots=True)
class MessageId:
    """Unique envelope identifier."""

    value: str
    """Globally unique message identifier."""

    @classmethod
    def new(cls) -> Self:
        """Create a new message identifier."""
        return cls(value=str(uuid4()))


@dataclass(frozen=True, slots=True)
class CorrelationId:
    """Causal chain identifier."""

    value: str
    """Identifier tying a causal message chain together."""

    @classmethod
    def new(cls) -> Self:
        """Create a new correlation identifier."""
        return cls(value=str(uuid4()))
