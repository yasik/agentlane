"""Application policy for the model assigned to each new child."""

from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel

from agentlane.messaging import AgentId
from agentlane.models import Model, ModelResponse

from ._run import RunMessageContent

type SpawnKind = Literal[
    "agent_tool", "default_agent_tool", "handoff", "default_handoff"
]


@dataclass(frozen=True, slots=True)
class SpawnModelConfig:
    """A model client and its complete request overrides for one child."""

    model: Model[ModelResponse]
    model_args: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class SpawnModelContext:
    """Stable identity, validated payload, and resolved child defaults."""

    kind: SpawnKind
    parent_id: AgentId
    child_id: AgentId
    parent_name: str
    child_name: str
    tool_name: str
    tool_call_id: str
    task: RunMessageContent
    spawn_parameters: BaseModel | None
    baseline: SpawnModelConfig


@dataclass(frozen=True, slots=True)
class SpawnModelSelection:
    """Complete assignment with optional safe audit labels."""

    config: SpawnModelConfig
    label: str | None = None
    fallback_reason: str | None = None


class SpawnModelRejected(Exception):
    """Expected policy refusal with a reason safe for tools and traces."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class SpawnModelProvider(Protocol):
    """Own the application schema and policy for new child assignments."""

    @property
    def parameters_model(self) -> type[BaseModel] | None:
        """Return the suggestion schema, or None for application-only routing."""
        ...

    def resolve(
        self, context: SpawnModelContext
    ) -> SpawnModelSelection | None | Awaitable[SpawnModelSelection | None]:
        """Select a complete assignment, retain defaults, or refuse the spawn."""
        ...
