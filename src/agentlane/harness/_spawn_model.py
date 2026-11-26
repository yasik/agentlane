"""Application policy for the model assigned to each new child."""

from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from agentlane.messaging import AgentId
from agentlane.models import Model, ModelResponse

from ._handoff import DelegationKind
from ._run import RunMessageContent


@dataclass(frozen=True, slots=True)
class SpawnModelConfig:
    """A model client and its complete request overrides for one child."""

    model: Model[ModelResponse]
    """Configured client retained by reference for the child's lifetime."""

    model_args: Mapping[str, object] | None
    """Complete request overrides; None or an empty mapping clears inherited args.

    The harness snapshots mutable argument data when it accepts the assignment.
    Child shims can still adjust the arguments for an individual request.
    """


@dataclass(frozen=True, slots=True)
class SpawnModelContext:
    """Stable identity, validated payload, and resolved child defaults."""

    kind: DelegationKind
    """TOOL returns a helper result; HANDOFF transfers the conversation."""

    parent_id: AgentId
    child_id: AgentId
    """Fresh identity reused for selection, delivery, and tracing."""

    parent_name: str
    child_name: str
    tool_name: str
    tool_call_id: str
    task: RunMessageContent
    """Detached validated task payload; editing it does not change child input.

    Predefined tools provide their input model. Generic helpers and handoffs
    provide the task text, including the default handoff message when omitted.
    """

    spawn_parameters: BaseModel | None
    """Validated application preferences, or None when suggestions are disabled."""

    baseline: SpawnModelConfig
    """Resolved child defaults with detached argument data and the original client."""


@dataclass(frozen=True, slots=True)
class SpawnModelSelection:
    """Complete assignment with optional safe audit labels."""

    config: SpawnModelConfig
    """Replace the entire model/arguments pair; arguments are never merged."""

    label: str | None = None
    """Optional application label safe to include in traces."""

    fallback_reason: str | None = None
    """Optional safe reason code for an explicit application fallback."""


class SpawnModelRejected(Exception):
    """Expected policy refusal with a reason safe for tools and traces."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class SpawnModelProvider(Protocol):
    """Own the application schema and policy for new child assignments."""

    @property
    def parameters_model(self) -> type[BaseModel] | None:
        """Return the suggestion schema, or None for application-only routing.

        Runner snapshots this schema at construction time.
        """
        ...

    def resolve(
        self, context: SpawnModelContext
    ) -> SpawnModelSelection | None | Awaitable[SpawnModelSelection | None]:
        """Select once per new child, before opening its model resources.

        Return None to retain the baseline or raise SpawnModelRejected for an
        expected refusal. Errors and cancellation propagate. Later messages to
        the same live child reuse its assignment without another policy call.
        """
        ...
