"""Versioned JSON snapshots for committed harness agent state."""

from datetime import UTC, datetime
from typing import Literal, Self, cast

from pydantic import (
    AliasChoices,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    NonNegativeInt,
    field_serializer,
    field_validator,
)

from agentlane.messaging import AgentId
from agentlane.models import ModelResponse, render_instruction_text

from ._render import render_request_messages
from ._run import RunHistoryItem, RunState, ShimState, copy_run_state


class _RunStatePayload(BaseModel):
    """JSON representation of one committed harness run state."""

    model_config = ConfigDict(extra="ignore")

    instructions: str | None
    history: list[dict[str, JsonValue]]
    responses: list[ModelResponse]
    shim_state: dict[str, JsonValue]
    turn_count: NonNegativeInt

    @field_validator("history")
    @classmethod
    def _require_message_roles(
        cls,
        history: list[dict[str, JsonValue]],
    ) -> list[dict[str, JsonValue]]:
        """Reject malformed canonical messages."""
        for index, message in enumerate(history):
            if not isinstance(message.get("role"), str):
                raise ValueError(f"history[{index}].role must be a string")
        return history


class AgentSnapshot(BaseModel):
    """Portable value containing one agent's last committed harness state.

    The snapshot stores canonical model-ready history rather than live prompt
    templates. Callers can serialize `to_json()` with the standard library and
    restore it with `from_json()` in another process.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore",
        frozen=True,
    )

    schema_version: Literal[1] = 1
    """JSON schema version used by this snapshot."""

    agent_id: AgentId
    """Stable logical address that owns the state."""

    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    """UTC timestamp for when the snapshot value was created."""

    run_state: RunState = Field(
        validation_alias=AliasChoices("run_state", "state"),
        serialization_alias="state",
    )
    """Isolated committed state captured for the agent."""

    @field_validator("agent_id", mode="before")
    @classmethod
    def _decode_agent_id(cls, value: object) -> AgentId:
        """Accept either the value object or its JSON representation."""
        if isinstance(value, AgentId):
            return value
        return AgentId.from_json(value)

    @field_serializer("agent_id")
    def _encode_agent_id(self, agent_id: AgentId) -> dict[str, object]:
        """Encode the logical address with its existing wire contract."""
        return agent_id.to_json()

    @field_validator("created_at")
    @classmethod
    def _normalize_created_at(cls, created_at: datetime) -> datetime:
        """Normalize accepted aware timestamps to UTC."""
        return created_at.astimezone(UTC)

    @field_validator("run_state", mode="before")
    @classmethod
    def _decode_run_state(cls, value: object) -> RunState:
        """Decode or isolate one committed run state."""
        if isinstance(value, RunState):
            copied_state = copy_run_state(value)
            if copied_state is None:
                raise AssertionError("Run state copy unexpectedly returned None.")
            return copied_state

        payload = _RunStatePayload.model_validate(value)
        return RunState(
            instructions=payload.instructions,
            history=cast(list[RunHistoryItem], payload.history),
            responses=payload.responses,
            shim_state=ShimState(context=payload.shim_state),
            turn_count=payload.turn_count,
        )

    @field_serializer("run_state")
    def _encode_run_state(self, run_state: RunState) -> _RunStatePayload:
        """Render live state into the canonical JSON payload."""
        instructions = run_state.instructions
        if instructions is not None and not isinstance(instructions, str):
            instructions = render_instruction_text(instructions)

        return _RunStatePayload.model_validate(
            {
                "instructions": instructions,
                "history": render_request_messages(None, run_state.history),
                "responses": run_state.responses,
                "shim_state": dict(run_state.shim_state),
                "turn_count": run_state.turn_count,
            }
        )

    def to_json(self) -> dict[str, object]:
        """Return the canonical JSON-safe snapshot mapping."""
        return cast(dict[str, object], self.model_dump(mode="json", by_alias=True))

    @classmethod
    def from_json(cls, data: object) -> Self:
        """Decode one version-1 snapshot mapping."""
        return cls.model_validate(data)
