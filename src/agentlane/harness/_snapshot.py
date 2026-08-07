"""Versioned JSON snapshots for committed harness agent state."""

from datetime import UTC, datetime
from typing import Literal, Self, cast

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    JsonValue,
    NonNegativeInt,
    field_serializer,
    field_validator,
)

from agentlane.messaging import AgentId
from agentlane.models import ModelResponse, render_instruction_text

from ._render import render_request_messages
from ._run import RunHistoryItem, RunState, ShimState


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
    """Portable JSON value containing one agent's committed harness state."""

    model_config = ConfigDict(extra="ignore")

    schema_version: Literal[1]
    """JSON schema version used by this snapshot."""

    agent_id: AgentId
    """Stable logical address that owns the state."""

    created_at: AwareDatetime
    """UTC timestamp for when the snapshot value was created."""

    state: _RunStatePayload
    """Canonical model-ready state captured for the agent."""

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

    @classmethod
    def capture(cls, *, agent_id: AgentId, run_state: RunState) -> Self:
        """Capture one live run state as a canonical snapshot value."""
        instructions = run_state.instructions
        if instructions is not None and not isinstance(instructions, str):
            instructions = render_instruction_text(instructions)

        return cls(
            schema_version=1,
            agent_id=agent_id,
            created_at=datetime.now(UTC),
            state=_RunStatePayload.model_validate(
                {
                    "instructions": instructions,
                    "history": render_request_messages(None, run_state.history),
                    "responses": run_state.responses,
                    "shim_state": run_state.shim_state,
                    "turn_count": run_state.turn_count,
                }
            ),
        )

    def to_run_state(self) -> RunState:
        """Return an isolated live run state restored from this snapshot."""
        state = self.state.model_copy(deep=True)
        return RunState(
            instructions=state.instructions,
            history=cast(list[RunHistoryItem], state.history),
            responses=state.responses,
            shim_state=ShimState(context=state.shim_state),
            turn_count=state.turn_count,
        )

    def to_json(self) -> dict[str, object]:
        """Return the canonical JSON-safe snapshot mapping."""
        return cast(dict[str, object], self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, data: object) -> Self:
        """Decode one version-1 snapshot mapping."""
        return cls.model_validate(data)
