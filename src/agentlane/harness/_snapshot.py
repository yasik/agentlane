"""Versioned JSON snapshots for committed harness agent state."""

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import ClassVar, Self, cast

from agentlane.messaging import AgentId
from agentlane.models import MessageDict, ModelResponse, render_instruction_text

from ._render import render_request_messages
from ._run import RunHistoryItem, RunState, ShimState, copy_run_state

SNAPSHOT_SCHEMA_VERSION = 1
"""Current `AgentSnapshot` JSON schema version."""


@dataclass(frozen=True, slots=True)
class AgentSnapshot:
    """Portable value containing one agent's last committed harness state.

    The snapshot stores canonical model-ready history rather than live prompt
    templates. Callers can serialize `to_json()` with the standard library and
    restore it with `from_json()` in another process.
    """

    agent_id: AgentId
    """Stable logical address that owns the state."""

    run_state: RunState
    """Isolated committed state captured for the agent."""

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """UTC timestamp for when the snapshot value was created."""

    schema_version: ClassVar[int] = SNAPSHOT_SCHEMA_VERSION
    """JSON schema version used by all snapshots produced by this class."""

    def __post_init__(self) -> None:
        """Validate the envelope and isolate caller-owned state."""
        if self.created_at.tzinfo is None:
            raise ValueError("Agent snapshot created_at must include a timezone.")

        copied_state = copy_run_state(self.run_state)
        if copied_state is None:
            raise AssertionError(
                "Agent snapshot state copy unexpectedly returned None."
            )
        object.__setattr__(self, "run_state", copied_state)
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))

    def to_json(self) -> dict[str, object]:
        """Return the canonical JSON-safe snapshot mapping.

        Raises:
            TypeError: If persisted state contains a value outside the JSON
                contract.
            ValueError: If a numeric value is not valid JSON.
        """
        state = self.run_state
        instructions = state.instructions
        if instructions is not None and not isinstance(instructions, str):
            instructions = render_instruction_text(instructions)

        history = render_request_messages(None, state.history)
        responses = [response.model_dump(mode="json") for response in state.responses]
        shim_state = {
            key: _json_value(value, path=f"shim_state.{key}")
            for key, value in state.shim_state.items()
        }

        return {
            "schema_version": self.schema_version,
            "agent_id": self.agent_id.to_json(),
            "created_at": self.created_at.isoformat(),
            "state": {
                "instructions": instructions,
                "history": _json_value(history, path="history"),
                "responses": _json_value(responses, path="responses"),
                "shim_state": shim_state,
                "turn_count": state.turn_count,
            },
        }

    @classmethod
    def from_json(cls, data: object) -> Self:
        """Decode one version-1 snapshot mapping.

        Unknown fields are ignored so additive metadata does not break older
        readers. Unsupported schema versions fail instead of guessing.

        Args:
            data: Untrusted JSON value to decode.

        Returns:
            Decoded snapshot with an isolated `RunState`.

        Raises:
            TypeError: If a required field has the wrong JSON type.
            ValueError: If the schema version or timestamp is invalid.
        """
        payload = _object(_json_value(data, path="snapshot"), path="snapshot")
        schema_version = _integer(payload.get("schema_version"), path="schema_version")
        if schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported agent snapshot schema version {schema_version}."
            )

        created_at_value = _string(payload.get("created_at"), path="created_at")
        try:
            created_at = datetime.fromisoformat(created_at_value)
        except ValueError as exc:
            raise ValueError("Agent snapshot created_at must be ISO 8601.") from exc
        if created_at.tzinfo is None:
            raise ValueError("Agent snapshot created_at must include a timezone.")

        state_payload = _object(payload.get("state"), path="state")
        instructions = state_payload.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise TypeError("Expected string or null at state.instructions.")

        raw_history = _array(state_payload.get("history"), path="state.history")
        history: list[RunHistoryItem] = [
            _message(item, path=f"state.history[{index}]")
            for index, item in enumerate(raw_history)
        ]
        raw_responses = _array(state_payload.get("responses"), path="state.responses")
        responses = [
            ModelResponse.model_validate(
                _object(item, path=f"state.responses[{index}]")
            )
            for index, item in enumerate(raw_responses)
        ]
        raw_shim_state = _object(
            state_payload.get("shim_state"), path="state.shim_state"
        )
        shim_state = ShimState(
            context={
                key: _json_value(value, path=f"state.shim_state.{key}")
                for key, value in raw_shim_state.items()
            }
        )
        turn_count = _non_negative_integer(
            state_payload.get("turn_count"), path="state.turn_count"
        )

        return cls(
            agent_id=AgentId.from_json(payload.get("agent_id")),
            run_state=RunState(
                instructions=instructions,
                history=history,
                responses=responses,
                shim_state=shim_state,
                turn_count=turn_count,
            ),
            created_at=created_at,
        )


def _json_value(value: object, *, path: str) -> object:
    """Return an isolated JSON value or raise with its state path."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Expected finite JSON number at {path}.")
        return value
    if isinstance(value, list):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(cast(list[object], value))
        ]
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, item in cast(dict[object, object], value).items():
            if not isinstance(key, str):
                raise TypeError(f"Expected string JSON object key at {path}.")
            result[key] = _json_value(item, path=f"{path}.{key}")
        return result
    raise TypeError(f"Expected JSON-safe value at {path}; got {type(value).__name__}.")


def _object(value: object, *, path: str) -> dict[str, object]:
    """Return one JSON object with string keys."""
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object at {path}.")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise TypeError(f"Expected string JSON object keys at {path}.")
    return cast(dict[str, object], mapping)


def _array(value: object, *, path: str) -> list[object]:
    """Return one JSON array."""
    if not isinstance(value, list):
        raise TypeError(f"Expected JSON array at {path}.")
    return cast(list[object], value)


def _message(value: object, *, path: str) -> MessageDict:
    """Return one canonical model message."""
    message = _object(value, path=path)
    if not isinstance(message.get("role"), str):
        raise TypeError(f"Expected string at {path}.role.")
    return cast(MessageDict, message)


def _string(value: object, *, path: str) -> str:
    """Return one JSON string."""
    if not isinstance(value, str):
        raise TypeError(f"Expected string at {path}.")
    return value


def _integer(value: object, *, path: str) -> int:
    """Return one JSON integer, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected integer at {path}.")
    return value


def _non_negative_integer(value: object, *, path: str) -> int:
    """Return one non-negative JSON integer."""
    result = _integer(value, path=path)
    if result < 0:
        raise ValueError(f"Expected non-negative integer at {path}.")
    return result
