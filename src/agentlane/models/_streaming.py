"""Shared streaming event primitives for model clients."""

import enum
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Literal, cast

from pydantic import BaseModel
from strenum import LowercaseStrEnum

from ._types import ModelResponse, ToolCall

type SteeringInput = str | list[dict[str, Any]]
"""Native user input submitted to an active response."""


class SteeringStatus(LowercaseStrEnum):
    """Local steering receipt states; acceptance precedes application."""

    SUBMITTED = "submitted"
    """Marks a steering submission awaiting provider acceptance or failure."""

    ACCEPTED = "accepted"
    """Marks provider-accepted steering that has not yet been committed to history."""

    APPLIED = "applied"
    """Marks steering committed to history after a matching successor starts."""

    FAILED = "failed"
    """Marks steering rejected by the provider with an attached error."""

    UNKNOWN = "unknown"
    """Marks steering whose send outcome cannot be determined safely."""


class ModelContinuation(LowercaseStrEnum):
    """How a completed model turn continues."""

    AUTOMATIC = "automatic"
    """Signals that the provider has already started the next response."""

    EXPLICIT = "explicit"
    """Signals that required tool results must be submitted to continue the response."""


@dataclass(slots=True)
class SteeringReceipt:
    """Live steering outcome; acceptance is not a committed history change."""

    input: SteeringInput
    previous_response_id: str
    status: Literal["submitted", "accepted", "applied", "failed", "unknown"] = (
        SteeringStatus.SUBMITTED.value
    )
    id: str | None = None
    response_id: str | None = None
    error: object | None = None


class ModelStreamEventKind(enum.StrEnum):
    """Normalized event kinds emitted by streaming model clients."""

    PROVIDER = "provider"
    TEXT_DELTA = "text_delta"
    TOOL_CALL_ARGUMENTS_DELTA = "tool_call_arguments_delta"
    REASONING = "reasoning"
    COMPLETED = "completed"
    ERROR = "error"


def _serialize_stream_value(value: object) -> object:
    """Convert a stream payload into a trace-safe value."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        # Covers pydantic v2 models, which includes the OpenAI SDK's BaseModel.
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(cast(Any, value))
    if isinstance(value, dict):
        value_dict = cast(dict[object, object], value)
        return {
            str(key): _serialize_stream_value(item) for key, item in value_dict.items()
        }
    if isinstance(value, list):
        value_list = cast(list[object], value)
        return [_serialize_stream_value(item) for item in value_list]
    if isinstance(value, tuple):
        value_tuple = cast(tuple[object, ...], value)
        return [_serialize_stream_value(item) for item in value_tuple]
    return str(value)


@dataclass(slots=True)
class ModelStreamEvent:
    """One normalized stream event with preserved provider detail."""

    kind: ModelStreamEventKind
    raw: object | None = None
    provider_event_type: str | None = None
    item_index: int | None = None
    item_type: str | None = None
    text: str | None = None
    tool_call_id: str | None = None
    tool_call_index: int | None = None
    arguments_delta: str | None = None
    reasoning: object | None = None
    reasoning_signature: str | None = None
    response: ModelResponse | None = None
    error: Exception | None = None
    continuation: Literal["automatic", "explicit"] | None = None
    async_tool_call: ToolCall | None = None
    required_tool_call_ids: tuple[str, ...] = ()
    input_messages: list[dict[str, Any]] | None = None
    steering: list[SteeringReceipt] | None = None

    def to_trace_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        """Export one event into a trace-safe dictionary.

        The raw provider payload is omitted by default because the normalized
        fields already carry the semantic content, and retaining raw chunks for
        every event can balloon trace memory on long streams. Pass
        ``include_raw=True`` when you deliberately need the provider-native
        payload persisted in the trace.
        """
        payload: dict[str, Any] = {
            "kind": self.kind.value,
        }
        if self.provider_event_type is not None:
            payload["provider_event_type"] = self.provider_event_type
        if self.item_index is not None:
            payload["item_index"] = self.item_index
        if self.item_type is not None:
            payload["item_type"] = self.item_type
        if self.text is not None:
            payload["text"] = self.text
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_call_index is not None:
            payload["tool_call_index"] = self.tool_call_index
        if self.arguments_delta is not None:
            payload["arguments_delta"] = self.arguments_delta
        if self.reasoning is not None:
            payload["reasoning"] = _serialize_stream_value(self.reasoning)
        if self.reasoning_signature is not None:
            payload["reasoning_signature"] = self.reasoning_signature
        if self.response is not None:
            payload["response"] = self.response.model_dump(mode="json")
        if self.error is not None:
            payload["error"] = str(self.error)

        # Include lifecycle metadata needed to follow steering and async tool delivery.
        if self.continuation:
            payload["continuation"] = self.continuation
        if self.async_tool_call is not None:
            payload["async_tool_call"] = self.async_tool_call.model_dump(mode="json")
        if self.required_tool_call_ids:
            payload["required_tool_call_ids"] = list(self.required_tool_call_ids)
        if self.input_messages is not None:
            payload["input_messages"] = self.input_messages
        if self.steering is not None:
            payload["steering"] = [asdict(receipt) for receipt in self.steering]

        # Include the native payload only when the caller requests it.
        if include_raw and self.raw is not None:
            payload["raw"] = _serialize_stream_value(self.raw)

        return payload
