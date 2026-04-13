"""Shared streaming event primitives for model clients."""

import enum
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, cast

from openai import BaseModel as OpenAIBaseModel
from pydantic import BaseModel

from ._types import ModelResponse


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
        return value.model_dump(mode="json")
    if isinstance(value, OpenAIBaseModel):
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

    def to_trace_dict(self) -> dict[str, Any]:
        """Export one event into a trace-safe dictionary."""
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
        if self.raw is not None:
            payload["raw"] = _serialize_stream_value(self.raw)
        return payload
