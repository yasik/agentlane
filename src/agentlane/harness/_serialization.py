"""Strict inspection conversion shared by run events and final results."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import PurePath
from typing import cast
from uuid import UUID

from pydantic import BaseModel, JsonValue, RootModel

from agentlane.models import PromptSpec, PromptTemplateBase


def serialize_value(value: object) -> JsonValue:
    """Convert a supported value while preserving its complete structure."""
    return _serialize_value(value, set())


def _serialize_value(value: object, active: set[int]) -> JsonValue:
    """Convert event fields recursively, tracking ancestors rather than aliases."""
    if isinstance(value, Enum):
        return _serialize_value(value.value, active)

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite numbers cannot be transported as JSON.")

        return value

    if isinstance(value, (PurePath, UUID)):
        return str(value)

    if isinstance(value, Exception):
        return {"type": type(value).__name__, "message": str(value)}

    identity = id(value)
    if identity in active:
        raise ValueError("Cyclic values cannot be transported as JSON.")

    active.add(identity)

    try:
        return _serialize_compound(value, active)
    finally:
        active.remove(identity)


def _serialize_compound(value: object, active: set[int]) -> JsonValue:
    """Convert containers and framework objects without arbitrary object introspection."""
    # Render executable templates through their public interface.
    # This describes the stored prompt, not necessarily a model request:
    # use RunLLMStartEvent.messages for messages passed to the model client.
    if isinstance(value, PromptSpec):
        prompt = cast(PromptSpec[object], value)
        return _serialize_value(
            {
                "messages": prompt.template.render_messages(prompt.values),
                "values": prompt.values,
                "response_format": prompt.template.response_format(),
            },
            active,
        )

    # Template implementations can be fieldless dataclasses; {} loses the prompt.
    if isinstance(value, PromptTemplateBase):
        raise TypeError("A prompt template must be paired with values in PromptSpec.")

    if isinstance(value, RootModel):
        return _serialize_value(cast(RootModel[object], value).root, active)

    if isinstance(value, BaseModel):
        # Public iteration keeps raw nested values and extras. Recursive model_dump
        # can erase PromptSpec templates before our explicit conversion sees them.
        return _serialize_value(dict(value), active)

    # ShimState is also a dataclass; its public mapping excludes runtime locks.
    if isinstance(value, Mapping):
        payload: dict[str, JsonValue] = {}
        for key, item in cast(Mapping[object, object], value).items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings.")

            payload[key] = _serialize_value(item, active)

        return payload

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _serialize_value(getattr(value, field.name), active)
            for field in fields(value)
        }

    if isinstance(value, (list, tuple)):
        return [
            _serialize_value(item, active) for item in cast(Sequence[object], value)
        ]

    raise TypeError(f"Unsupported event value: {type(value).__qualname__}.")
