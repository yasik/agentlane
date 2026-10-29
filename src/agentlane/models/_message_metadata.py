"""Runtime metadata on canonical messages, separate from provider fields."""

from collections.abc import Mapping, Sequence
from typing import cast

from ._interface import MessageDict

_METADATA_KEY = "_agentlane"
"""Reserved message field for JSON-safe AgentLane runtime metadata."""


def internal_message(message: MessageDict, *, origin: str) -> MessageDict:
    """Copy a message and mark it as internal context from the given origin.

    The model receives the message contents. Public projections omit the message.
    The origin identifies its runtime owner and is not a provider message name.
    """
    if not origin:
        raise ValueError("Internal messages require a nonempty origin.")

    return {**message, _METADATA_KEY: {"visibility": "internal", "origin": origin}}


def is_internal_message(value: object, *, origin: str | None = None) -> bool:
    """Identify internal messages, optionally restricted to one runtime owner."""
    if not isinstance(value, Mapping):
        return False

    message = cast(Mapping[str, object], value)
    if not isinstance(message.get("role"), str):
        return False

    metadata = message.get(_METADATA_KEY)
    if not isinstance(metadata, Mapping):
        return False

    fields = cast(Mapping[str, object], metadata)
    return fields.get("visibility") == "internal" and (
        origin is None or fields.get("origin") == origin
    )


def provider_messages(messages: Sequence[MessageDict]) -> list[MessageDict]:
    """Copy model input without runtime metadata, preserving contents and order.

    Call this at provider boundaries, after hooks that need message visibility.
    Do not use it to capture authoritative history or snapshots.
    """
    return [
        {key: value for key, value in message.items() if key != _METADATA_KEY}
        for message in messages
    ]
