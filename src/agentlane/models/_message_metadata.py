"""Runtime metadata on canonical messages, separate from provider fields."""

from collections.abc import Mapping, Sequence
from typing import cast

from ._interface import MessageDict

_METADATA_KEY = "_agentlane"
"""Reserved message field for JSON-safe AgentLane runtime metadata."""


def internal_message(
    message: MessageDict, *, origin: str, ephemeral: bool = False
) -> MessageDict:
    """Copy a message and mark it as internal context from the given origin.

    The model receives the message contents. Public projections omit the message.
    The origin identifies its runtime owner and is not a provider message name.

    Args:
        message: Message to copy. The function does not change it.
        origin: Nonempty name of the runtime owner of the context.
        ephemeral: When `True`, the message is run-scoped context. The model
            receives it on each model call of the run that contains it. The
            harness removes it from committed state, run results, and
            snapshots when the run ends.

    Returns:
        A copy of the message with the runtime metadata.

    Raises:
        ValueError: If `origin` is empty.
    """
    if not origin:
        raise ValueError("Internal messages require a nonempty origin.")

    metadata: dict[str, object] = {"visibility": "internal", "origin": origin}

    # Add the lifetime flag only when it is set, so persistent context keeps
    # the same metadata in existing snapshots.
    if ephemeral:
        metadata["ephemeral"] = True

    return {**message, _METADATA_KEY: metadata}


def is_internal_message(value: object, *, origin: str | None = None) -> bool:
    """Identify internal messages, optionally restricted to one runtime owner."""
    return _internal_metadata(value, origin=origin) is not None


def is_ephemeral_message(value: object, *, origin: str | None = None) -> bool:
    """Identify run-scoped internal messages, optionally from one runtime owner.

    An ephemeral message is always internal. The metadata flag must be exactly
    `True`; other values do not make a message ephemeral.
    """
    fields = _internal_metadata(value, origin=origin)
    return fields is not None and fields.get("ephemeral") is True


def provider_messages(messages: Sequence[MessageDict]) -> list[MessageDict]:
    """Copy model input without runtime metadata, preserving contents and order.

    Call this at provider boundaries, after hooks that need message visibility.
    Do not use it to capture authoritative history or snapshots.
    """
    return [
        {key: value for key, value in message.items() if key != _METADATA_KEY}
        for message in messages
    ]


def _internal_metadata(
    value: object, *, origin: str | None
) -> Mapping[str, object] | None:
    """Return the metadata fields of an internal message, or `None`."""
    if not isinstance(value, Mapping):
        return None

    message = cast(Mapping[str, object], value)
    metadata = message.get(_METADATA_KEY)
    if not isinstance(metadata, Mapping):
        return None

    fields = cast(Mapping[str, object], metadata)
    if fields.get("visibility") != "internal":
        return None
    if origin is not None and fields.get("origin") != origin:
        return None

    return fields
