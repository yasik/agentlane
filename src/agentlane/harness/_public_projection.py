"""Public JSON projection that excludes internal context messages."""

from pydantic import JsonValue

from agentlane.models import is_internal_message


def public_message_projection(value: JsonValue) -> JsonValue:
    """Copy a JSON payload without internal messages at any nesting level.

    Use this on serialized events, results, or conversation records before
    publication. Native records and authoritative snapshots remain complete.
    Internal list entries and mapping values are omitted; a standalone internal
    message becomes null. Ordinary messages remain visible by default.
    """
    if is_internal_message(value):
        return None

    if isinstance(value, dict):
        return {
            key: public_message_projection(item)
            for key, item in value.items()
            if not is_internal_message(item)
        }

    if isinstance(value, list):
        return [
            public_message_projection(item)
            for item in value
            if not is_internal_message(item)
        ]

    return value
