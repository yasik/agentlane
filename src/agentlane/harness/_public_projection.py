"""Public JSON projection that excludes internal context messages."""

from pydantic import JsonValue

from agentlane.models import is_internal_message


def project_messages(value: JsonValue) -> JsonValue:
    """Copy a JSON payload without internal messages at any nesting level.

    Event and result serialization calls this before publication.
    Live state and authoritative snapshots remain complete.
    Internal list entries are omitted. Internal mapping values and standalone
    messages become null. Ordinary messages remain visible by default.
    """
    if is_internal_message(value):
        return None

    if isinstance(value, dict):
        return {key: project_messages(item) for key, item in value.items()}

    if isinstance(value, list):
        return [
            project_messages(item) for item in value if not is_internal_message(item)
        ]

    return value
