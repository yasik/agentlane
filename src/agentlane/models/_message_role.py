"""Shared role values for model conversation messages."""

from strenum import LowercaseStrEnum


class MessageRole(LowercaseStrEnum):
    """Message roles used by the harness and model adapters.

    Use `.value` when constructing provider payloads to retain plain strings.
    """

    SYSTEM = "system"
    """Marks system instructions in the shared conversation format."""

    DEVELOPER = "developer"
    """Marks developer instructions for adapters that support this role."""

    USER = "user"
    """Marks user input and model-facing continuation messages."""

    ASSISTANT = "assistant"
    """Marks model output, including text and tool-call requests."""

    TOOL = "tool"
    """Marks the result associated with an earlier model tool call."""
