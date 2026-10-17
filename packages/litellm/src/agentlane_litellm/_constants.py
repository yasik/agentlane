"""Adapter validation messages."""

NATIVE_RESPONSES_HISTORY_ERROR = (
    "Native Responses history requires ResponsesClient. "
    "Convert history explicitly before using LiteLLM."
)
"""Used to reject native Responses history that LiteLLM cannot replay."""
