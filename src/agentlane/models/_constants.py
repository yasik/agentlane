"""Shared validation feedback for model adapters."""

SCHEMA_NORMALIZATION_ERROR = "schema normalization unexpectedly returned None"
"""Used in errors when structured-output validation cannot resolve its schema."""

ESCAPE_SEQUENCE_RETRY_MESSAGE = (
    "Your previous response contained repeated Unicode "
    "escape sequences and could not be processed. Please "
    "regenerate your response without using repeated combining "
    "diacritical marks or \\uXXXX sequences."
)
"""Sent to the model when repeated escape sequences make its response unusable."""
