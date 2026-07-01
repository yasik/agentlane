"""Errors for the agent-definition loader."""


class AgentFileError(ValueError):
    """Raised when a named agent markdown file cannot be loaded.

    Subclasses `ValueError` so callers that already guard descriptor
    construction with `except ValueError` keep working unchanged.
    """
