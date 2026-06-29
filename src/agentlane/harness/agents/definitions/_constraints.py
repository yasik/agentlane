"""Constraints and the error type for the agent-definition loader."""

AGENT_MAX_DESCRIPTION_LENGTH = 1024
"""Maximum agent description length limit.

An over-length description is truncated to this many characters. The description
is a short delegation hint, so truncation is safe; the instruction body uses a
pointer instead (see `AGENT_MAX_INSTRUCTIONS_LINES`).
"""

AGENT_MAX_INSTRUCTIONS_LINES = 1000
"""Maximum instruction-body lines kept inline before truncation.

A longer body is truncated to this many lines with a pointer back to the source
file, so the model can read the rest with its own tools instead of the whole
agent being dropped. Internal guard, not a spec value.
"""

AGENT_MAX_SUBAGENT_DEPTH = 4
"""Maximum sub-agent nesting depth resolved at load time.

Matches the runner's default `agent_max_depth` so a tree that loads cleanly
cannot exceed the runtime delegation cap.
"""


class AgentFileError(ValueError):
    """Raised when a named agent markdown file cannot be loaded.

    Subclasses `ValueError` so callers that already guard descriptor
    construction with `except ValueError` keep working unchanged.
    """
