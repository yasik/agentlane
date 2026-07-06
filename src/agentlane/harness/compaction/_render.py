"""Public rendering helpers for compaction implementations."""

from collections.abc import Sequence

from agentlane.models import MessageDict

from .._render import render_request_messages as _render_request_messages
from .._run import RunHistoryItem, RunInstructions


def render_request_messages(
    instructions: RunInstructions,
    history: Sequence[RunHistoryItem],
) -> list[MessageDict]:
    """Render compaction input with the same message builder as the runner."""
    return _render_request_messages(instructions, history)
