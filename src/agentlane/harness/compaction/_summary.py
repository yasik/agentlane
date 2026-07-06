"""Summary message helpers for harness conversation compaction."""

from jinja2 import Template

from agentlane.models import MessageDict

from .._run import RunHistoryItem
from ._constants import SUMMARY_CLOSE_TAG, SUMMARY_OPEN_TAG
from ._prompt import DEFAULT_SUMMARY_ITEM_TEMPLATE


def render_summary_item(*, bridge: str, summary_text: str) -> MessageDict:
    """Return the canonical user-role summary message.

    Args:
        bridge: Continuation text that tells the next model how to interpret
            the generated summary. Callers may pass a domain-specific bridge.
        summary_text: Model-written summary of the compacted history head.
            The text is inserted between stable summary markers so later
            compaction passes can detect and replace prior summaries.
    """
    content = Template(
        DEFAULT_SUMMARY_ITEM_TEMPLATE,
        trim_blocks=True,
        lstrip_blocks=True,
    ).render(
        open_tag=SUMMARY_OPEN_TAG,
        bridge=bridge,
        summary_text=summary_text,
        close_tag=SUMMARY_CLOSE_TAG,
    )
    return {
        "role": "user",
        "content": content.strip(),
    }


def is_summary_item(item: RunHistoryItem) -> bool:
    """Return whether one raw history item is a compaction summary message."""
    if not isinstance(item, dict):
        return False
    if item.get("role") != "user":
        return False
    content = item.get("content")
    return (
        isinstance(content, str)
        and SUMMARY_OPEN_TAG in content
        and SUMMARY_CLOSE_TAG in content
    )
