"""Summary message helpers for harness conversation compaction."""

from jinja2 import Template

from agentlane.models import (
    MessageDict,
    MessageRole,
    internal_message,
    is_internal_message,
)

from .._run import RunHistoryItem
from ._constants import SUMMARY_CLOSE_TAG, SUMMARY_OPEN_TAG
from ._prompt import DEFAULT_SUMMARY_ITEM_TEMPLATE

_SUMMARY_ITEM_TEMPLATE = Template(
    DEFAULT_SUMMARY_ITEM_TEMPLATE,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_summary_item(*, bridge: str, summary_text: str) -> MessageDict:
    """Return an internal user-role summary with stable content tags.

    Args:
        bridge: Continuation text that tells the next model how to interpret
            the generated summary. Callers may pass a domain-specific bridge.
        summary_text: Model-written summary of the compacted history head.
            The text is inserted between stable summary markers so later
            compaction passes can detect and replace prior summaries.
    """
    content = _SUMMARY_ITEM_TEMPLATE.render(
        open_tag=SUMMARY_OPEN_TAG,
        bridge=bridge,
        summary_text=summary_text,
        close_tag=SUMMARY_CLOSE_TAG,
    )

    # Metadata hides synthetic context from public transcript serializers.
    # The provider still receives the tagged content, and snapshots retain it.
    return internal_message(
        {
            "role": MessageRole.USER.value,
            "content": content.strip(),
        },
        origin="agentlane.compaction",
    )


def is_summary_item(item: RunHistoryItem) -> bool:
    """Return whether one raw history item is a compaction summary message."""
    if not isinstance(item, dict):
        return False
    if item.get("role") != MessageRole.USER.value:
        return False

    if is_internal_message(item, origin="agentlane.compaction"):
        return True

    # Older snapshots have tags without internal metadata. Keep recognizing
    # the envelope, without treating ordinary text that quotes tags as a summary.
    content = item.get("content")
    if not isinstance(content, str):
        return False

    lines = content.strip().splitlines()

    return (
        len(lines) >= 2
        and lines[0].strip() == SUMMARY_OPEN_TAG
        and lines[-1].strip() == SUMMARY_CLOSE_TAG
    )
