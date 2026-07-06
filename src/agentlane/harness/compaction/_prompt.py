"""Default text used by harness conversation compaction."""

DEFAULT_SUMMARY_BRIDGE = (
    "Another language model started to solve this problem and produced a "
    "summary of its thinking process. You also have access to the state of the "
    "tools that were used by that language model. Use this to build on the "
    "work that has already been done and avoid duplicating work. Here is the "
    "summary produced by the other language model, use the information in this "
    "summary to assist with your own analysis:"
)
"""Summary prefix used as the handoff bridge before generated summary text."""

DEFAULT_COMPACTION_PROMPT = """
You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for
another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work."""
"""Summarization prompt sent to the model that writes the handoff summary."""

DEFAULT_SUMMARY_ITEM_TEMPLATE = """{{ open_tag }}
{{ bridge }}

{{ summary_text }}
{{ close_tag }}"""
"""Jinja2 template for the synthetic user-role summary history item."""
