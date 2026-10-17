"""Named constants for harness conversation compaction."""

from typing import Final

SUMMARY_BEFORE_TAIL: Final = "before_tail"
"""Selects placement of the generated summary before the retained recent history."""

SUMMARY_AFTER_TAIL: Final = "after_tail"
"""Selects placement of the generated summary after the retained recent history."""

FAILURE_INJECT: Final = "inject"
"""Selects a model-visible notice when compaction fails."""

FAILURE_SKIP: Final = "skip"
"""Selects continuation with the original history and no compaction failure notice."""

SOURCE_SERVER_USAGE: Final = "server_usage"
"""Marks token accounting that accepts the provider-reported usage."""

SOURCE_ESTIMATE: Final = "estimate"
"""Marks token accounting based only on the local estimate."""

SOURCE_MIXED: Final = "mixed"
"""Marks a local estimate retained after checking provider usage against trigger policy."""

REASON_AUTO: Final = "auto"
"""Recorded when token accounting triggers compaction automatically."""

DEFAULT_SHIM_NAME = "compaction"
"""Used as the default prefix for compaction reports and attempt keys."""

DEFAULT_TRIGGER_RATIO = 0.9
"""Default automatic trigger point as a fraction of the configured context window."""

DEFAULT_KEEP_RECENT_TOKENS = 20_000
"""Default approximate-token budget for verbatim recent history retained after compaction."""

DEFAULT_KEEP_RECENT_MESSAGES = 4
"""Default minimum count of newest history items retained after compaction."""

DEFAULT_SUMMARY_MAX_TOKENS = 4_096
"""Default summarizer output cap when the caller has not supplied an output cap."""

BYTES_PER_TOKEN = 4
"""Approximate UTF-8 bytes per token used by the byte heuristic estimator."""

NON_TEXT_PART_TOKENS = 1_000
"""Flat approximate-token charge for each non-text message content part."""

MIN_BLOCKS_TO_SUMMARIZE = 3
"""Minimum number of older turn blocks required before compaction should summarize."""

MIN_SHRINK_RATIO = 0.9
"""Maximum post/pre estimate ratio considered a meaningful shrink after compaction."""

SUMMARY_OPEN_TAG = "<<COMPACTION_SUMMARY_V1_9B2D4C8F_BEGIN>>"
"""Opening marker used to identify a synthetic compaction summary history item."""

SUMMARY_CLOSE_TAG = "<<COMPACTION_SUMMARY_V1_9B2D4C8F_END>>"
"""Closing marker used to identify a synthetic compaction summary history item."""
