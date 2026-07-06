"""Named constants for harness conversation compaction."""

DEFAULT_TRIGGER_RATIO = 0.9
"""Default automatic trigger point as a fraction of the configured context window."""

DEFAULT_KEEP_RECENT_TOKENS = 20_000
"""Default approximate-token budget for verbatim recent history retained after compaction."""

DEFAULT_SUMMARY_MAX_TOKENS = 4_096
"""Default summarizer output cap when the caller has not supplied `max_tokens`."""

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
