"""Custom exceptions for harness conversation compaction."""


class CompactionError(RuntimeError):
    """One compaction attempt could not produce a valid replacement history."""


class ContextOverflowError(CompactionError):
    """The summarization request cannot fit the summarizer context window."""
