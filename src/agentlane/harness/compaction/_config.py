"""Configuration objects for harness conversation compaction."""

from dataclasses import dataclass

from ._constants import (
    DEFAULT_KEEP_RECENT_MESSAGES,
    DEFAULT_KEEP_RECENT_TOKENS,
    DEFAULT_SUMMARY_MAX_TOKENS,
    DEFAULT_TRIGGER_RATIO,
)
from ._prompt import DEFAULT_COMPACTION_PROMPT, DEFAULT_SUMMARY_BRIDGE
from ._types import OnFailure, SummaryPlacement


@dataclass(frozen=True, kw_only=True, slots=True)
class CompactionShimConfig:
    """Validated trigger and failure settings for the compaction shim."""

    context_window: int
    """Maximum context window, in tokens, for the model being protected."""

    trigger_ratio: float = DEFAULT_TRIGGER_RATIO
    """Fraction of `context_window` used when `trigger_tokens` is unset."""

    trigger_tokens: int | None = None
    """Absolute token threshold for compaction; overrides `trigger_ratio`."""

    on_failure: OnFailure = "inject"
    """Whether compaction failures inject a model note or skip silently."""

    name: str = "compaction"
    """Stable prefix used for report and attempt-key identity."""

    def __post_init__(self) -> None:
        """Validate compaction trigger settings."""
        if self.context_window <= 0:
            raise ValueError("context_window must be positive.")
        if not (0 < self.trigger_ratio <= 1):
            raise ValueError("trigger_ratio must be greater than 0 and at most 1.")
        if self.trigger_tokens is not None and self.trigger_tokens <= 0:
            raise ValueError("trigger_tokens must be positive when provided.")
        if (
            self.trigger_tokens is not None
            and self.trigger_tokens > self.context_window
        ):
            raise ValueError("trigger_tokens cannot exceed context_window.")
        if self.on_failure not in {"inject", "skip"}:
            raise ValueError("on_failure must be 'inject' or 'skip'.")
        if not self.name:
            raise ValueError("name must be non-empty.")

    def resolved_trigger_tokens(self) -> int:
        """Return the absolute token count that should trigger compaction."""
        if self.trigger_tokens is not None:
            return self.trigger_tokens
        return max(1, int(self.context_window * self.trigger_ratio))


@dataclass(frozen=True, kw_only=True, slots=True)
class DefaultCompactorConfig:
    """Validated defaults for the stock summary-plus-tail compactor."""

    prompt: str = DEFAULT_COMPACTION_PROMPT
    """Prompt sent to the summarizer model for the compacted history head."""

    summary_bridge: str = DEFAULT_SUMMARY_BRIDGE
    """Bridge text prepended to the generated summary in replacement history."""

    keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS
    """Approximate-token budget for verbatim recent history retained as tail."""

    keep_recent_messages: int = DEFAULT_KEEP_RECENT_MESSAGES
    """Minimum number of newest history items retained as verbatim tail."""

    summary_placement: SummaryPlacement = "before_tail"
    """Where the summary item is placed relative to retained recent history."""

    summary_max_tokens: int | None = DEFAULT_SUMMARY_MAX_TOKENS
    """Default summarizer output cap; `None` leaves the provider unbounded."""

    def __post_init__(self) -> None:
        """Validate stock compactor settings."""
        if not self.prompt:
            raise ValueError("prompt must be non-empty.")
        if not self.summary_bridge:
            raise ValueError("summary_bridge must be non-empty.")
        if self.keep_recent_tokens <= 0:
            raise ValueError("keep_recent_tokens must be positive.")
        if self.keep_recent_messages <= 0:
            raise ValueError("keep_recent_messages must be positive.")
        if self.summary_placement not in {"before_tail", "after_tail"}:
            raise ValueError("summary_placement must be 'before_tail' or 'after_tail'.")
        if self.summary_max_tokens is not None and self.summary_max_tokens <= 0:
            raise ValueError("summary_max_tokens must be positive when provided.")
