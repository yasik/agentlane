"""Public contracts for harness conversation compaction."""

from ._config import CompactionShimConfig, DefaultCompactorConfig
from ._constants import (
    BYTES_PER_TOKEN,
    DEFAULT_KEEP_RECENT_MESSAGES,
    DEFAULT_KEEP_RECENT_TOKENS,
    DEFAULT_SUMMARY_MAX_TOKENS,
    DEFAULT_TRIGGER_RATIO,
    MIN_BLOCKS_TO_SUMMARIZE,
    MIN_SHRINK_RATIO,
    NON_TEXT_PART_TOKENS,
    SUMMARY_CLOSE_TAG,
    SUMMARY_OPEN_TAG,
)
from ._default import DefaultCompactor
from ._errors import CompactionError, ContextOverflowError
from ._estimate import (
    estimate_message_tokens,
)
from ._prompt import (
    DEFAULT_COMPACTION_PROMPT,
    DEFAULT_SUMMARY_BRIDGE,
    DEFAULT_SUMMARY_ITEM_TEMPLATE,
)
from ._render import render_request_messages
from ._shim import CompactionShim
from ._summary import is_summary_item, render_summary_item
from ._types import (
    CompactionReport,
    CompactionRequest,
    CompactionResult,
    Compactor,
    ContextSignal,
    OnCompact,
    OnFailure,
    SummaryPlacement,
    TokenEstimator,
)

__all__ = [
    "BYTES_PER_TOKEN",
    "DEFAULT_COMPACTION_PROMPT",
    "DEFAULT_KEEP_RECENT_MESSAGES",
    "DEFAULT_KEEP_RECENT_TOKENS",
    "DEFAULT_SUMMARY_BRIDGE",
    "DEFAULT_SUMMARY_ITEM_TEMPLATE",
    "DEFAULT_SUMMARY_MAX_TOKENS",
    "DEFAULT_TRIGGER_RATIO",
    "MIN_BLOCKS_TO_SUMMARIZE",
    "MIN_SHRINK_RATIO",
    "NON_TEXT_PART_TOKENS",
    "SUMMARY_CLOSE_TAG",
    "SUMMARY_OPEN_TAG",
    "CompactionError",
    "CompactionReport",
    "CompactionRequest",
    "CompactionResult",
    "CompactionShim",
    "CompactionShimConfig",
    "Compactor",
    "ContextOverflowError",
    "ContextSignal",
    "DefaultCompactor",
    "DefaultCompactorConfig",
    "OnCompact",
    "OnFailure",
    "SummaryPlacement",
    "TokenEstimator",
    "estimate_message_tokens",
    "is_summary_item",
    "render_request_messages",
    "render_summary_item",
]
