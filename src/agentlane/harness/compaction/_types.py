"""Public type contracts for harness conversation compaction."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from agentlane.models import MessageDict, Model, ModelResponse

from .._run import RunHistoryItem, RunInstructions

type SummaryPlacement = Literal["before_tail", "after_tail"]
"""Where a generated summary item is placed relative to the retained tail."""

type TokenEstimator = Callable[[Sequence[MessageDict]], int]
"""Callable that estimates token cost for already-rendered model messages."""

type OnFailure = Literal["inject", "skip"]
"""Compaction failure policy: add a model-visible notice or skip silently."""

type ContextSource = Literal["server_usage", "estimate", "mixed"]
"""Source of the effective token estimate."""

type CompactionTrigger = Literal["auto", "manual"]
"""Whether accounting or an explicit request triggered compaction."""

type OnCompact = Callable[["CompactionReport"], Awaitable[None] | None]
"""Observer callback invoked after each compaction attempt report is created."""


@dataclass(frozen=True, kw_only=True, slots=True)
class ContextSignal:
    """Token-accounting snapshot for one trigger evaluation."""

    estimated_tokens: int
    """Effective estimate of the next request."""

    reported_tokens: int | None
    """Last server-reported total tokens, or `None` when unavailable."""

    instructions_tokens: int
    """Estimated tokens for rendered run instructions alone."""

    context_window: int
    """Configured model context window."""

    trigger_tokens: int
    """Token count at which compaction fires."""

    source: ContextSource
    """How `estimated_tokens` was derived."""

    turn_count: int
    """Run turn count at the time of evaluation."""

    history_item_count: int
    """Run history item count at the time of evaluation."""

    @property
    def remaining_tokens(self) -> int:
        """Return estimated tokens remaining before the configured window."""
        return self.context_window - self.estimated_tokens

    @property
    def used_fraction(self) -> float:
        """Return estimated context usage as a fraction of the full window."""
        if self.context_window <= 0:
            return 1.0
        return self.estimated_tokens / self.context_window


@dataclass(frozen=True, kw_only=True, slots=True)
class CompactionRequest:
    """Read-only input to one compactor invocation."""

    instructions: RunInstructions
    """Rendered-run instruction source active when compaction is requested."""

    history: tuple[RunHistoryItem, ...]
    """Immutable snapshot of the persisted run history to compact."""

    signal: ContextSignal
    """Trigger-evaluation signal that caused this compaction attempt."""

    model: Model[ModelResponse]
    """Model used by the compactor to write any generated summary."""

    model_args: dict[str, Any] | None
    """Provider/model call arguments passed to the summarizer model."""

    estimator: TokenEstimator
    """Estimator used for compactor-local budget and shrink decisions."""

    reason: CompactionTrigger
    """Whether the attempt was triggered by accounting or explicit request."""


@dataclass(kw_only=True, slots=True)
class CompactionResult:
    """Replacement history and observability details from one compactor."""

    history: list[RunHistoryItem]
    """Replacement history to install when compaction succeeds.

    Keep each ephemeral message from the request history. The active run still
    needs it, and the harness removes it when the run ends.
    """

    summary_content: str
    """Rendered summary item content or an empty string for no-op results."""

    dropped_items: tuple[RunHistoryItem, ...]
    """History items removed from the replacement history."""

    summarizer_response: ModelResponse | None
    """Raw summarizer response when a model-written summary was generated."""


@runtime_checkable
class Compactor(Protocol):
    """The extension seam for custom compaction logic.

    Implementations own the replacement history. They must keep ephemeral
    messages (see `is_ephemeral_message`) in that history and must not send
    them to a summarizer. Ephemeral context belongs to the active run only.
    """

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        """Return replacement history for `request.history`."""
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class CompactionReport:
    """Observability record for one compaction attempt."""

    signal: ContextSignal
    """Trigger signal observed before the compactor ran."""

    reason: CompactionTrigger
    """Whether this attempt was automatic or manually requested."""

    compacted: bool
    """Whether the attempt committed a model-backed or dropping rewrite."""

    items_before: int
    """Number of history items present before the attempt."""

    items_after: int
    """Number of history items present after the attempt."""

    estimated_tokens_after: int
    """Estimated token count after the replacement history was installed."""

    summary_chars: int
    """Character length of the rendered summary content."""

    dropped_items: tuple[RunHistoryItem, ...]
    """History items excluded from the replacement history."""

    summarizer_usage: tuple[int, int, int] | None
    """Summarizer usage as prompt, completion, and total tokens when available."""

    duration_seconds: float
    """Wall-clock duration of the compaction attempt."""

    attempt_key: str
    """Stable key observers can use to deduplicate retry reports."""

    error: str | None = None
    """Failure message when the attempt was skipped instead of raised."""
