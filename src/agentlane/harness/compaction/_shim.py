"""Harness shim that installs conversation compaction results."""

import inspect
import time
from collections.abc import MutableMapping, Sequence
from typing import Literal, Protocol, cast, runtime_checkable

from agentlane.models import MessageDict, Model, ModelResponse, get_usage_totals

from ..shims import BoundShim, PreparedTurn, Shim, ShimBindingContext
from ._config import CompactionShimConfig
from ._default import DefaultCompactor
from ._estimate import estimate_message_tokens
from ._render import render_request_messages
from ._types import (
    CompactionReport,
    CompactionRequest,
    CompactionResult,
    Compactor,
    ContextSignal,
    OnCompact,
    TokenEstimator,
)

_FAILURE_NOTICE_STATE_KEY = "agentlane.compaction.failure-notices"


@runtime_checkable
class _ModelBackedTask(Protocol):
    @property
    def model(self) -> Model[ModelResponse] | None:
        """Return the model used by the runner."""


class CompactionShim(Shim):
    """Trigger and install conversation compaction during turn preparation."""

    def __init__(
        self,
        config: CompactionShimConfig,
        compactor: Compactor | None = None,
        *,
        estimator: TokenEstimator = estimate_message_tokens,
        on_compact: OnCompact | None = None,
    ) -> None:
        """Initialize a compaction shim.

        Args:
            config: Trigger threshold, failure policy, and shim name.
            compactor: Optional compactor implementation. `None` uses
                `DefaultCompactor`.
            estimator: Token estimator for trigger and report accounting.
            on_compact: Optional observer called after each compaction attempt.
        """
        self._config = config
        self._compactor = compactor if compactor is not None else DefaultCompactor()
        self._estimator = estimator
        self._on_compact = on_compact

    @property
    def name(self) -> str:
        """Return the configured report and attempt-key prefix."""
        return self._config.name

    @property
    def config(self) -> CompactionShimConfig:
        """Return the trigger and failure configuration."""
        return self._config

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        """Bind this shim to the model exposed by one harness agent."""
        if not isinstance(context.task, _ModelBackedTask):
            raise TypeError("CompactionShim requires a task with a model.")

        model = context.task.model
        if model is None:
            raise TypeError("CompactionShim requires a task with a configured model.")

        return _BoundCompactionShim(
            config=self._config,
            compactor=self._compactor,
            estimator=self._estimator,
            model=model,
            on_compact=self._on_compact,
        )


class _BoundCompactionShim(BoundShim):
    """Bound compaction session for one model-backed harness agent."""

    def __init__(
        self,
        *,
        config: CompactionShimConfig,
        compactor: Compactor,
        estimator: TokenEstimator,
        model: Model[ModelResponse],
        on_compact: OnCompact | None,
    ) -> None:
        self._config = config
        self._compactor = compactor
        self._estimator = estimator
        self._model = model
        self._on_compact = on_compact

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        """Compact persisted history when the configured threshold is reached."""
        # The shim runs before the runner builds the next request. All accounting
        # must therefore use the same rendered input shape that the runner would
        # send if no compaction happened.
        signal = _context_signal(
            turn=turn,
            config=self._config,
            estimator=self._estimator,
        )
        if signal.estimated_tokens < signal.trigger_tokens:
            return

        request = CompactionRequest(
            instructions=turn.run_state.instructions,
            history=tuple(turn.run_state.history),
            signal=signal,
            model=self._model,
            model_args=(dict(turn.model_args) if turn.model_args is not None else None),
            estimator=self._estimator,
            reason="auto",
        )
        started_at = time.monotonic()
        try:
            result = await self._compactor.compact(request)
            report = _success_report(
                shim_name=self._config.name,
                signal=signal,
                request=request,
                result=result,
                duration_seconds=time.monotonic() - started_at,
            )
        except Exception as error:
            report = _failure_report(
                shim_name=self._config.name,
                signal=signal,
                request=request,
                duration_seconds=time.monotonic() - started_at,
                error=error,
            )
            await _notify_compaction(self._on_compact, report)
            if self._config.on_failure == "skip":
                return
            _store_failure_notice(
                turn,
                shim_name=self._config.name,
                notice=_failure_notice(report),
            )
            return

        if report.compacted:
            turn.replace_history(result.history)
        await _notify_compaction(self._on_compact, report)

    async def transform_messages(
        self,
        turn: PreparedTurn,
        messages: list[MessageDict],
    ) -> list[MessageDict] | None:
        """Inject a non-persisted compaction failure notice for this turn."""
        notices = _take_failure_notices(turn, shim_name=self._config.name)
        if not notices:
            return None
        return [*messages, *notices]


def _context_signal(
    *,
    turn: PreparedTurn,
    config: CompactionShimConfig,
    estimator: TokenEstimator,
) -> ContextSignal:
    # Render once up front so the local estimator observes exactly the messages
    # that would be sent to the protected model for this turn.
    messages = render_request_messages(
        turn.run_state.instructions,
        turn.run_state.history,
    )
    local_estimate = estimator(messages)
    reported_tokens = _latest_reported_tokens(turn.run_state.responses)
    trigger_tokens = config.resolved_trigger_tokens()
    estimated_tokens, source = _effective_tokens(
        local_estimate=local_estimate,
        reported_tokens=reported_tokens,
        trigger_tokens=trigger_tokens,
    )
    instructions_tokens = estimator(
        render_request_messages(turn.run_state.instructions, [])
    )
    return ContextSignal(
        estimated_tokens=estimated_tokens,
        reported_tokens=reported_tokens,
        instructions_tokens=instructions_tokens,
        context_window=config.context_window,
        trigger_tokens=trigger_tokens,
        source=source,
        turn_count=turn.run_state.turn_count,
        history_item_count=len(turn.run_state.history),
    )


def _latest_reported_tokens(responses: Sequence[ModelResponse]) -> int | None:
    for response in reversed(responses):
        usage = get_usage_totals(response)
        if usage is not None:
            return usage.total_tokens
    return None


def _effective_tokens(
    *,
    local_estimate: int,
    reported_tokens: int | None,
    trigger_tokens: int,
) -> tuple[int, Literal["server_usage", "estimate", "mixed"]]:
    if reported_tokens is None:
        return local_estimate, "estimate"
    # Provider usage is useful when it agrees with or exceeds a locally-large
    # request, but it may be stale after handoff/resume. Do not let an inherited
    # prior response trigger compaction when the current rendered request is
    # below the local threshold.
    if reported_tokens >= local_estimate and local_estimate >= trigger_tokens:
        return reported_tokens, "server_usage"
    if reported_tokens == local_estimate:
        return local_estimate, "server_usage"
    return local_estimate, "mixed"


def _success_report(
    *,
    shim_name: str,
    signal: ContextSignal,
    request: CompactionRequest,
    result: CompactionResult,
    duration_seconds: float,
) -> CompactionReport:
    # Custom compactors can legitimately no-op by returning the same history.
    # Treat that as a successful attempt report without installing a rewrite.
    compacted = not _same_history(result.history, request.history)
    estimated_tokens_after = (
        request.estimator(render_request_messages(request.instructions, result.history))
        if compacted
        else signal.estimated_tokens
    )
    return CompactionReport(
        signal=signal,
        reason=request.reason,
        compacted=compacted,
        items_before=len(request.history),
        items_after=len(result.history) if compacted else len(request.history),
        estimated_tokens_after=estimated_tokens_after,
        summary_chars=len(result.summary_content),
        dropped_items=result.dropped_items,
        summarizer_usage=_summarizer_usage(result.summarizer_response),
        duration_seconds=duration_seconds,
        attempt_key=_attempt_key(
            shim_name=shim_name,
            signal=signal,
            reason=request.reason,
        ),
        error=None,
    )


def _failure_report(
    *,
    shim_name: str,
    signal: ContextSignal,
    request: CompactionRequest,
    duration_seconds: float,
    error: Exception,
) -> CompactionReport:
    return CompactionReport(
        signal=signal,
        reason=request.reason,
        compacted=False,
        items_before=len(request.history),
        items_after=len(request.history),
        estimated_tokens_after=signal.estimated_tokens,
        summary_chars=0,
        dropped_items=(),
        summarizer_usage=None,
        duration_seconds=duration_seconds,
        attempt_key=_attempt_key(
            shim_name=shim_name,
            signal=signal,
            reason=request.reason,
        ),
        error=str(error),
    )


def _summarizer_usage(response: ModelResponse | None) -> tuple[int, int, int] | None:
    usage = get_usage_totals(response)
    if usage is None:
        return None

    return (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)


def _same_history(
    left: Sequence[object],
    right: Sequence[object],
) -> bool:
    return len(left) == len(right) and all(
        left_item == right_item
        for left_item, right_item in zip(left, right, strict=True)
    )


def _attempt_key(*, shim_name: str, signal: ContextSignal, reason: str) -> str:
    return (
        f"{shim_name}:{reason}:turn={signal.turn_count}:"
        f"history={signal.history_item_count}:tokens={signal.estimated_tokens}"
    )


async def _notify_compaction(
    on_compact: OnCompact | None,
    report: CompactionReport,
) -> None:
    if on_compact is None:
        return
    try:
        result = on_compact(report)
        if inspect.isawaitable(result):
            await result
    except Exception as error:
        # Observers are diagnostics hooks. A broken observer must not affect
        # history replacement, failure injection, or normal turn delivery.
        _ = error
        return


def _store_failure_notice(
    turn: PreparedTurn,
    *,
    shim_name: str,
    notice: MessageDict,
) -> None:
    raw_context = turn.transient_state.context
    if not isinstance(raw_context, MutableMapping):
        # Custom non-mapping transient contexts cannot carry a one-turn note.
        # Persisting the notice is the remaining way to keep the model informed.
        turn.append_history_item(notice)
        return

    context = cast(MutableMapping[str, object], raw_context)
    key = _failure_notice_state_key(shim_name)
    raw_notices: object = context.get(key)
    notices: list[MessageDict] = []

    if isinstance(raw_notices, list):
        notices.extend(cast(list[MessageDict], raw_notices))

    notices.append(notice)
    context[key] = notices


def _take_failure_notices(
    turn: PreparedTurn,
    *,
    shim_name: str,
) -> list[MessageDict]:
    raw_context = turn.transient_state.context
    if not isinstance(raw_context, MutableMapping):
        return []

    context = cast(MutableMapping[str, object], raw_context)
    raw_notices: object = context.pop(_failure_notice_state_key(shim_name), None)
    if not isinstance(raw_notices, list):
        return []

    notices = cast(list[object], raw_notices)
    return [cast(MessageDict, item) for item in notices if isinstance(item, dict)]


def _failure_notice_state_key(shim_name: str) -> str:
    return f"{shim_name}:{_FAILURE_NOTICE_STATE_KEY}"


def _failure_notice(report: CompactionReport) -> MessageDict:
    error = report.error or "unknown compaction failure"
    return {
        "role": "user",
        "content": (
            "Context compaction failed before this turn. The original "
            "conversation history was preserved, so continue normally. "
            f"Compaction error: {_single_line(error)}"
        ),
    }


def _single_line(value: str) -> str:
    return " ".join(value.split())
