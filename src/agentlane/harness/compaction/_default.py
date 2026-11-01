"""Compact history by selecting what to preserve before generating a summary."""

from typing import Any

from agentlane.models import (
    MessageDict,
    MessageRole,
    ModelResponse,
    get_content_or_none,
    provider_messages,
)

from .._run import RunHistoryItem
from ._config import DefaultCompactorConfig
from ._constants import MIN_BLOCKS_TO_SUMMARIZE, MIN_SHRINK_RATIO, SUMMARY_AFTER_TAIL
from ._errors import CompactionError, ContextOverflowError
from ._preservation import Preservation, select_preservation
from ._render import render_request_messages
from ._summary import render_summary_item
from ._types import (
    CompactionRequest,
    CompactionResult,
    PreservationSelector,
    SummaryPlacement,
)


class DefaultCompactor:
    """Replace older conversation history with a summary and retained context.

    Preservation decides which complete conversation blocks and owner-supplied
    state must remain visible. The default policy keeps recent history and exact
    required messages. A client selector can change the optional blocks and state
    to retain; the framework still validates required blocks and token limits.

    The compactor returns a replacement candidate without installing it. The
    compaction shim installs that candidate, and the harness validates the full
    request after all message transforms before accepting the replacement.
    """

    def __init__(
        self,
        config: DefaultCompactorConfig | None = None,
        *,
        selector: PreservationSelector | None = None,
    ) -> None:
        """Initialize the compactor with validated stock defaults.

        Args:
            config: Optional summary prompt, bridge, tail budget, placement,
                and output-cap settings. `None` uses `DefaultCompactorConfig`.
            selector: Optional async policy that extends or replaces the default
                block and state selection before summarization.
        """
        self._config = config or DefaultCompactorConfig()
        self._selector = selector

    @property
    def config(self) -> DefaultCompactorConfig:
        """Return the immutable configuration used by this compactor."""
        return self._config

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        """Select preserved context, summarize older history, and assemble once.

        First reserve room for summary output, then select and validate the
        blocks and state to retain. Skip summary generation when there is no
        useful history to compact. These decisions precede the summary call.

        Summarize the older history, including retained older evidence needed
        to understand it; leave the retained recent suffix out of that input.
        Check that the summary request fits and require a complete, nonempty
        response. Combine the tagged summary with the selected context in the
        configured order, then require a sufficient reduction in history size.

        Return the candidate and removed items for the caller to install and
        report. Selection, generation, or size errors leave the input unchanged.
        """
        call_args = _summary_model_args(
            request_model_args=request.model_args,
            summary_max_tokens=self._config.summary_max_tokens,
        )

        # Decide all retention and state projections before spending a summary call.
        preservation = await select_preservation(
            request=request,
            keep_recent_tokens=_effective_keep_recent_tokens(
                request=request,
                configured_keep_recent_tokens=self._config.keep_recent_tokens,
                call_args=call_args,
            ),
            keep_recent_messages=self._config.keep_recent_messages,
            summary_tokens=_output_token_cap(call_args),
            summary_bridge=self._config.summary_bridge,
            selector=self._selector,
        )
        if _should_skip_compaction(request=request, preservation=preservation):
            return _no_op_result(request.history)

        summarizer_messages = render_request_messages(
            request.instructions,
            preservation.summary_input,
        )
        summarizer_messages.insert(
            0,
            {"role": MessageRole.SYSTEM.value, "content": self._config.prompt},
        )
        _raise_if_summarizer_request_overflows(
            messages=summarizer_messages,
            request=request,
            call_args=call_args,
        )

        summarizer_response = await request.model(
            provider_messages(summarizer_messages),
            extra_call_args=call_args,
        )
        summary_text = _summary_text(summarizer_response)
        summary_item = render_summary_item(
            bridge=self._config.summary_bridge,
            summary_text=summary_text,
        )

        # Assemble exactly once; no later restoration changes this candidate.
        replacement_history = _replacement_history(
            summary_item=summary_item,
            retained_history=preservation.retained_history,
            placement=self._config.summary_placement,
        )
        _raise_if_replacement_does_not_shrink(
            request=request,
            replacement_history=replacement_history,
        )

        return CompactionResult(
            history=replacement_history,
            summary_content=str(summary_item["content"]),
            dropped_items=tuple(preservation.dropped_items),
            summarizer_response=summarizer_response,
        )


def _should_skip_compaction(
    *,
    request: CompactionRequest,
    preservation: Preservation,
) -> bool:
    if not preservation.dropped_items:
        return True
    if request.signal.estimated_tokens >= request.signal.trigger_tokens:
        return False

    if preservation.compacted_block_count == 0:
        return True

    if preservation.has_compacted_context:
        return False

    return preservation.compacted_block_count < MIN_BLOCKS_TO_SUMMARIZE


def _no_op_result(history: tuple[RunHistoryItem, ...]) -> CompactionResult:
    return CompactionResult(
        history=list(history),
        summary_content="",
        dropped_items=(),
        summarizer_response=None,
    )


def _summary_model_args(
    *,
    request_model_args: dict[str, Any] | None,
    summary_max_tokens: int | None,
) -> dict[str, Any] | None:
    # Copy shared options, but remove every alias for the main response cap.
    # Otherwise a provider can select that cap instead of the summary cap, or
    # reject the request because it has two output limits. Keep main args intact.
    call_args = dict(request_model_args or {})
    for key in ("max_tokens", "max_output_tokens", "max_completion_tokens"):
        call_args.pop(key, None)

    # None leaves the summary call without an explicit per-call output limit.
    if summary_max_tokens is not None:
        call_args["max_tokens"] = summary_max_tokens

    return call_args or None


def _effective_keep_recent_tokens(
    *,
    request: CompactionRequest,
    configured_keep_recent_tokens: int,
    call_args: dict[str, Any] | None,
) -> int:
    output_cap = _output_token_cap(call_args)
    tail_budget = (
        request.signal.trigger_tokens
        - request.signal.instructions_tokens
        - request.request_overhead_tokens
        - output_cap
    )
    return min(configured_keep_recent_tokens, max(1, tail_budget))


def _raise_if_summarizer_request_overflows(
    *,
    messages: list[MessageDict],
    request: CompactionRequest,
    call_args: dict[str, Any] | None,
) -> None:
    prompt_tokens = request.estimator(messages)
    output_cap = _output_token_cap(call_args)
    required_tokens = prompt_tokens + output_cap
    if required_tokens <= request.signal.context_window:
        return

    raise ContextOverflowError(
        "Default compactor summarization request would exceed the configured "
        "context window "
        f"({required_tokens} estimated tokens > "
        f"{request.signal.context_window} context window)."
    )


def _output_token_cap(call_args: dict[str, Any] | None) -> int:
    if call_args is None:
        return 0

    max_tokens = call_args.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        return max_tokens

    return 0


def _summary_text(response: ModelResponse) -> str:
    # A truncated or interrupted summary can omit facts from the history it
    # replaces. Keep the original history unless generation completed normally.
    if not response.choices or response.choices[0].finish_reason != "stop":
        raise CompactionError("Default compactor received an incomplete summary.")

    summary = get_content_or_none(response)
    if summary is None or not summary.strip():
        raise CompactionError("Default compactor received an empty summary response.")

    return summary.strip()


def _replacement_history(
    *,
    summary_item: MessageDict,
    retained_history: list[RunHistoryItem],
    placement: SummaryPlacement,
) -> list[RunHistoryItem]:
    if placement == SUMMARY_AFTER_TAIL:
        return [*retained_history, summary_item]

    return [summary_item, *retained_history]


def _raise_if_replacement_does_not_shrink(
    *,
    request: CompactionRequest,
    replacement_history: list[RunHistoryItem],
) -> None:
    original_tokens = request.estimator(render_request_messages(None, request.history))
    if original_tokens <= 0:
        return

    replacement_tokens = request.estimator(
        render_request_messages(None, replacement_history)
    )
    shrink_ratio = replacement_tokens / original_tokens
    if shrink_ratio <= MIN_SHRINK_RATIO:
        return

    raise CompactionError(
        "Default compactor replacement did not shrink history enough "
        f"({shrink_ratio:.2f} post/pre estimate > {MIN_SHRINK_RATIO:.2f})."
    )
