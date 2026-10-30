"""Default compactor implementation for harness conversation compaction."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

from agentlane.models import (
    MessageDict,
    MessageRole,
    ModelResponse,
    get_content_or_none,
    is_ephemeral_message,
    is_internal_message,
    provider_messages,
)

from .._run import RunHistoryItem
from ._config import DefaultCompactorConfig
from ._constants import MIN_BLOCKS_TO_SUMMARIZE, MIN_SHRINK_RATIO, SUMMARY_AFTER_TAIL
from ._errors import CompactionError, ContextOverflowError
from ._render import render_request_messages
from ._summary import is_summary_item, render_summary_item
from ._types import CompactionRequest, CompactionResult, SummaryPlacement


class DefaultCompactor:
    """Stock summary-plus-tail compactor for persisted harness history."""

    def __init__(self, config: DefaultCompactorConfig | None = None) -> None:
        """Initialize the compactor with validated stock defaults.

        Args:
            config: Optional summary prompt, bridge, tail budget, placement,
                and output-cap settings. `None` uses `DefaultCompactorConfig`.
        """
        self._config = config or DefaultCompactorConfig()

    @property
    def config(self) -> DefaultCompactorConfig:
        """Return the immutable configuration used by this compactor."""
        return self._config

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        """Return replacement history with an older-head summary plus tail."""
        call_args = _summary_model_args(
            request_model_args=request.model_args,
            summary_max_tokens=self._config.summary_max_tokens,
        )
        split = _split_history_for_summary(
            request=request,
            keep_recent_tokens=_effective_keep_recent_tokens(
                request=request,
                configured_keep_recent_tokens=self._config.keep_recent_tokens,
                call_args=call_args,
            ),
            keep_recent_messages=self._config.keep_recent_messages,
        )
        if _should_skip_compaction(request=request, split=split):
            return _no_op_result(request.history)

        summarizer_messages = render_request_messages(
            request.instructions,
            split.compacted_history,
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
        replacement_history = _replacement_history(
            summary_item=summary_item,
            tail_history=split.tail_history,
            placement=self._config.summary_placement,
        )
        _raise_if_replacement_does_not_shrink(
            request=request,
            replacement_history=replacement_history,
        )

        return CompactionResult(
            history=replacement_history,
            summary_content=str(summary_item["content"]),
            dropped_items=tuple(split.compacted_history),
            summarizer_response=summarizer_response,
        )


@dataclass(slots=True)
class _HistoryBlock:
    """Internal atomic unit for summary/tail splitting."""

    items: list[RunHistoryItem]
    is_summary: bool


@dataclass(frozen=True, slots=True)
class _HistorySplit:
    """Internal split between summarized history and retained tail."""

    compacted_history: list[RunHistoryItem]
    tail_history: list[RunHistoryItem]
    compacted_block_count: int
    has_compacted_summary: bool


def _split_history_for_summary(
    *,
    request: CompactionRequest,
    keep_recent_tokens: int,
    keep_recent_messages: int,
) -> _HistorySplit:
    tail_blocks_reversed: list[_HistoryBlock] = []
    compacted_blocks_reversed: list[_HistoryBlock] = []
    compacted_block_count = 0
    has_compacted_summary = False
    tail_tokens = 0
    tail_items = 0
    tail_closed = False
    blocks = _history_blocks(request.history)
    tail_anchor = _tail_anchor(blocks)
    has_tail = False

    # Walk newest-to-oldest so the retained tail is always a contiguous suffix.
    # Once a real block falls outside both recency budgets, every older real
    # block must be summarized; otherwise compaction could leave holes in the
    # conversation that make the model see cause without effect.
    for index in range(len(blocks) - 1, -1, -1):
        block = blocks[index]

        # Ephemeral context is always kept and does not use the tail budgets.
        # Inside the open tail it keeps its position. After the tail closes, it
        # waits on the summarizer side until the split moves it to the tail.
        if _is_ephemeral_block(block):
            if tail_closed:
                compacted_blocks_reversed.append(block)
            else:
                tail_blocks_reversed.append(block)
            continue

        # A previous summary is treated as old compressed context, not as fresh
        # tail. Recompaction should replace summaries instead of stacking them.
        if block.is_summary:
            compacted_blocks_reversed.append(block)
            has_compacted_summary = True
            if has_tail:
                tail_closed = True
            continue

        # After the tail is closed, avoid estimating older blocks. They are
        # already committed to the summarizer side of the split.
        protected = tail_anchor is not None and index >= tail_anchor
        if tail_closed and not protected:
            compacted_blocks_reversed.append(block)
            compacted_block_count += 1
            continue

        block_tokens = _estimate_history_tokens(request, block.items)
        should_keep = protected or _should_keep_in_tail(
            tail_items=tail_items,
            tail_tokens=tail_tokens,
            block_tokens=block_tokens,
            keep_recent_messages=keep_recent_messages,
            keep_recent_tokens=keep_recent_tokens,
        )
        if should_keep:
            tail_blocks_reversed.append(block)
            has_tail = True
            tail_tokens += block_tokens
            tail_items += len(block.items)
            continue

        compacted_blocks_reversed.append(block)
        compacted_block_count += 1
        tail_closed = True

    # The summarizer never receives ephemeral context, and the run still needs
    # it. Move older ephemeral items, including items inside mixed blocks, to
    # the start of the tail in their original order.
    compacted_history = _flatten_blocks(reversed(compacted_blocks_reversed))
    moved_ephemeral = [item for item in compacted_history if is_ephemeral_message(item)]
    if moved_ephemeral:
        compacted_history = [
            item for item in compacted_history if not is_ephemeral_message(item)
        ]

    return _HistorySplit(
        compacted_history=compacted_history,
        tail_history=[
            *moved_ephemeral,
            *_flatten_blocks(reversed(tail_blocks_reversed)),
        ],
        compacted_block_count=compacted_block_count,
        has_compacted_summary=has_compacted_summary,
    )


def _tail_anchor(blocks: list[_HistoryBlock]) -> int | None:
    """Protect the newest conversation block and any trailing internal context."""
    fallback: int | None = None
    for index in range(len(blocks) - 1, -1, -1):
        block = blocks[index]

        # The split always keeps ephemeral context, so it never anchors retention.
        if block.is_summary or _is_ephemeral_block(block):
            continue

        # With no ordinary conversation, protect only the newest context block.
        if fallback is None:
            fallback = index
        if not all(is_internal_message(item) for item in block.items):
            return index

    return fallback


def _is_ephemeral_block(block: _HistoryBlock) -> bool:
    return all(is_ephemeral_message(item) for item in block.items)


def _should_keep_in_tail(
    *,
    tail_items: int,
    tail_tokens: int,
    block_tokens: int,
    keep_recent_messages: int,
    keep_recent_tokens: int,
) -> bool:
    # The message floor is evaluated before the token ceiling. That deliberately
    # favors conversational continuity over an exact tail budget: the later
    # overflow and shrink checks still protect the model context if the minimum
    # recent items are too large to compact safely.
    if tail_items < keep_recent_messages:
        return True

    return tail_tokens + block_tokens <= keep_recent_tokens


def _history_blocks(history: tuple[RunHistoryItem, ...]) -> list[_HistoryBlock]:
    blocks: list[_HistoryBlock] = []
    index = 0
    while index < len(history):
        item = history[index]

        if _tool_result_call_id(item) is not None:
            # Tool results must stay attached to nearby context. A tool result
            # without a visible assistant opener is malformed history, but
            # gluing it to the previous block is safer than splitting it into a
            # standalone tail item the model cannot attribute to a call.
            if blocks:
                blocks[-1].items.append(item)
            else:
                blocks.append(_HistoryBlock(items=[item], is_summary=False))
            index += 1
            continue

        tool_call_ids = _assistant_tool_call_ids(item)
        if not tool_call_ids:
            blocks.append(_HistoryBlock(items=[item], is_summary=is_summary_item(item)))
            index += 1
            continue

        # Assistant tool calls and their matching tool results are one atomic
        # block. Keeping only one side would produce an invalid request for
        # providers that require tool-call/result pairing.
        block_items = [item]
        remaining_tool_call_ids = tool_call_ids
        index += 1
        while index < len(history):
            tool_call_id = _tool_result_call_id(history[index])
            if tool_call_id not in remaining_tool_call_ids:
                break

            block_items.append(history[index])
            remaining_tool_call_ids.remove(tool_call_id)
            index += 1
            if not remaining_tool_call_ids:
                break

        blocks.append(_HistoryBlock(items=block_items, is_summary=False))

    return blocks


def _flatten_blocks(blocks: Iterable[_HistoryBlock]) -> list[RunHistoryItem]:
    return [item for block in blocks for item in block.items]


def _assistant_tool_call_ids(item: RunHistoryItem) -> set[str]:
    if isinstance(item, ModelResponse):
        if not item.choices:
            return set()
        tool_calls = item.choices[0].message.tool_calls or []
        return {tool_call.id for tool_call in tool_calls if tool_call.id}

    if not isinstance(item, dict) or item.get("role") != MessageRole.ASSISTANT.value:
        return set()

    raw_tool_calls = item.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return set()

    ids: list[str] = []
    for raw_tool_call in cast(list[object], raw_tool_calls):
        if not isinstance(raw_tool_call, dict):
            continue

        tool_call = cast(dict[str, object], raw_tool_call)
        tool_call_id = tool_call.get("id")
        if isinstance(tool_call_id, str) and tool_call_id:
            ids.append(tool_call_id)

    return set(ids)


def _tool_result_call_id(item: RunHistoryItem) -> str | None:
    if not isinstance(item, dict) or item.get("role") != MessageRole.TOOL.value:
        return None

    tool_call_id = item.get("tool_call_id")
    if isinstance(tool_call_id, str) and tool_call_id:
        return tool_call_id

    return None


def _estimate_history_tokens(
    request: CompactionRequest,
    history: list[RunHistoryItem],
) -> int:
    return request.estimator(render_request_messages(None, history))


def _should_skip_compaction(
    *,
    request: CompactionRequest,
    split: _HistorySplit,
) -> bool:
    if not split.compacted_history:
        return True
    if request.signal.estimated_tokens >= request.signal.trigger_tokens:
        return False
    if split.compacted_block_count == 0:
        return True
    if split.has_compacted_summary:
        return False
    return split.compacted_block_count < MIN_BLOCKS_TO_SUMMARIZE


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
    call_args = dict(request_model_args or {})
    if (
        summary_max_tokens is not None
        and "max_tokens" not in call_args
        and "max_output_tokens" not in call_args
    ):
        call_args["max_tokens"] = summary_max_tokens

    return call_args or None


def _effective_keep_recent_tokens(
    *,
    request: CompactionRequest,
    configured_keep_recent_tokens: int,
    call_args: dict[str, Any] | None,
) -> int:
    output_cap = _output_token_cap(call_args)

    # Kept ephemeral context is a fixed cost of the next request, like the
    # instructions and the output cap.
    ephemeral_items = [item for item in request.history if is_ephemeral_message(item)]
    ephemeral_tokens = (
        _estimate_history_tokens(request, ephemeral_items) if ephemeral_items else 0
    )

    tail_budget = (
        request.signal.trigger_tokens
        - request.signal.instructions_tokens
        - output_cap
        - ephemeral_tokens
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

    for key in ("max_tokens", "max_output_tokens"):
        max_tokens = call_args.get(key)
        if isinstance(max_tokens, int) and max_tokens > 0:
            return max_tokens

    return 0


def _summary_text(response: ModelResponse) -> str:
    summary = get_content_or_none(response)
    if summary is None or not summary.strip():
        raise CompactionError("Default compactor received an empty summary response.")
    return summary.strip()


def _replacement_history(
    *,
    summary_item: MessageDict,
    tail_history: list[RunHistoryItem],
    placement: SummaryPlacement,
) -> list[RunHistoryItem]:
    if placement == SUMMARY_AFTER_TAIL:
        return [*tail_history, summary_item]
    return [summary_item, *tail_history]


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
