"""Pre-summary preservation decisions and validation for the stock compactor."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import cache
from typing import cast

from agentlane.models import (
    MessageDict,
    MessageRole,
    internal_message,
    is_ephemeral_message,
    is_internal_message,
)

from .._run import RunHistoryItem
from ._errors import CompactionError, ContextOverflowError
from ._render import render_request_messages
from ._summary import is_summary_item, render_summary_item
from ._types import (
    CompactionRequest,
    PreservationBlock,
    PreservationRequest,
    PreservationSelection,
    PreservationSelector,
    PreservedContext,
)

_CONTEXT_ORIGIN = "agentlane.compaction.context"
"""Owner of the combined exact-state projection replaced on each compaction."""


@dataclass(slots=True)
class _HistoryBlock:
    """Internal atomic unit for summary/tail splitting."""

    items: list[RunHistoryItem]
    messages: list[MessageDict]

    @property
    def ephemeral(self) -> bool:
        return all(is_ephemeral_message(item) for item in self.items)

    @property
    def replaceable(self) -> bool:
        return is_summary_item(self.items[0]) or (
            len(self.items) == 1
            and is_internal_message(self.items[0], origin=_CONTEXT_ORIGIN)
        )


@dataclass(frozen=True, slots=True)
class Preservation:
    """Validated material for one summary call and one replacement assembly."""

    dropped_items: list[RunHistoryItem]
    retained_history: list[RunHistoryItem]
    compacted_block_count: int
    has_compacted_context: bool
    summary_input: list[RunHistoryItem]


def _select_recent_blocks(
    *,
    block_tokens: Callable[[int], int],
    keep_recent_tokens: int,
    keep_recent_messages: int,
    blocks: list[_HistoryBlock],
) -> tuple[int, ...]:
    """Select a contiguous recent suffix before adding exact state anchors."""
    retained_ids: list[int] = []
    tail_tokens = 0
    tail_items = 0
    tail_closed = False
    tail_anchor = _tail_anchor(blocks)

    # Once a real block exceeds both recency preferences, every older real
    # block goes to the summarizer. Prior summaries never become fresh tail.
    for index in range(len(blocks) - 1, -1, -1):
        block = blocks[index]
        if block.ephemeral:
            continue
        if block.replaceable:
            if retained_ids:
                tail_closed = True
            continue

        protected = tail_anchor is not None and index >= tail_anchor
        if tail_closed and not protected:
            continue

        tokens = block_tokens(index)

        # The message floor can exceed the soft budget. Preflight below can
        # still trim optional blocks to fit the hard context-window budget.
        if (
            protected
            or tail_items < keep_recent_messages
            or tail_tokens + tokens <= keep_recent_tokens
        ):
            retained_ids.append(index)
            tail_tokens += tokens
            tail_items += sum(not is_ephemeral_message(item) for item in block.items)
        else:
            tail_closed = True

    return tuple(reversed(retained_ids))


def _tail_anchor(blocks: list[_HistoryBlock]) -> int | None:
    """Protect the newest conversation block and any trailing internal context."""
    fallback: int | None = None
    for index in range(len(blocks) - 1, -1, -1):
        block = blocks[index]
        if block.replaceable or block.ephemeral:
            continue

        # With no ordinary conversation, protect only the newest context block.
        if fallback is None:
            fallback = index
        if not all(is_internal_message(item) for item in block.items):
            return index

    return fallback


def _history_blocks(history: tuple[RunHistoryItem, ...]) -> list[_HistoryBlock]:
    blocks: list[_HistoryBlock] = []
    pending_tool_ids: set[str] = set()
    for item in history:
        # Native responses and raw messages use the same rendered tool shape.
        messages = render_request_messages(None, [item])
        result_ids = {
            call_id
            for message in messages
            if (call_id := _tool_result_call_id(message)) is not None
        }

        # Keep calls, intervening steering, and results together. Extra results
        # stay with the preceding block so a retained tail cannot orphan them.
        if blocks and (pending_tool_ids or result_ids):
            blocks[-1].items.append(item)
            blocks[-1].messages.extend(messages)
        else:
            blocks.append(_HistoryBlock(items=[item], messages=messages))

        for message in messages:
            pending_tool_ids.update(_assistant_tool_call_ids(message))
        pending_tool_ids.difference_update(result_ids)

    return blocks


def _assistant_tool_call_ids(item: MessageDict) -> set[str]:
    if item.get("role") != MessageRole.ASSISTANT.value:
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


def _tool_result_call_id(item: MessageDict) -> str | None:
    if item.get("role") != MessageRole.TOOL.value:
        return None

    tool_call_id = item.get("tool_call_id")
    if isinstance(tool_call_id, str) and tool_call_id:
        return tool_call_id

    return None


async def select_preservation(
    *,
    request: CompactionRequest,
    keep_recent_tokens: int,
    keep_recent_messages: int,
    summary_tokens: int,
    summary_bridge: str,
    selector: PreservationSelector | None,
) -> Preservation:
    """Select on detached input, then validate against the original blocks."""
    blocks = _history_blocks(request.history)
    _validate_context(request.preserved_context)

    anchor = _tail_anchor(blocks)
    ephemeral_messages = [
        message
        for block in blocks
        for message in block.messages
        if is_ephemeral_message(message)
    ]

    # Reuse per-block costs without estimating older blocks the default never
    # considers. Combined candidate estimates still use the complete messages.
    @cache
    def block_tokens(index: int, *, include_ephemeral: bool = True) -> int:
        messages = blocks[index].messages
        if not include_ephemeral:
            messages = [
                message for message in messages if not is_ephemeral_message(message)
            ]

        return request.estimator(messages) if messages else 0

    required_ids = {
        index
        for index, block in enumerate(blocks)
        if block.ephemeral
        or (not block.replaceable and anchor is not None and index >= anchor)
    }
    missing_anchors: list[MessageDict] = []
    for message in request.required_messages:
        # Repeated identical requests need only their latest occurrence protected.
        match = next(
            (
                index
                for index in range(len(blocks) - 1, -1, -1)
                if not blocks[index].replaceable and message in blocks[index].messages
            ),
            None,
        )
        if match is not None:
            required_ids.add(match)
        else:
            missing_anchors.append(message)

    # The hard budget includes state, exact anchors, and the summary envelope.
    # The recency budget is softer: a complete recent tool round may exceed it.
    envelope_tokens = request.estimator(
        [render_summary_item(bridge=summary_bridge, summary_text="")]
    )
    budget = (
        request.signal.context_window
        - request.signal.instructions_tokens
        - request.request_overhead_tokens
        - summary_tokens
        - envelope_tokens
    )
    context_tokens = request.estimator(
        [
            *_render_context(request.preserved_context, missing_anchors),
            *missing_anchors,
        ]
    )
    older_anchor_tokens = sum(
        block_tokens(index, include_ephemeral=False)
        for index in required_ids
        if anchor is None or index < anchor
    )
    recent_ids = _select_recent_blocks(
        block_tokens=lambda index: block_tokens(index, include_ephemeral=False),
        keep_recent_tokens=max(
            0, keep_recent_tokens - context_tokens - older_anchor_tokens
        ),
        keep_recent_messages=keep_recent_messages,
        blocks=blocks,
    )
    default = PreservationSelection(
        block_ids=tuple(sorted(set(recent_ids) | required_ids)),
        context=request.preserved_context,
    )

    def project(
        selection: PreservationSelection,
    ) -> tuple[set[int], list[MessageDict], list[MessageDict]]:
        ids = _validate_selection(selection, blocks, required_ids)
        messages = [
            message
            for index, block in enumerate(blocks)
            for message in block.messages
            if index in ids or is_ephemeral_message(message)
        ]
        context = _render_context(selection.context, [*messages, *missing_anchors])
        context.extend(missing_anchors)

        return ids, messages, context

    def estimate_tokens(selection: PreservationSelection) -> int:
        _, messages, context = project(selection)

        return request.estimator([*context, *messages])

    # Recency preferences must leave room for the reserved summary.
    for index in default.block_ids:
        if estimate_tokens(default) <= budget:
            break
        if index not in required_ids:
            default = replace(
                default,
                block_ids=tuple(item for item in default.block_ids if item != index),
            )

    selection = default
    if selector is not None:
        # Callers can inspect mutable provider dictionaries, but cannot edit the
        # runtime history or the canonical default through this detached view.
        # Deepcopy keeps the estimator function bound to the original blocks.
        selection = await selector(
            deepcopy(
                PreservationRequest(
                    blocks=tuple(
                        PreservationBlock(
                            id=index,
                            messages=tuple(block.messages),
                            tokens=block_tokens(index),
                            required=index in required_ids,
                            retainable=not block.replaceable,
                        )
                        for index, block in enumerate(blocks)
                    ),
                    default=default,
                    token_budget=max(0, budget),
                    # The estimator includes fixed ephemeral context; only the
                    # optional tail uses the reduced recency allowance.
                    retention_budget=keep_recent_tokens
                    + (
                        request.estimator(ephemeral_messages)
                        if ephemeral_messages
                        else 0
                    ),
                    instructions=tuple(
                        render_request_messages(request.instructions, [])
                    ),
                    required_messages=request.required_messages,
                    estimate_tokens=estimate_tokens,
                )
            )
        )

    selected_ids, retained_messages, context = project(selection)

    # Nothing to summarize is a true no-op, including its history and state.
    # Otherwise reject impossible preservation before making the summary call.
    if len(selected_ids) < len(blocks):
        tokens = request.estimator([*context, *retained_messages])
        if tokens > budget:
            raise ContextOverflowError(
                f"Preserved context exceeds the compaction budget ({tokens} > {budget})."
            )

    # Retained older evidence also informs the summary. Only the contiguous
    # recent suffix is excluded, so ranking cannot remove the task's premises
    # from the summarizer input while preserving an isolated older result.
    last_dropped_conversation = max(
        (
            index
            for index, block in enumerate(blocks)
            if index not in selected_ids and not block.replaceable
        ),
        default=-1,
    )

    retained: list[RunHistoryItem] = [*context]
    dropped: list[RunHistoryItem] = []
    summary_input: list[RunHistoryItem] = [
        message for message in missing_anchors if not is_ephemeral_message(message)
    ]
    compacted_block_count = 0
    has_compacted_context = False
    for index, block in enumerate(blocks):
        if index in selected_ids:
            retained.extend(block.items)
        else:
            # Run-scoped context survives even when its containing tool block
            # is dropped. It remains fixed input, never summary material.
            retained.extend(item for item in block.items if is_ephemeral_message(item))
            dropped.extend(
                item for item in block.items if not is_ephemeral_message(item)
            )
            compacted_block_count += not block.replaceable
            has_compacted_context |= block.replaceable
        if index not in selected_ids or index <= last_dropped_conversation:
            summary_input.extend(
                item for item in block.items if not is_ephemeral_message(item)
            )

    return Preservation(
        dropped_items=dropped,
        retained_history=retained,
        compacted_block_count=compacted_block_count,
        has_compacted_context=has_compacted_context,
        summary_input=summary_input,
    )


def _validate_selection(
    selection: object,
    blocks: list[_HistoryBlock],
    required_ids: set[int],
) -> set[int]:
    """Reject invalid policy output before summary generation or history edits."""
    if not isinstance(selection, PreservationSelection):
        raise CompactionError("Selector must return PreservationSelection.")

    raw_ids = cast(object, selection.block_ids)
    if not isinstance(raw_ids, tuple) or any(
        type(index) is not int or not 0 <= index < len(blocks)
        for index in selection.block_ids
    ):
        raise CompactionError("Selector returned invalid block IDs.")

    selected = set(selection.block_ids)
    if len(selected) != len(selection.block_ids):
        raise CompactionError("Selector returned duplicate block IDs.")

    if not required_ids <= selected:
        raise CompactionError("Selector omitted required conversation blocks.")

    if any(blocks[index].replaceable for index in selected):
        raise CompactionError("Selector cannot retain replaced summaries or state.")

    _validate_context(selection.context)

    return selected


def _validate_context(context: object) -> None:
    origins: set[str] = set()
    if not isinstance(context, tuple):
        raise CompactionError("Preserved context must be a tuple.")

    for item in cast(tuple[object, ...], context):
        if not isinstance(item, PreservedContext):
            raise CompactionError("Context entries must be PreservedContext values.")

        origin = cast(object, item.origin)
        content = cast(object, item.content)
        represented_by = cast(object, item.represented_by)
        if (
            not isinstance(origin, str)
            or not origin.strip()
            or not isinstance(content, str)
            or not content.strip()
            or not isinstance(represented_by, tuple)
            or any(
                not isinstance(message, dict)
                for message in cast(tuple[object, ...], represented_by)
            )
        ):
            raise CompactionError(
                "Preserved context requires an owner and nonempty text."
            )

        if item.origin in origins:
            raise CompactionError("Preserved context owners must be unique.")

        origins.add(item.origin)


def _render_context(
    context: tuple[PreservedContext, ...],
    retained_messages: list[MessageDict],
) -> list[MessageDict]:
    """Resolve state projection and deduplication before summary generation."""
    sections = [
        f"### {item.origin}\n{item.content}"
        for item in context
        if not any(message in retained_messages for message in item.represented_by)
    ]
    if not sections:
        return []

    # One owned envelope makes prior client-extracted state replaceable even
    # when the new selector has not yet declared which owner keys it will use.
    return [
        internal_message(
            {
                "role": MessageRole.USER.value,
                "content": "Preserved context:\n\n" + "\n\n".join(sections),
            },
            origin=_CONTEXT_ORIGIN,
        )
    ]
