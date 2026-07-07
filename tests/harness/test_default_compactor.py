import re
from collections.abc import Sequence
from typing import Any, cast

import pytest

from agentlane.harness import RunHistoryItem
from agentlane.harness.compaction import (
    DEFAULT_COMPACTION_PROMPT,
    CompactionError,
    CompactionRequest,
    ContextOverflowError,
    ContextSignal,
    DefaultCompactor,
    DefaultCompactorConfig,
    is_summary_item,
    render_summary_item,
)
from agentlane.models import MessageDict, Model, ModelResponse, Tools
from agentlane.runtime import CancellationToken
from agentlane.tracing import Span

from .tools_test_utils import SequenceModel, make_assistant_response, make_tool_call

_TOKEN_PATTERN = re.compile(r"\[t=(\d+)]")


class _RecordingModel(SequenceModel):
    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__(responses)
        self.call_args: list[dict[str, object] | None] = []

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: Tools | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        self.call_args.append(dict(extra_call_args) if extra_call_args else None)
        return await super().get_response(
            messages=messages,
            extra_call_args=extra_call_args,
            schema=schema,
            tools=tools,
            cancellation_token=cancellation_token,
            parent_span=parent_span,
            **kwargs,
        )


def _token_estimator(messages: Sequence[MessageDict]) -> int:
    return sum(
        int(match)
        for message in messages
        for match in _TOKEN_PATTERN.findall(str(message.get("content", "")))
    )


def _message_content(item: RunHistoryItem) -> str:
    assert isinstance(item, dict)
    message = cast(MessageDict, item)
    return str(message.get("content", ""))


def _request(
    history: list[RunHistoryItem],
    model: Model[ModelResponse],
    *,
    instructions: str = "Base instruction.",
    model_args: dict[str, Any] | None = None,
    context_window: int = 10_000,
) -> CompactionRequest:
    return CompactionRequest(
        instructions=instructions,
        history=tuple(history),
        signal=ContextSignal(
            estimated_tokens=_token_estimator(
                [{"role": "user", "content": str(item)} for item in history]
            ),
            reported_tokens=None,
            instructions_tokens=0,
            context_window=context_window,
            trigger_tokens=max(1, context_window - 1),
            source="estimate",
            turn_count=7,
            history_item_count=len(history),
        ),
        model=model,
        model_args=model_args,
        estimator=_token_estimator,
        reason="auto",
    )


@pytest.mark.asyncio
async def test_default_compactor_summarizes_old_head_and_retains_recent_tail() -> None:
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "old user [t=4]"},
        {"role": "assistant", "content": "old assistant [t=4]"},
        {"role": "user", "content": "old follow-up [t=4]"},
        {"role": "assistant", "content": "recent answer [t=2]"},
        {"role": "user", "content": "recent prompt [t=2]"},
    ]
    model = _RecordingModel([make_assistant_response("summarized head [t=1]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=4,
            keep_recent_messages=2,
            summary_max_tokens=64,
        )
    )

    result = await compactor.compact(_request(history, model))

    assert len(model.calls) == 1
    assert model.calls[0][0] == {
        "role": "system",
        "content": DEFAULT_COMPACTION_PROMPT,
    }
    assert model.calls[0][1] == {"role": "system", "content": "Base instruction."}
    assert model.calls[0][2:] == history[:3]
    assert model.call_args == [{"max_tokens": 64}]
    assert model.call_tools == [None]

    assert is_summary_item(result.history[0])
    assert "summarized head [t=1]" in _message_content(result.history[0])
    assert result.history[1:] == history[-2:]
    assert result.dropped_items == tuple(history[:3])
    assert result.summarizer_response is not None
    assert result.summary_content == _message_content(result.history[0])


@pytest.mark.asyncio
async def test_default_compactor_can_place_summary_after_tail_and_preserve_max_tokens() -> (
    None
):
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "old user [t=4]"},
        {"role": "assistant", "content": "old assistant [t=4]"},
        {"role": "user", "content": "old follow-up [t=4]"},
        {"role": "assistant", "content": "recent answer [t=1]"},
    ]
    model = _RecordingModel([make_assistant_response("tail handoff [t=1]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=1,
            keep_recent_messages=1,
            summary_placement="after_tail",
            summary_max_tokens=64,
        )
    )

    result = await compactor.compact(
        _request(
            history,
            model,
            model_args={"temperature": 0.2, "max_tokens": 12},
        )
    )

    assert result.history[0] == history[-1]
    assert is_summary_item(result.history[1])
    assert model.call_args == [{"temperature": 0.2, "max_tokens": 12}]


@pytest.mark.asyncio
async def test_default_compactor_after_tail_recompaction_keeps_real_tail() -> None:
    prior_summary = render_summary_item(
        bridge="Continue from the prior summary.",
        summary_text="prior summary [t=4]",
    )
    latest_item: MessageDict = {
        "role": "assistant",
        "content": "latest real item [t=4]",
    }
    history: list[RunHistoryItem] = [
        latest_item,
        prior_summary,
    ]
    model = _RecordingModel([make_assistant_response("new summary [t=1]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=1,
            keep_recent_messages=1,
            summary_placement="after_tail",
            summary_max_tokens=1,
        )
    )

    result = await compactor.compact(_request(history, model, context_window=8))

    assert result.history[0] == latest_item
    assert is_summary_item(result.history[1])
    assert result.dropped_items == (prior_summary,)


@pytest.mark.asyncio
async def test_default_compactor_leaves_summary_output_uncapped_when_config_is_none() -> (
    None
):
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "old user [t=4]"},
        {"role": "assistant", "content": "old assistant [t=4]"},
        {"role": "user", "content": "old follow-up [t=4]"},
        {"role": "assistant", "content": "recent answer [t=1]"},
    ]
    model = _RecordingModel([make_assistant_response("uncapped summary [t=1]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=1,
            keep_recent_messages=1,
            summary_max_tokens=None,
        )
    )

    result = await compactor.compact(_request(history, model, context_window=13))

    assert is_summary_item(result.history[0])
    assert model.call_args == [None]


@pytest.mark.asyncio
async def test_default_compactor_preserves_native_max_output_tokens() -> None:
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "old user [t=4]"},
        {"role": "assistant", "content": "old assistant [t=4]"},
        {"role": "user", "content": "old follow-up [t=4]"},
        {"role": "assistant", "content": "recent answer [t=1]"},
    ]
    model = _RecordingModel([make_assistant_response("responses cap [t=1]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=1,
            keep_recent_messages=1,
            summary_max_tokens=64,
        )
    )

    result = await compactor.compact(
        _request(
            history,
            model,
            model_args={"max_output_tokens": 2},
            context_window=14,
        )
    )

    assert is_summary_item(result.history[0])
    assert model.call_args == [{"max_output_tokens": 2}]


@pytest.mark.asyncio
async def test_default_compactor_default_tail_budget_adapts_to_context_window() -> None:
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "old user [t=1200]"},
        {"role": "assistant", "content": "old assistant [t=1200]"},
        {"role": "user", "content": "old follow-up [t=1200]"},
        {"role": "assistant", "content": "recent answer [t=1200]"},
        {"role": "user", "content": "recent prompt [t=1200]"},
        {"role": "assistant", "content": "latest answer [t=1200]"},
    ]
    model = _RecordingModel([make_assistant_response("default summary [t=1]")])
    compactor = DefaultCompactor()

    result = await compactor.compact(_request(history, model, context_window=7201))

    assert result.history[1:] == history[-4:]
    assert result.dropped_items == tuple(history[:2])
    assert model.call_args == [{"max_tokens": 4096}]


@pytest.mark.asyncio
async def test_default_compactor_message_floor_keeps_recent_tail_beyond_token_budget() -> (
    None
):
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "old user [t=3]"},
        {"role": "assistant", "content": "old assistant [t=3]"},
        {"role": "user", "content": "old follow-up [t=3]"},
        {"role": "assistant", "content": "recent answer [t=3]"},
        {"role": "user", "content": "recent prompt [t=3]"},
        {"role": "assistant", "content": "latest answer [t=3]"},
    ]
    model = _RecordingModel([make_assistant_response("summary [t=1]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=1,
            keep_recent_messages=3,
            summary_max_tokens=64,
        )
    )

    result = await compactor.compact(_request(history, model))

    assert result.history[1:] == history[-3:]
    assert result.dropped_items == tuple(history[:3])


@pytest.mark.asyncio
async def test_default_compactor_keeps_newest_item_when_it_exceeds_tail_budget() -> (
    None
):
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "old user [t=4]"},
        {"role": "assistant", "content": "old assistant [t=4]"},
        {"role": "user", "content": "old follow-up [t=4]"},
        {"role": "assistant", "content": "oversized latest answer [t=8]"},
    ]
    model = _RecordingModel([make_assistant_response("summary [t=1]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=1,
            keep_recent_messages=1,
            summary_max_tokens=64,
        )
    )

    result = await compactor.compact(_request(history, model))

    assert result.history[1:] == history[-1:]
    assert result.dropped_items == tuple(history[:-1])


@pytest.mark.asyncio
async def test_default_compactor_keeps_tail_contiguous_after_large_middle_block() -> (
    None
):
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "old large [t=10]"},
        {"role": "assistant", "content": "old small [t=1]"},
        {"role": "user", "content": "middle large [t=10]"},
        {"role": "assistant", "content": "newest small [t=1]"},
    ]
    model = _RecordingModel([make_assistant_response("summary [t=1]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=3,
            keep_recent_messages=1,
            summary_max_tokens=64,
        )
    )

    result = await compactor.compact(_request(history, model))

    assert result.history[1:] == history[-1:]
    assert result.dropped_items == tuple(history[:-1])


@pytest.mark.asyncio
async def test_default_compactor_retains_tool_call_round_trip_atomically() -> None:
    tool_response = make_assistant_response(
        None,
        tool_calls=[
            make_tool_call(
                tool_id="call_1",
                name="read",
                arguments='{"path": "README.md"}',
            )
        ],
    )
    tool_result: MessageDict = {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "tool output [t=8]",
    }
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "old user [t=4]"},
        {"role": "assistant", "content": "old assistant [t=4]"},
        {"role": "user", "content": "old follow-up [t=4]"},
        tool_response,
        tool_result,
    ]
    model = _RecordingModel([make_assistant_response("summary [t=1]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=1,
            keep_recent_messages=1,
            summary_max_tokens=64,
        )
    )

    result = await compactor.compact(_request(history, model))

    assert result.history[1:] == [tool_response, tool_result]
    assert result.dropped_items == tuple(history[:3])
    assert model.calls[0][2:] == history[:3]


@pytest.mark.asyncio
async def test_default_compactor_retains_dict_tool_call_round_trip_atomically() -> None:
    tool_response: MessageDict = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": '{"path": "README.md"}',
                },
            }
        ],
    }
    tool_result: MessageDict = {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "tool output [t=8]",
    }
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "old user [t=4]"},
        {"role": "assistant", "content": "old assistant [t=4]"},
        {"role": "user", "content": "old follow-up [t=4]"},
        tool_response,
        tool_result,
    ]
    model = _RecordingModel([make_assistant_response("summary [t=1]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=1,
            keep_recent_messages=1,
            summary_max_tokens=64,
        )
    )

    result = await compactor.compact(_request(history, model))

    assert result.history[1:] == [tool_response, tool_result]
    assert result.dropped_items == tuple(history[:3])


@pytest.mark.asyncio
async def test_default_compactor_keeps_extra_tool_results_with_preceding_block() -> (
    None
):
    tool_response = make_assistant_response(
        None,
        tool_calls=[
            make_tool_call(
                tool_id="call_1",
                name="read",
                arguments='{"path": "README.md"}',
            )
        ],
    )
    tool_result: MessageDict = {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "tool output [t=8]",
    }
    duplicate_tool_result: MessageDict = {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "duplicate output [t=1]",
    }
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "old user [t=4]"},
        {"role": "assistant", "content": "old assistant [t=4]"},
        {"role": "user", "content": "old follow-up [t=4]"},
        tool_response,
        tool_result,
        duplicate_tool_result,
    ]
    model = _RecordingModel([make_assistant_response("summary [t=1]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=1,
            keep_recent_messages=1,
            summary_max_tokens=64,
        )
    )

    result = await compactor.compact(_request(history, model))

    assert result.history[1:] == [tool_response, tool_result, duplicate_tool_result]
    assert result.dropped_items == tuple(history[:3])


@pytest.mark.asyncio
async def test_default_compactor_recompacts_prior_summary_with_one_fresh_block() -> (
    None
):
    prior_summary = render_summary_item(
        bridge="Continue from the prior summary.",
        summary_text="prior summary [t=4]",
    )
    history: list[RunHistoryItem] = [
        prior_summary,
        {"role": "user", "content": "fresh older turn [t=4]"},
        {"role": "assistant", "content": "latest answer [t=1]"},
    ]
    model = _RecordingModel([make_assistant_response("combined summary [t=1]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=1,
            keep_recent_messages=1,
            summary_max_tokens=64,
        )
    )

    result = await compactor.compact(_request(history, model))

    assert model.calls[0][2:] == history[:-1]
    assert result.dropped_items == tuple(history[:-1])
    assert result.history[1:] == history[-1:]


@pytest.mark.asyncio
async def test_default_compactor_compacts_fewer_blocks_when_trigger_still_fired() -> (
    None
):
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "huge older turn [t=6]"},
        {"role": "assistant", "content": "huge latest answer [t=6]"},
    ]
    model = _RecordingModel([make_assistant_response("summary [t=1]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=1,
            keep_recent_messages=1,
            summary_max_tokens=1,
        )
    )

    result = await compactor.compact(_request(history, model, context_window=13))

    assert result.history[1:] == history[-1:]
    assert result.dropped_items == tuple(history[:1])


@pytest.mark.asyncio
async def test_default_compactor_recompacts_oversized_prior_summary_when_triggered() -> (
    None
):
    prior_summary = render_summary_item(
        bridge="Continue from the prior summary.",
        summary_text="oversized prior summary [t=12]",
    )
    history: list[RunHistoryItem] = [
        prior_summary,
        {"role": "assistant", "content": "latest answer [t=1]"},
    ]
    model = _RecordingModel([make_assistant_response("shorter summary [t=1]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=1,
            keep_recent_messages=1,
            summary_max_tokens=1,
        )
    )

    result = await compactor.compact(_request(history, model, context_window=14))

    assert model.calls[0][2:] == [prior_summary]
    assert result.dropped_items == (prior_summary,)
    assert result.history[1:] == history[-1:]


@pytest.mark.asyncio
async def test_default_compactor_shrink_check_ignores_instruction_tokens() -> None:
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "old user [t=5]"},
        {"role": "assistant", "content": "old assistant [t=5]"},
        {"role": "user", "content": "old follow-up [t=5]"},
        {"role": "assistant", "content": "latest answer [t=15]"},
    ]
    model = _RecordingModel([make_assistant_response("large summary [t=10]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=15,
            keep_recent_messages=1,
            summary_max_tokens=64,
        )
    )

    result = await compactor.compact(
        _request(
            history,
            model,
            instructions="Large system prompt [t=100]",
        )
    )

    assert result.history[1:] == history[-1:]
    assert result.dropped_items == tuple(history[:3])


@pytest.mark.asyncio
async def test_default_compactor_returns_noop_when_too_few_blocks_would_be_summarized() -> (
    None
):
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "old user [t=4]"},
        {"role": "assistant", "content": "old assistant [t=4]"},
        {"role": "assistant", "content": "recent answer [t=1]"},
        {"role": "user", "content": "recent prompt [t=1]"},
    ]
    model = _RecordingModel([make_assistant_response("unused [t=1]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=2,
            keep_recent_messages=2,
            summary_max_tokens=64,
        )
    )

    result = await compactor.compact(_request(history, model))

    assert result.history == history
    assert result.summary_content == ""
    assert result.dropped_items == ()
    assert result.summarizer_response is None
    assert model.calls == []


@pytest.mark.asyncio
async def test_default_compactor_replaces_prior_summary_when_recompacting() -> None:
    prior_summary = render_summary_item(
        bridge="Continue from the prior summary.",
        summary_text="prior summary [t=4]",
    )
    history: list[RunHistoryItem] = [
        prior_summary,
        {"role": "user", "content": "old user [t=4]"},
        {"role": "assistant", "content": "old assistant [t=4]"},
        {"role": "user", "content": "old follow-up [t=4]"},
        {"role": "assistant", "content": "recent answer [t=1]"},
    ]
    model = _RecordingModel([make_assistant_response("new summary [t=1]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=1,
            keep_recent_messages=1,
            summary_max_tokens=64,
        )
    )

    result = await compactor.compact(_request(history, model))

    assert prior_summary in model.calls[0]
    assert result.dropped_items == tuple(history[:-1])
    assert len([item for item in result.history if is_summary_item(item)]) == 1
    assert "new summary [t=1]" in _message_content(result.history[0])
    assert prior_summary not in result.history


@pytest.mark.parametrize(
    "response",
    [
        make_assistant_response("   "),
        make_assistant_response(None),
    ],
)
@pytest.mark.asyncio
async def test_default_compactor_rejects_empty_summary_response(
    response: ModelResponse,
) -> None:
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "old user [t=4]"},
        {"role": "assistant", "content": "old assistant [t=4]"},
        {"role": "user", "content": "old follow-up [t=4]"},
        {"role": "assistant", "content": "recent answer [t=1]"},
    ]
    model = _RecordingModel([response])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=1,
            keep_recent_messages=1,
            summary_max_tokens=64,
        )
    )

    with pytest.raises(CompactionError, match="empty summary"):
        await compactor.compact(_request(history, model))


@pytest.mark.asyncio
async def test_default_compactor_rejects_non_shrinking_replacement() -> None:
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "old user [t=2]"},
        {"role": "assistant", "content": "old assistant [t=2]"},
        {"role": "user", "content": "old follow-up [t=2]"},
        {"role": "assistant", "content": "recent answer [t=2]"},
    ]
    model = _RecordingModel([make_assistant_response("bloated summary [t=6]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(
            keep_recent_tokens=2,
            keep_recent_messages=1,
            summary_max_tokens=64,
        )
    )

    with pytest.raises(CompactionError, match="did not shrink"):
        await compactor.compact(_request(history, model))


@pytest.mark.asyncio
async def test_default_compactor_rejects_summarizer_request_overflow() -> None:
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "old user [t=4]"},
        {"role": "assistant", "content": "old assistant [t=4]"},
        {"role": "user", "content": "old follow-up [t=4]"},
        {"role": "assistant", "content": "recent answer [t=1]"},
    ]
    model = _RecordingModel([make_assistant_response("unused [t=1]")])
    compactor = DefaultCompactor(
        DefaultCompactorConfig(keep_recent_tokens=1, keep_recent_messages=1)
    )

    with pytest.raises(ContextOverflowError, match="exceed"):
        await compactor.compact(_request(history, model, context_window=10))

    assert model.calls == []
