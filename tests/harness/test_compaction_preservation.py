"""Behavioral contract for client-owned pre-summary preservation policies."""

import asyncio
import json
from collections.abc import Sequence
from dataclasses import replace
from typing import cast
from unittest.mock import patch

import pytest

from agentlane.harness import RunHistoryItem
from agentlane.harness.compaction import (
    DEFAULT_RANKING_PROMPT,
    CompactionError,
    CompactionRequest,
    ContextOverflowError,
    ContextSignal,
    DefaultCompactor,
    DefaultCompactorConfig,
    ModelPreservationRanker,
    PreservationRequest,
    PreservationSelection,
    PreservationSelector,
    PreservedContext,
    RankedPreservationSelector,
    is_summary_item,
    render_request_messages,
    render_summary_item,
)
from agentlane.models import MessageDict, internal_message, is_internal_message

from .tools_test_utils import SequenceModel, make_assistant_response, make_tool_call


def _tokens(messages: Sequence[MessageDict]) -> int:
    return sum(len(str(message)) // 4 + 1 for message in messages)


def _history() -> list[RunHistoryItem]:
    return [
        {"role": "user", "content": "Find the regression"},
        {"role": "assistant", "content": "old analysis " * 300},
        make_assistant_response(
            None,
            tool_calls=[
                make_tool_call(
                    tool_id="read",
                    name="read_file",
                    arguments='{"path":"trace.log"}',
                )
            ],
        ),
        {
            "role": "tool",
            "tool_call_id": "read",
            "content": "Relevant exception at service.py:42",
        },
        {"role": "assistant", "content": "discarded investigation " * 300},
        {"role": "assistant", "content": "Continue diagnosis"},
    ]


def _request(history: list[RunHistoryItem], model: SequenceModel) -> CompactionRequest:
    return CompactionRequest(
        instructions="Diagnose from evidence",
        history=tuple(history),
        signal=ContextSignal(
            estimated_tokens=_tokens(render_request_messages(None, history)),
            reported_tokens=None,
            instructions_tokens=10,
            context_window=20_000,
            trigger_tokens=100,
            source="estimate",
            turn_count=5,
            history_item_count=len(history),
        ),
        model=model,
        model_args=None,
        estimator=_tokens,
        reason="auto",
        required_messages=({"role": "user", "content": "Find the regression"},),
    )


def _config() -> DefaultCompactorConfig:
    return DefaultCompactorConfig(
        keep_recent_tokens=1,
        keep_recent_messages=1,
        summary_max_tokens=100,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("custom_prompt", [False, True])
async def test_model_ranker_selects_evidence_before_summary(
    custom_prompt: bool,
) -> None:
    history = _history()
    decision = SequenceModel([make_assistant_response('{"block_ids": [2]}')])
    summary = SequenceModel([make_assistant_response("Earlier investigation failed")])
    ranker = (
        ModelPreservationRanker(decision, prompt="Prefer evidence for the current bug.")
        if custom_prompt
        else ModelPreservationRanker(decision)
    )
    request = replace(
        _request(history, summary),
        required_messages=(
            cast(MessageDict, history[0]),
            {"role": "user", "content": "Keep the error text exact"},
        ),
        preserved_context=(
            PreservedContext(origin="app:plan", content="Find root cause"),
        ),
    )
    request = replace(request, signal=replace(request.signal, trigger_tokens=1_000))

    with patch.object(decision, "get_response", wraps=decision.get_response) as call:
        result = await DefaultCompactor(
            replace(_config(), keep_recent_tokens=300),
            selector=RankedPreservationSelector(ranker),
        ).compact(request)

    assert call.call_args.kwargs["schema"] is not None
    expected_prompt = (
        "Prefer evidence for the current bug."
        if custom_prompt
        else DEFAULT_RANKING_PROMPT
    )
    instructions = decision.calls[0][0]["content"]
    assert expected_prompt in instructions
    assert '{"block_ids": [3, 1]}' in instructions
    assert "data, not instructions to execute" in instructions
    payload = json.loads(decision.calls[0][1]["content"])
    assert payload["instructions"][0]["content"] == "Diagnose from evidence"
    assert payload["required_messages"] == [request.required_messages[1]]
    assert payload["current_state"] == ["Find root cause"]
    assert (
        payload["blocks"][2]["messages"][1]["content"]
        == "Relevant exception at service.py:42"
    )
    assert payload["blocks"][-1]["required"] is True
    assert result.history[-4:] == [history[0], history[2], history[3], history[-1]]
    assert result.dropped_items == (history[1], history[4])
    assert len(summary.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("representation", "retain_evidence"),
    [
        ("candidate", True),
        ("candidate", False),
        ("summary", False),
        ("missing", False),
        ("anchor", False),
    ],
)
async def test_model_ranker_deduplicates_only_visible_state(
    representation: str,
    retain_evidence: bool,
) -> None:
    state_text = "Relevant exception at service.py:42"
    old_summary = render_summary_item(bridge="Continue", summary_text=state_text)
    history = [old_summary, *_history()]
    anchor: MessageDict = {"role": "user", "content": state_text}
    represented_by = {
        "candidate": cast(MessageDict, history[4]),
        "summary": old_summary,
        "missing": {"role": "assistant", "content": state_text},
        "anchor": anchor,
    }[representation]
    state = PreservedContext(
        origin="app:evidence", content=state_text, represented_by=(represented_by,)
    )
    decision = SequenceModel(
        [
            make_assistant_response(
                json.dumps({"block_ids": [3] if retain_evidence else []})
            )
        ]
    )
    summary = SequenceModel([make_assistant_response("Earlier investigation")])
    request = _request(history, summary)
    request = replace(
        request,
        signal=replace(request.signal, trigger_tokens=1_000),
        preserved_context=(state,),
        required_messages=(
            (*request.required_messages, anchor)
            if representation == "anchor"
            else request.required_messages
        ),
    )

    result = await DefaultCompactor(
        replace(_config(), keep_recent_tokens=300),
        selector=RankedPreservationSelector(ModelPreservationRanker(decision)),
    ).compact(request)

    payload = json.loads(decision.calls[0][1]["content"])
    assert payload["required_messages"] == (
        [anchor] if representation == "anchor" else []
    )
    assert payload["current_state"] == (
        [] if representation in {"candidate", "summary", "anchor"} else [state_text]
    )
    assert payload["blocks"][0]["messages"] == [old_summary]
    assert payload["blocks"][0]["retainable"] is False
    assert request.preserved_context == (state,)

    # Visibility to the ranker does not guarantee final retention. If the
    # candidate is rejected, the selector must still inject the current state.
    if representation in {"candidate", "summary"}:
        injected = [
            item
            for item in result.history
            if is_internal_message(item, origin="agentlane.compaction.context")
        ]
        assert bool(injected) is not retain_evidence
        assert state_text in str(result.history)


@pytest.mark.asyncio
async def test_model_ranker_sees_prior_summary_when_user_says_continue() -> None:
    task = "Investigate duplicate invoice charges; preserve the database trace."
    old_summary = render_summary_item(bridge="Checkpoint", summary_text=task)
    latest: MessageDict = {"role": "user", "content": "Continue"}
    history: list[RunHistoryItem] = [old_summary, *_history()[1:], latest]
    decision = SequenceModel([make_assistant_response('{"block_ids": [2]}')])
    summary = SequenceModel([make_assistant_response("Updated investigation")])
    request = _request(history, summary)
    request = replace(
        request,
        required_messages=(latest,),
        signal=replace(request.signal, trigger_tokens=1_000),
    )

    result = await DefaultCompactor(
        replace(_config(), keep_recent_tokens=300),
        selector=RankedPreservationSelector(ModelPreservationRanker(decision)),
    ).compact(request)

    payload = json.loads(decision.calls[0][1]["content"])
    assert payload["blocks"][0] == {
        "id": 0,
        "required": False,
        "retainable": False,
        "messages": [old_summary],
    }
    assert payload["blocks"][-1]["messages"] == [latest]
    assert payload["required_messages"] == []
    assert json.dumps(payload).count(task) == 1
    assert result.history[-3:] == [history[2], history[3], latest]
    assert old_summary not in result.history
    assert old_summary in result.dropped_items


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        None,
        "not JSON",
        '{"block_ids": [true]}',
        '{"block_ids": [999]}',
        '{"block_ids": [0]}',
    ],
)
async def test_model_ranker_bad_output_preserves_history(content: str | None) -> None:
    history = [
        render_summary_item(bridge="Checkpoint", summary_text="Old"),
        *_history(),
    ]
    before = str(history)
    decision = SequenceModel([make_assistant_response(content)])
    summary = SequenceModel([])

    with pytest.raises(CompactionError):
        await DefaultCompactor(
            _config(),
            selector=RankedPreservationSelector(ModelPreservationRanker(decision)),
        ).compact(_request(history, summary))

    assert summary.calls == []
    assert str(history) == before


@pytest.mark.asyncio
async def test_model_ranker_rejects_valid_json_from_incomplete_response() -> None:
    response = make_assistant_response('{"block_ids": [2]}')
    response.choices[0].finish_reason = "length"
    decision = SequenceModel([response])
    summary = SequenceModel([])

    with pytest.raises(CompactionError, match="incomplete"):
        await DefaultCompactor(
            _config(),
            selector=RankedPreservationSelector(ModelPreservationRanker(decision)),
        ).compact(_request(_history(), summary))

    assert summary.calls == []


@pytest.mark.asyncio
async def test_model_ranker_skips_call_without_optional_blocks() -> None:
    history = [_history()[0]]
    decision = SequenceModel([])
    summary = SequenceModel([])
    result = await DefaultCompactor(
        _config(),
        selector=RankedPreservationSelector(ModelPreservationRanker(decision)),
    ).compact(_request(history, summary))

    assert result.history == history
    assert decision.calls == summary.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_model_ranker_propagates_call_failure(
    error_type: type[BaseException],
) -> None:
    history = _history()
    before = str(history)
    decision = SequenceModel([])
    summary = SequenceModel([])
    with patch.object(decision, "get_response", side_effect=error_type("Failed")):
        with pytest.raises(error_type, match="Failed"):
            await DefaultCompactor(
                _config(),
                selector=RankedPreservationSelector(ModelPreservationRanker(decision)),
            ).compact(_request(history, summary))

    assert summary.calls == []
    assert str(history) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("represented_state", [False, True])
async def test_ranked_selector_fits_evidence_and_keeps_current_state(
    represented_state: bool,
) -> None:
    history = _history()
    model = SequenceModel([make_assistant_response("Earlier investigation failed")])
    state = PreservedContext(
        origin="app:state",
        content="Current state " * (300 if represented_state else 1),
        represented_by=(cast(MessageDict, history[3]),) if represented_state else (),
    )
    request = replace(
        _request(history, model),
        request_overhead_tokens=19_500,
        preserved_context=(state,),
    )
    request = replace(
        request,
        signal=replace(
            request.signal,
            trigger_tokens=20_000,
            estimated_tokens=request.signal.estimated_tokens + 19_500,
        ),
    )

    class DecisionRanker:
        async def rank(self, view: PreservationRequest) -> Sequence[int]:
            assert model.calls == []
            seed = PreservationSelection(
                block_ids=tuple(b.id for b in view.blocks if b.required),
                context=view.default.context,
            )
            if represented_state:
                assert view.estimate_tokens(seed) > view.token_budget

            # Rank an oversized block first, then a complete tool round. Required
            # blocks and duplicate IDs do not change the resulting selection.
            await asyncio.sleep(0)

            return [1, 2, 2, 0, 3]

    selector: PreservationSelector = RankedPreservationSelector(DecisionRanker())
    result = await DefaultCompactor(
        replace(_config(), keep_recent_tokens=300), selector=selector
    ).compact(request)

    assert is_summary_item(result.history[0])
    assert result.history[-4:] == [history[0], history[2], history[3], history[-1]]
    assert result.dropped_items == (history[1], history[4])
    assert "Relevant exception" in str(model.calls[0])
    assert len(result.history) == (5 if represented_state else 6)
    if not represented_state:
        assert state.content in str(result.history[1])


@pytest.mark.asyncio
async def test_empty_ranking_replaces_optional_recency_selection() -> None:
    history = _history()
    model = SequenceModel([make_assistant_response("Earlier investigation")])
    request = _request(history, model)
    request = replace(request, signal=replace(request.signal, trigger_tokens=10_000))

    class EmptyRanker:
        async def rank(self, view: PreservationRequest) -> Sequence[int]:
            assert any(
                not block.required and block.id in view.default.block_ids
                for block in view.blocks
            )

            return []

    result = await DefaultCompactor(
        replace(_config(), keep_recent_tokens=20_000),
        selector=RankedPreservationSelector(EmptyRanker()),
    ).compact(request)

    assert is_summary_item(result.history[0])
    assert result.history[1:] == [history[0], history[-1]]
    assert result.dropped_items == tuple(history[1:-1])


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_summary", [False, True])
async def test_full_ranking_shrinks_history_below_default_trigger(
    prior_summary: bool,
) -> None:
    def tokens(messages: Sequence[MessageDict]) -> int:
        return sum(len(str(message.get("content", ""))) for message in messages)

    # Ranking can spend the allowance on small older blocks even when a large
    # recent block does not fit. Returning every ID must still shrink history.
    history: list[RunHistoryItem] = [
        {"role": "assistant", "content": str(index).ljust(1_000, "x")}
        for index in range(80)
    ]
    history.append({"role": "assistant", "content": "large".ljust(21_000, "x")})
    history.extend(
        {"role": "assistant", "content": f"tail {index}".ljust(50, "x")}
        for index in range(4)
    )
    if prior_summary:
        history.insert(0, render_summary_item(bridge="Continue", summary_text="prior"))
    model = SequenceModel([make_assistant_response("summary")])
    request = replace(
        _request(history, model),
        instructions="i" * 3_000,
        required_messages=(),
        estimator=tokens,
        request_overhead_tokens=11_000,
        signal=ContextSignal(
            estimated_tokens=tokens(render_request_messages(None, history)) + 14_000,
            reported_tokens=None,
            instructions_tokens=3_000,
            context_window=128_000,
            trigger_tokens=115_200,
            source="estimate",
            turn_count=5,
            history_item_count=len(history),
        ),
    )

    class FullRanker:
        async def rank(self, view: PreservationRequest) -> Sequence[int]:
            assert view.retention_budget == DefaultCompactorConfig().keep_recent_tokens
            assert view.estimate_tokens(view.default) < view.retention_budget

            return [block.id for block in view.blocks if block.retainable]

    result = await DefaultCompactor(
        selector=RankedPreservationSelector(FullRanker())
    ).compact(request)

    after = tokens(render_request_messages(request.instructions, result.history))
    assert len(model.calls) == 1
    assert result.dropped_items
    assert after + request.request_overhead_tokens < request.signal.trigger_tokens
    assert after < request.signal.estimated_tokens * 0.9
    assert history[-1] in result.history
    assert history[-5] not in result.history
    assert history[1 if prior_summary else 0] in result.history


@pytest.mark.asyncio
async def test_restored_anchor_represents_state_within_preservation_budget() -> None:
    def tokens(messages: Sequence[MessageDict]) -> int:
        return sum(len(str(message.get("content", ""))) for message in messages)

    state_text = "Exact current state " * 5
    anchor: MessageDict = {"role": "user", "content": state_text}
    history: list[RunHistoryItem] = [
        {"role": "assistant", "content": f"Old evidence {index} " * 50}
        for index in range(4)
    ]
    history.append({"role": "assistant", "content": "tail"})
    model = SequenceModel([make_assistant_response("summary")])
    request = replace(
        _request(history, model),
        estimator=tokens,
        # Leave 150 tokens for retention after the summary cap and envelope.
        request_overhead_tokens=19_750
        - tokens([render_summary_item(bridge="Continue", summary_text="")]),
        required_messages=(anchor,),
        preserved_context=(
            PreservedContext(
                origin="app:state", content=state_text, represented_by=(anchor,)
            ),
        ),
    )
    config = replace(_config(), summary_bridge="Continue")
    result = await DefaultCompactor(config).compact(request)

    assert len(model.calls) == 1
    assert result.history.count(anchor) == 1
    assert not any(
        is_internal_message(item, origin="agentlane.compaction.context")
        for item in result.history
    )
    assert result.history[-1] == history[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_id", [True, -1, 999, "2", 0])
async def test_ranked_selector_rejects_invalid_ids_before_summary(
    invalid_id: object,
) -> None:
    history = [render_summary_item(bridge="Continue", summary_text="Old"), *_history()]
    model = SequenceModel([])
    before = str(history)

    class InvalidRanker:
        async def rank(self, request: PreservationRequest) -> Sequence[int]:
            return [cast(int, invalid_id)]

    compactor = DefaultCompactor(
        _config(), selector=RankedPreservationSelector(InvalidRanker())
    )
    with pytest.raises(CompactionError, match="Ranker returned"):
        await compactor.compact(_request(history, model))

    assert model.calls == []
    assert str(history) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_ranker_failure_does_not_summarize_or_replace_history(
    error_type: type[BaseException],
) -> None:
    history = _history()
    model = SequenceModel([])
    before = str(history)

    class FailedRanker:
        async def rank(self, request: PreservationRequest) -> Sequence[int]:
            await asyncio.sleep(0)
            raise error_type("Ranking interrupted")

    compactor = DefaultCompactor(
        _config(), selector=RankedPreservationSelector(FailedRanker())
    )
    with pytest.raises(error_type, match="Ranking interrupted"):
        await compactor.compact(_request(history, model))

    assert model.calls == []
    assert str(history) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("extend_default", [True, False])
async def test_async_ranking_retains_whole_blocks_before_summary(
    extend_default: bool,
) -> None:
    model = SequenceModel([make_assistant_response("Earlier investigation failed")])
    history = _history()
    ranked = asyncio.Event()

    async def select(request: PreservationRequest) -> PreservationSelection:
        assert model.calls == []
        assert request.instructions[0]["content"] == "Diagnose from evidence"
        assert request.required_messages[0]["content"] == "Find the regression"
        candidates = [
            block for block in request.blocks if block.retainable and not block.required
        ]

        # An external ranking call can await here; it need not handle tool pairing.
        await asyncio.sleep(0)
        winner = next(
            block for block in candidates if "Relevant exception" in str(block.messages)
        )
        assert len(winner.messages) == 2
        assert winner.tokens == _tokens(winner.messages)
        assert request.estimate_tokens(request.default) <= request.token_budget
        base = (
            request.default.block_ids
            if extend_default
            else tuple(block.id for block in request.blocks if block.required)
        )
        ranked.set()

        return PreservationSelection(block_ids=(winner.id, *base), context=())

    result = await DefaultCompactor(_config(), selector=select).compact(
        _request(history, model)
    )

    assert ranked.is_set()
    assert is_summary_item(result.history[0])
    assert result.history[1:] == [history[0], history[2], history[3], history[-1]]
    assert result.dropped_items == (history[1], history[4])
    assert "Relevant exception" in str(model.calls[0])
    assert "Find the regression" in str(model.calls[0])
    assert "Continue diagnosis" not in str(model.calls[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        "unknown",
        "duplicate",
        "required",
        "boolean",
        "summary",
        "context",
        "owner",
        "shape",
    ],
)
async def test_invalid_selection_fails_before_summary(invalid: str) -> None:
    model = SequenceModel([])
    history = [
        render_summary_item(bridge="Continue", summary_text="Previous work"),
        *_history(),
    ]
    original = str(history)

    async def select(request: PreservationRequest) -> PreservationSelection:
        ids = request.default.block_ids
        if invalid == "unknown":
            return PreservationSelection(block_ids=(*ids, 999), context=())

        if invalid == "duplicate":
            return PreservationSelection(block_ids=(*ids, ids[0]), context=())

        if invalid == "required":
            return PreservationSelection(block_ids=(), context=())

        if invalid == "boolean":
            return PreservationSelection(block_ids=(*ids, True), context=())

        if invalid == "summary":
            return PreservationSelection(block_ids=(*ids, 0), context=())

        if invalid == "context":
            return replace(
                request.default, context=(PreservedContext(origin="test", content=""),)
            )

        if invalid == "owner":
            item = PreservedContext(origin="test", content="current state")
            return replace(request.default, context=(item, item))

        return cast(PreservationSelection, object())

    with pytest.raises(CompactionError):
        await DefaultCompactor(_config(), selector=select).compact(
            _request(history, model)
        )
    assert model.calls == []
    assert str(history) == original


@pytest.mark.asyncio
async def test_policy_view_cannot_mutate_canonical_history_or_state() -> None:
    model = SequenceModel([make_assistant_response("summary")])
    history = _history()
    owner = PreservedContext(
        origin="app:state",
        content="Current task state",
        represented_by=({"role": "assistant", "content": "not retained"},),
    )
    request = replace(_request(history, model), preserved_context=(owner,))

    async def select(view: PreservationRequest) -> PreservationSelection:
        view.blocks[0].messages[0]["content"] = "corrupted"
        view.instructions[0]["content"] = "corrupted"
        view.required_messages[0]["content"] = "corrupted"
        view.default.context[0].represented_by[0]["content"] = "corrupted"

        return view.default

    result = await DefaultCompactor(_config(), selector=select).compact(request)
    assert "corrupted" not in str(history)
    assert "corrupted" not in str(result.history)
    assert "corrupted" not in str(model.calls)
    assert request.preserved_context[0].represented_by[0]["content"] == "not retained"


@pytest.mark.asyncio
async def test_context_deduplication_uses_final_selection_and_replaces_previous_state() -> (
    None
):
    history = _history()
    source = render_request_messages(None, [history[2]])[0]
    owner = PreservedContext(
        origin="app:state", content="Exact current state", represented_by=(source,)
    )
    for retain_source in (True, False):
        model = SequenceModel([make_assistant_response("summary")])

        async def select(
            view: PreservationRequest, retain_source: bool = retain_source
        ) -> PreservationSelection:
            ids = view.default.block_ids
            if retain_source:
                source_id = next(
                    block.id for block in view.blocks if source in block.messages
                )
                ids = (*ids, source_id)

            return replace(view.default, block_ids=ids)

        request = replace(_request(history, model), preserved_context=(owner,))
        result = await DefaultCompactor(_config(), selector=select).compact(request)
        projected = [
            item
            for item in result.history
            if is_internal_message(item, origin="agentlane.compaction.context")
        ]
        assert len(projected) == (0 if retain_source else 1)

    # A saved projection is replaced once, even when it follows the recent tail.
    history.append(
        internal_message(
            {
                "role": "user",
                "content": "Preserved context:\n\n### app:state\nStale state",
            },
            origin="agentlane.compaction.context",
        )
    )
    model = SequenceModel([make_assistant_response("summary")])
    result = await DefaultCompactor(_config()).compact(
        replace(_request(history, model), preserved_context=(owner,))
    )
    projected = [
        item
        for item in result.history
        if is_internal_message(item, origin="agentlane.compaction.context")
    ]
    assert len(projected) == 1
    assert isinstance(projected[0], dict)
    content = projected[0]["content"]
    assert isinstance(content, str)
    assert owner.content in content
    assert result.history[-1] == history[-2]


@pytest.mark.asyncio
async def test_selector_can_replace_default_state_and_extract_context() -> None:
    model = SequenceModel([make_assistant_response("summary")])
    owner = PreservedContext(origin="default:state", content="Optional default state")

    async def select(view: PreservationRequest) -> PreservationSelection:
        return replace(
            view.default,
            context=(
                PreservedContext(
                    origin="client:findings",
                    content="Verified error location: service.py:42",
                ),
            ),
        )

    result = await DefaultCompactor(_config(), selector=select).compact(
        replace(
            _request(_history(), model),
            preserved_context=(owner,),
        )
    )
    assert "Optional default state" not in str(result.history)
    assert "Verified error location" in str(result.history)


@pytest.mark.asyncio
async def test_oversized_extracted_context_rejected_before_summary() -> None:
    model = SequenceModel([])

    async def select(view: PreservationRequest) -> PreservationSelection:
        return replace(
            view.default,
            context=(
                PreservedContext(
                    origin="oversize",
                    content="x" * (view.token_budget * 5),
                ),
            ),
        )

    with pytest.raises(ContextOverflowError, match="Preserved context exceeds"):
        await DefaultCompactor(_config(), selector=select).compact(
            _request(_history(), model)
        )
    assert model.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [True, False])
async def test_selector_error_never_calls_summary_or_mutates_history(
    cancel: bool,
) -> None:
    model = SequenceModel([])
    history = _history()
    original = str(history)

    async def select(view: PreservationRequest) -> PreservationSelection:
        del view
        if cancel:
            raise asyncio.CancelledError()

        raise RuntimeError("Ranking service unavailable")

    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await DefaultCompactor(_config(), selector=select).compact(
            _request(history, model)
        )
    assert str(history) == original
    assert model.calls == []


@pytest.mark.asyncio
async def test_candidate_estimate_matches_assembled_context_with_missing_anchor() -> (
    None
):
    history = _history()
    source = render_request_messages(None, [history[2]])[0]
    owner = PreservedContext(
        origin="app:state", content="Exact state " * 20, represented_by=(source,)
    )
    missing = {"role": "user", "content": "Current request outside history"}
    model = SequenceModel([make_assistant_response("summary")])
    estimated: list[int] = []

    async def select(view: PreservationRequest) -> PreservationSelection:
        source_id = next(block.id for block in view.blocks if source in block.messages)
        candidate = replace(
            view.default, block_ids=(*view.default.block_ids, source_id)
        )
        estimated.append(view.estimate_tokens(candidate))

        # Editing the inspection view cannot change canonical costing or output.
        view.blocks[source_id].messages[0]["content"] = "altered " * 1000
        view.required_messages[0]["content"] = "altered"
        assert view.estimate_tokens(candidate) == estimated[0]

        return candidate

    request = replace(
        _request(history, model),
        required_messages=(missing,),
        preserved_context=(owner,),
    )
    result = await DefaultCompactor(_config(), selector=select).compact(request)
    assert estimated == [_tokens(render_request_messages(None, result.history[1:]))]
    assert result.history[1] == missing
    assert "Exact state" not in str(result.history)  # Source block represents it.
    assert "Current request outside history" in str(model.calls[0])


@pytest.mark.asyncio
async def test_interleaved_steering_and_parallel_tool_results_are_one_block() -> None:
    history = _history()
    opener = make_assistant_response(
        None,
        tool_calls=[
            make_tool_call(tool_id="first", name="read_file", arguments='{"path":"a"}'),
            make_tool_call(
                tool_id="second", name="read_file", arguments='{"path":"b"}'
            ),
        ],
    )
    round_trip: list[RunHistoryItem] = [
        opener,
        {"role": "tool", "tool_call_id": "first", "content": "first result"},
        {"role": "user", "content": "New instruction during tools"},
        {"role": "tool", "tool_call_id": "second", "content": "second result"},
    ]
    history[2:4] = round_trip
    model = SequenceModel([make_assistant_response("summary")])

    async def select(view: PreservationRequest) -> PreservationSelection:
        block = next(
            block for block in view.blocks if "second result" in str(block.messages)
        )
        assert len(block.messages) == 4

        return replace(view.default, block_ids=(*view.default.block_ids, block.id))

    result = await DefaultCompactor(_config(), selector=select).compact(
        _request(history, model)
    )
    assert result.history[2:6] == round_trip
    assert all(item not in result.dropped_items for item in round_trip)


@pytest.mark.asyncio
@pytest.mark.parametrize("remove_state", [True, False])
async def test_repeated_extracted_context_is_replaced_or_explicitly_removed(
    remove_state: bool,
) -> None:
    model = SequenceModel([make_assistant_response("summary")] * 2)
    iteration = 0
    estimates: list[int] = []

    async def select(view: PreservationRequest) -> PreservationSelection:
        nonlocal iteration
        iteration += 1
        old = [
            block
            for block in view.blocks
            if any(
                is_internal_message(message, origin="agentlane.compaction.context")
                for message in block.messages
            )
        ]
        if iteration == 2:
            assert len(old) == 1
            assert not old[0].retainable
            assert not old[0].required
        context = (
            ()
            if iteration == 2 and remove_state
            else (
                PreservedContext(
                    origin="client:facts", content=f"Current finding {iteration}"
                ),
                PreservedContext(origin="client:artifact", content="Artifact a.json"),
            )
        )
        selection = replace(view.default, context=context)
        estimates.append(view.estimate_tokens(selection))

        return selection

    compactor = DefaultCompactor(_config(), selector=select)
    first = await compactor.compact(_request(_history(), model))
    envelope = next(
        item
        for item in first.history
        if is_internal_message(item, origin="agentlane.compaction.context")
    )

    # A trailing prior projection must not displace the latest real conversation.
    history = (
        [item for item in first.history if item is not envelope]
        + _history()
        + [envelope]
    )
    result = await compactor.compact(_request(history, model))
    current = [
        item
        for item in result.history
        if is_internal_message(item, origin="agentlane.compaction.context")
    ]
    assert len(current) == (0 if remove_state else 1)
    assert "Current finding 1" not in str(result.history)
    assert ("Current finding 2" in str(result.history)) != remove_state
    assert estimates[-1] == _tokens(render_request_messages(None, result.history[1:]))
