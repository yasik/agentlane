import re
from collections.abc import Sequence
from typing import Any

import pytest

from agentlane.harness import AgentDescriptor, RunHistoryItem, RunState, Task
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.compaction import (
    CompactionError,
    CompactionReport,
    CompactionRequest,
    CompactionResult,
    CompactionShim,
    CompactionShimConfig,
    is_summary_item,
    render_summary_item,
)
from agentlane.harness.shims import BoundShim, PreparedTurn, ShimBindingContext
from agentlane.models import MessageDict, Model, ModelResponse
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine
from agentlane.tracing import Span

from .tools_test_utils import SequenceModel, make_assistant_response

_TOKEN_PATTERN = re.compile(r"\[t=(\d+)]")


class _ModelTask(Task):
    def __init__(self, model: Model[ModelResponse] | None) -> None:
        super().__init__(SingleThreadedRuntimeEngine())
        self._model = model

    @property
    def model(self) -> Model[ModelResponse] | None:
        return self._model


class _RecordingCompactor:
    def __init__(
        self,
        result: CompactionResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.requests: list[CompactionRequest] = []

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise AssertionError("Expected a queued compaction result.")
        return self._result


class _UnusedModel(SequenceModel):
    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: object | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        raise AssertionError("Compaction shim tests should not call the task model.")


def _token_estimator(messages: Sequence[MessageDict]) -> int:
    return sum(
        int(match)
        for message in messages
        for match in _TOKEN_PATTERN.findall(str(message.get("content", "")))
    )


def _turn(history: list[RunHistoryItem]) -> PreparedTurn:
    return PreparedTurn(
        run_state=RunState(
            instructions="Base instruction [t=1]",
            history=history,
            responses=[],
            turn_count=3,
        ),
        tools=None,
        model_args={"temperature": 0.2},
    )


def _response_with_usage() -> ModelResponse:
    return ModelResponse.model_validate(
        {
            "id": "chatcmpl_usage",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "summary",
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
            },
        }
    )


def _response_without_choices() -> ModelResponse:
    return ModelResponse.model_validate(
        {
            "id": "chatcmpl_empty",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [],
        }
    )


async def _bound_shim(
    shim: CompactionShim,
    model: Model[ModelResponse] | None = None,
) -> BoundShim:
    task = _ModelTask(model or _UnusedModel([]))
    return await shim.bind(ShimBindingContext(task=task))


@pytest.mark.asyncio
async def test_compaction_shim_skips_when_request_is_below_trigger() -> None:
    compactor = _RecordingCompactor()
    reports: list[CompactionReport] = []
    shim = CompactionShim(
        CompactionShimConfig(context_window=100, trigger_tokens=10),
        compactor,
        estimator=_token_estimator,
        on_compact=reports.append,
    )
    turn = _turn([{"role": "user", "content": "short history [t=3]"}])
    bound = await _bound_shim(shim)

    await bound.prepare_turn(turn)

    assert compactor.requests == []
    assert reports == []
    assert turn.run_state.history == [
        {"role": "user", "content": "short history [t=3]"}
    ]


@pytest.mark.asyncio
async def test_compaction_shim_installs_result_and_reports_attempt() -> None:
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "old [t=4]"},
        {"role": "assistant", "content": "recent [t=4]"},
    ]
    summary = render_summary_item(
        bridge="Continue.",
        summary_text="summary [t=1]",
    )
    summarizer_response = _response_with_usage()
    compactor = _RecordingCompactor(
        CompactionResult(
            history=[summary, history[-1]],
            summary_content=str(summary["content"]),
            dropped_items=(history[0],),
            summarizer_response=summarizer_response,
        )
    )
    reports: list[CompactionReport] = []

    async def on_compact(report: CompactionReport) -> None:
        reports.append(report)

    shim = CompactionShim(
        CompactionShimConfig(context_window=20, trigger_tokens=5),
        compactor,
        estimator=_token_estimator,
        on_compact=on_compact,
    )
    turn = _turn(history)
    bound = await _bound_shim(shim)

    await bound.prepare_turn(turn)

    assert turn.run_state.history == [summary, history[-1]]
    request = compactor.requests[0]
    assert request.history == tuple(history)
    assert request.model_args == {"temperature": 0.2}
    assert request.signal.estimated_tokens == 9
    assert request.signal.instructions_tokens == 1
    assert request.signal.trigger_tokens == 5
    assert request.signal.source == "estimate"

    report = reports[0]
    assert report.compacted is True
    assert report.items_before == 2
    assert report.items_after == 2
    assert report.estimated_tokens_after == 6
    assert report.summary_chars == len(str(summary["content"]))
    assert report.dropped_items == (history[0],)
    assert report.summarizer_usage == (11, 3, 14)
    assert report.attempt_key == "compaction:auto:turn=3:history=2:tokens=9"
    assert report.error is None


@pytest.mark.asyncio
async def test_compaction_shim_uses_reported_usage_when_it_exceeds_estimate() -> None:
    summary = render_summary_item(
        bridge="Continue.",
        summary_text="summary [t=1]",
    )
    compactor = _RecordingCompactor(
        CompactionResult(
            history=[summary],
            summary_content=str(summary["content"]),
            dropped_items=(),
            summarizer_response=None,
        )
    )
    shim = CompactionShim(
        CompactionShimConfig(context_window=100, trigger_tokens=10),
        compactor,
        estimator=_token_estimator,
    )
    turn = _turn([{"role": "user", "content": "short history [t=3]"}])
    turn.run_state.responses.append(_response_with_usage())
    turn.run_state.history = [{"role": "user", "content": "local threshold [t=10]"}]
    bound = await _bound_shim(shim)

    await bound.prepare_turn(turn)

    signal = compactor.requests[0].signal
    assert signal.reported_tokens == 14
    assert signal.estimated_tokens == 14
    assert signal.source == "server_usage"


@pytest.mark.asyncio
async def test_compaction_shim_does_not_trigger_from_reported_usage_alone() -> None:
    compactor = _RecordingCompactor()
    shim = CompactionShim(
        CompactionShimConfig(context_window=100, trigger_tokens=10),
        compactor,
        estimator=_token_estimator,
    )
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "small child context [t=3]"}
    ]
    turn = _turn(history)
    turn.run_state.responses.append(_response_with_usage())
    bound = await _bound_shim(shim)

    await bound.prepare_turn(turn)

    assert compactor.requests == []
    assert turn.run_state.history == history


@pytest.mark.asyncio
async def test_compaction_shim_uses_local_estimate_when_reported_usage_is_lower() -> (
    None
):
    summary = render_summary_item(
        bridge="Continue.",
        summary_text="summary [t=1]",
    )
    compactor = _RecordingCompactor(
        CompactionResult(
            history=[summary],
            summary_content=str(summary["content"]),
            dropped_items=(),
            summarizer_response=None,
        )
    )
    shim = CompactionShim(
        CompactionShimConfig(context_window=100, trigger_tokens=10),
        compactor,
        estimator=_token_estimator,
    )
    turn = _turn([{"role": "user", "content": "larger local estimate [t=20]"}])
    turn.run_state.responses.append(_response_with_usage())
    bound = await _bound_shim(shim)

    await bound.prepare_turn(turn)

    signal = compactor.requests[0].signal
    assert signal.reported_tokens == 14
    assert signal.estimated_tokens == 21
    assert signal.source == "mixed"


@pytest.mark.asyncio
async def test_compaction_shim_runs_default_compactor_when_no_compactor_is_supplied() -> (
    None
):
    history: list[RunHistoryItem] = [
        {"role": "user", "content": "old [t=4]"},
        {"role": "assistant", "content": "middle [t=4]"},
        {"role": "user", "content": "recent question [t=4]"},
        {"role": "assistant", "content": "recent answer [t=4]"},
        {"role": "user", "content": "latest [t=4]"},
    ]
    model = SequenceModel([make_assistant_response("summary [t=1]")])
    shim = CompactionShim(
        CompactionShimConfig(context_window=10_000, trigger_tokens=5),
        estimator=_token_estimator,
    )
    turn = _turn(history)
    bound = await _bound_shim(shim, model)

    await bound.prepare_turn(turn)

    assert len(model.calls) == 1
    assert model.calls[0][1]["content"] == "Base instruction [t=1]"
    assert model.calls[0][2:] == history[:1]
    assert is_summary_item(turn.run_state.history[0])
    assert turn.run_state.history[1:] == history[-4:]


@pytest.mark.asyncio
async def test_compaction_shim_skips_failed_compaction_when_configured() -> None:
    compactor = _RecordingCompactor(error=CompactionError("summary failed"))
    reports: list[CompactionReport] = []
    shim = CompactionShim(
        CompactionShimConfig(
            context_window=20,
            trigger_tokens=5,
            on_failure="skip",
        ),
        compactor,
        estimator=_token_estimator,
        on_compact=reports.append,
    )
    history: list[RunHistoryItem] = [{"role": "user", "content": "old [t=8]"}]
    turn = _turn(history)
    bound = await _bound_shim(shim)

    await bound.prepare_turn(turn)

    assert turn.run_state.history == history
    assert reports[0].compacted is False
    assert reports[0].error == "summary failed"
    assert (
        await bound.transform_messages(
            turn,
            [{"role": "user", "content": "next prompt"}],
        )
        is None
    )


@pytest.mark.asyncio
async def test_compaction_shim_skips_compactor_failure_when_observer_fails() -> None:
    compactor = _RecordingCompactor(error=CompactionError("summary failed"))
    reports: list[CompactionReport] = []

    def on_compact(report: CompactionReport) -> None:
        reports.append(report)
        raise RuntimeError("observer down")

    shim = CompactionShim(
        CompactionShimConfig(
            context_window=20,
            trigger_tokens=5,
            on_failure="skip",
        ),
        compactor,
        estimator=_token_estimator,
        on_compact=on_compact,
    )
    history: list[RunHistoryItem] = [{"role": "user", "content": "old [t=8]"}]
    turn = _turn(history)
    bound = await _bound_shim(shim)

    await bound.prepare_turn(turn)

    assert turn.run_state.history == history
    assert reports[0].compacted is False
    assert reports[0].error == "summary failed"


@pytest.mark.asyncio
async def test_compaction_shim_injects_failed_compaction_notice_by_default() -> None:
    compactor = _RecordingCompactor(error=CompactionError("summary failed"))
    reports: list[CompactionReport] = []
    shim = CompactionShim(
        CompactionShimConfig(context_window=20, trigger_tokens=5),
        compactor,
        estimator=_token_estimator,
        on_compact=reports.append,
    )
    turn = _turn([{"role": "user", "content": "old [t=8]"}])
    bound = await _bound_shim(shim)
    messages: list[MessageDict] = [{"role": "user", "content": "next prompt"}]

    await bound.prepare_turn(turn)
    transformed = await bound.transform_messages(turn, messages)

    assert turn.run_state.history == [{"role": "user", "content": "old [t=8]"}]
    assert reports[0].compacted is False
    assert reports[0].error == "summary failed"
    assert transformed is not None
    assert transformed[:-1] == messages
    assert transformed[-1]["role"] == "user"
    assert "Context compaction failed" in str(transformed[-1]["content"])
    assert "summary failed" in str(transformed[-1]["content"])
    assert await bound.transform_messages(turn, messages) is None


@pytest.mark.asyncio
async def test_compaction_shim_reports_noop_result_without_replacing_history() -> None:
    history: list[RunHistoryItem] = [{"role": "user", "content": "old [t=8]"}]
    compactor = _RecordingCompactor(
        CompactionResult(
            history=list(history),
            summary_content="",
            dropped_items=(),
            summarizer_response=make_assistant_response("unused"),
        )
    )
    reports: list[CompactionReport] = []
    shim = CompactionShim(
        CompactionShimConfig(context_window=20, trigger_tokens=5),
        compactor,
        estimator=_token_estimator,
        on_compact=reports.append,
    )
    turn = _turn(history)
    bound = await _bound_shim(shim)

    await bound.prepare_turn(turn)

    assert turn.run_state.history == history
    assert reports[0].compacted is False
    assert reports[0].items_before == 1
    assert reports[0].items_after == 1


@pytest.mark.asyncio
async def test_compaction_shim_keeps_successful_replacement_when_observer_fails() -> (
    None
):
    history: list[RunHistoryItem] = [{"role": "user", "content": "old [t=8]"}]
    summary = render_summary_item(
        bridge="Continue.",
        summary_text="summary [t=1]",
    )
    compactor = _RecordingCompactor(
        CompactionResult(
            history=[summary],
            summary_content=str(summary["content"]),
            dropped_items=(history[0],),
            summarizer_response=None,
        )
    )
    reports: list[CompactionReport] = []

    async def on_compact(report: CompactionReport) -> None:
        reports.append(report)
        raise RuntimeError("observer down")

    shim = CompactionShim(
        CompactionShimConfig(context_window=20, trigger_tokens=5),
        compactor,
        estimator=_token_estimator,
        on_compact=on_compact,
    )
    turn = _turn(history)
    bound = await _bound_shim(shim)

    await bound.prepare_turn(turn)

    assert turn.run_state.history == [summary]
    assert reports[0].compacted is True


@pytest.mark.asyncio
async def test_compaction_shim_skips_malformed_compactor_result_when_configured() -> (
    None
):
    history: list[RunHistoryItem] = [{"role": "user", "content": "old [t=8]"}]
    compactor = _RecordingCompactor(
        CompactionResult(
            history=[_response_without_choices()],
            summary_content="bad replacement",
            dropped_items=(history[0],),
            summarizer_response=None,
        )
    )
    reports: list[CompactionReport] = []
    shim = CompactionShim(
        CompactionShimConfig(
            context_window=20,
            trigger_tokens=5,
            on_failure="skip",
        ),
        compactor,
        estimator=_token_estimator,
        on_compact=reports.append,
    )
    turn = _turn(history)
    bound = await _bound_shim(shim)

    await bound.prepare_turn(turn)

    assert turn.run_state.history == history
    assert reports[0].compacted is False
    assert reports[0].error is not None


@pytest.mark.asyncio
async def test_compaction_shim_requires_model_backed_task() -> None:
    shim = CompactionShim(CompactionShimConfig(context_window=20, trigger_tokens=5))
    task = Task(SingleThreadedRuntimeEngine())

    with pytest.raises(TypeError, match="task with a model"):
        await shim.bind(ShimBindingContext(task=task))


@pytest.mark.asyncio
async def test_compaction_shim_requires_configured_model() -> None:
    shim = CompactionShim(CompactionShimConfig(context_window=20, trigger_tokens=5))
    task = _ModelTask(None)

    with pytest.raises(TypeError, match="configured model"):
        await shim.bind(ShimBindingContext(task=task))


@pytest.mark.asyncio
async def test_compaction_shim_runs_inside_default_agent_lifecycle() -> None:
    summary = render_summary_item(
        bridge="Continue.",
        summary_text="summary [t=1]",
    )
    compactor = _RecordingCompactor(
        CompactionResult(
            history=[summary],
            summary_content=str(summary["content"]),
            dropped_items=("old prompt [t=8]",),
            summarizer_response=None,
        )
    )
    model = SequenceModel([make_assistant_response("done")])
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="CompactingAgent",
            model=model,
            shims=(
                CompactionShim(
                    CompactionShimConfig(context_window=20, trigger_tokens=5),
                    compactor,
                    estimator=_token_estimator,
                ),
            ),
        )
    )

    result = await agent.run("old prompt [t=8]")

    assert result.final_output == "done"
    assert compactor.requests[0].history == ("old prompt [t=8]",)
    assert model.calls[0] == [summary]


@pytest.mark.asyncio
async def test_compaction_shim_failure_notice_runs_inside_default_agent_lifecycle() -> (
    None
):
    compactor = _RecordingCompactor(error=CompactionError("summary failed"))
    model = SequenceModel([make_assistant_response("done")])
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="CompactingAgent",
            model=model,
            shims=(
                CompactionShim(
                    CompactionShimConfig(context_window=20, trigger_tokens=5),
                    compactor,
                    estimator=_token_estimator,
                ),
            ),
        )
    )

    result = await agent.run("old prompt [t=8]")

    assert result.final_output == "done"
    assert compactor.requests[0].history == ("old prompt [t=8]",)
    assert model.calls[0][0] == {"role": "user", "content": "old prompt [t=8]"}
    assert model.calls[0][-1]["role"] == "user"
    assert "Context compaction failed" in str(model.calls[0][-1]["content"])
    assert "summary failed" in str(model.calls[0][-1]["content"])
