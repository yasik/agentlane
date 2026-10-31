"""Native continuity and final request validation regressions."""

import re
from collections.abc import Sequence
from copy import deepcopy

import pytest

from agentlane.harness import (
    AgentDescriptor,
    AgentSnapshot,
    RunHistoryItem,
    RunState,
    Task,
)
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.compaction import (
    CompactionReport,
    CompactionShim,
    CompactionShimConfig,
    ContextOverflowError,
    DefaultCompactor,
    DefaultCompactorConfig,
    is_summary_item,
    render_request_messages,
)
from agentlane.harness.shims import BoundShim, PreparedTurn, ShimBindingContext
from agentlane.harness.shims._manager import BoundShimManager
from agentlane.models import MessageDict, ModelResponse, Tools, internal_message
from agentlane.runtime import SingleThreadedRuntimeEngine

from .tools_test_utils import SequenceModel, make_assistant_response, named_tool


class _ModelTask(Task):
    def __init__(self, model: SequenceModel) -> None:
        super().__init__(SingleThreadedRuntimeEngine())
        self.model = model


def _token_estimator(messages: Sequence[MessageDict]) -> int:
    return sum(
        int(match)
        for message in messages
        for match in re.findall(r"\[t=(\d+)]", str(message.get("content", "")))
    )


def _compaction(
    *,
    window: int = 100,
    trigger: int = 40,
    reports: list[CompactionReport] | None = None,
) -> CompactionShim:
    return CompactionShim(
        CompactionShimConfig(
            context_window=window, trigger_tokens=trigger, on_failure="skip"
        ),
        DefaultCompactor(
            DefaultCompactorConfig(
                keep_recent_tokens=1,
                keep_recent_messages=1,
                summary_max_tokens=2,
            )
        ),
        estimator=_token_estimator,
        on_compact=reports.append if reports is not None else None,
    )


def _evidence() -> list[RunHistoryItem]:
    return [{"role": "assistant", "content": f"evidence {i} [t=10]"} for i in range(5)]


@pytest.mark.asyncio
async def test_repeated_compaction_and_json_restore_keep_latest_human_request() -> None:
    model = SequenceModel(
        [make_assistant_response("handoff [t=1]"), make_assistant_response("done")] * 3
    )
    descriptor = AgentDescriptor(name="continuity", model=model, shims=(_compaction(),))
    state = RunState(
        instructions="Entire system\nKeep every byte [t=1]", history=[], responses=[]
    )
    agent = DefaultAgent(descriptor=descriptor)

    for index in range(3):
        correction = {
            "role": "user",
            "content": f"Correction {index}: keep approval pending [t=2]",
        }
        state.history.extend(
            [
                correction,
                internal_message(
                    {"role": "user", "content": "Internal inventory"}, origin="test"
                ),
                *_evidence(),
            ]
        )
        result = await agent.run(state)
        messages = model.calls[-1]
        assert messages[0] == {"role": "system", "content": state.instructions}
        assert messages.count(correction) == 1
        assert messages.index(correction) < len(messages) - 1
        assert sum(is_summary_item(message) for message in messages) == 1
        assert result.run_state is not None

        snapshot = agent.snapshot()
        assert snapshot is not None
        snapshot = AgentSnapshot.from_json(snapshot.to_json())
        state = snapshot.to_run_state()
        agent = DefaultAgent(descriptor=descriptor, snapshot=snapshot)


@pytest.mark.asyncio
async def test_shared_definition_keeps_helper_assignment_separate() -> None:
    definition = _compaction()
    for assignment in ("root request", "helper request", "root request"):
        model = SequenceModel(
            [make_assistant_response("summary [t=1]"), make_assistant_response("done")]
        )
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name=assignment, model=model, shims=(definition,)
            )
        )
        await agent.run(
            RunState(
                instructions="system", history=[assignment, *_evidence()], responses=[]
            )
        )
        assert {"role": "user", "content": assignment} in model.calls[-1]
        other = "helper request" if assignment == "root request" else "root request"
        assert other not in str(model.calls[-1])


class _InjectedContext(BoundShim):
    async def transform_messages(
        self, turn: PreparedTurn, messages: list[MessageDict]
    ) -> list[MessageDict]:
        turn.append_history_item(
            internal_message(
                {"role": "user", "content": "injected [t=100]"}, origin="test"
            )
        )
        return [*messages, {"role": "user", "content": "injected [t=100]"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("compact_first", [True, False])
async def test_final_validation_rolls_back_compaction_and_transformed_history(
    compact_first: bool,
) -> None:
    model = SequenceModel([make_assistant_response("summary [t=1]")])
    task = _ModelTask(model)
    reports: list[CompactionReport] = []
    bound = await _compaction(reports=reports).bind(ShimBindingContext(task=task))
    injection = _InjectedContext()
    sessions = (bound, injection) if compact_first else (injection, bound)
    manager = BoundShimManager(sessions)
    state = RunState(
        instructions="full system [t=1]",
        history=["request", *_evidence()],
        responses=[],
    )
    original = deepcopy(state.history)
    turn = PreparedTurn(run_state=state, tools=None, model_args=None)
    await manager.on_run_start(state, turn.transient_state)
    await manager.prepare_turn(turn)
    assert any(is_summary_item(item) for item in state.history)

    with pytest.raises(ContextOverflowError, match="Original history was preserved"):
        await manager.transform_messages(
            turn, render_request_messages(state.instructions, state.history)
        )

    assert state.history == original
    assert state.instructions == "full system [t=1]"
    assert len(model.calls) == 1
    assert len(reports) == 1
    assert reports[0].compacted is False
    assert reports[0].error is not None
    assert reports[0].items_after == len(original)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish", ["length", "content_filter", "tool_calls", "function_call", "empty"]
)
async def test_incomplete_summary_keeps_original_history(finish: str) -> None:
    response = make_assistant_response("unfinished [t=1]").model_dump()
    if finish == "empty":
        response["choices"] = []
    else:
        response["choices"][0]["finish_reason"] = finish

    model = SequenceModel(
        [ModelResponse.model_validate(response), make_assistant_response("done")]
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="incomplete", model=model, shims=(_compaction(),)
        )
    )
    original = ["request", *_evidence()]
    result = await agent.run(
        RunState(instructions="system", history=original, responses=[])
    )
    assert result.run_state is not None
    assert result.run_state.history[: len(original)] == original
    assert not any(is_summary_item(item) for item in result.run_state.history)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overhead", ["max_tokens", "max_output_tokens", "max_completion_tokens", "tools"]
)
async def test_overhead_triggers_compaction_below_history_threshold(
    overhead: str,
) -> None:
    model = SequenceModel(
        [make_assistant_response("summary [t=1]"), make_assistant_response("done")]
    )
    tools = Tools(tools=[named_tool("schema [t=40]")]) if overhead == "tools" else None
    model_args = None if overhead == "tools" else {overhead: 40}
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="overhead",
            model=model,
            tools=tools,
            model_args=model_args,
            shims=(_compaction(window=100, trigger=80),),
        )
    )
    await agent.run(
        RunState(instructions="system", history=["request", *_evidence()], responses=[])
    )
    assert len(model.calls) == 2
    assert any(is_summary_item(item) for item in model.calls[-1])


@pytest.mark.asyncio
async def test_ordinary_turns_preserve_request_prefix() -> None:
    model = SequenceModel([make_assistant_response("done")] * 3)
    descriptor = AgentDescriptor(name="prefix", model=model, shims=(_compaction(),))
    agent = DefaultAgent(descriptor=descriptor)
    await agent.run("first request")
    await agent.run("second request")
    snapshot = agent.snapshot()
    assert snapshot is not None
    restored = DefaultAgent(
        descriptor=descriptor, snapshot=AgentSnapshot.from_json(snapshot.to_json())
    )
    await restored.run("third request")

    for previous, current in zip(model.calls, model.calls[1:], strict=False):
        assert current[: len(previous)] == previous


@pytest.mark.asyncio
async def test_repeated_compactions_in_one_invocation_keep_captured_request() -> None:
    model = SequenceModel([make_assistant_response("summary [t=1]")] * 3)
    bound = await _compaction().bind(ShimBindingContext(task=_ModelTask(model)))
    manager = BoundShimManager((bound,))
    request = {"role": "user", "content": "Human correction [t=2]"}
    state = RunState(instructions="system", history=[request], responses=[])
    first = PreparedTurn(run_state=state, tools=None, model_args=None)
    await manager.on_run_start(state, first.transient_state)

    for index in range(3):
        state.history.extend(_evidence())
        state.history.append(
            internal_message(
                {"role": "user", "content": f"Context {index}"}, origin="test"
            )
        )
        turn = PreparedTurn(run_state=state, tools=None, model_args=None)
        await manager.prepare_turn(turn)
        messages = await manager.transform_messages(
            turn, render_request_messages(state.instructions, state.history)
        )
        assert messages.count(request) == 1
        assert sum(is_summary_item(message) for message in messages) == 1

    assert len(model.calls) == 3


class _FailingContext(BoundShim):
    def __init__(self, *, during_prepare: bool) -> None:
        self._during_prepare = during_prepare

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        if self._during_prepare:
            turn.replace_history(["discarded candidate"])
            raise RuntimeError("prepare failed")

    async def transform_messages(
        self, turn: PreparedTurn, messages: list[MessageDict]
    ) -> None:
        del messages
        turn.replace_history(["discarded candidate"])
        raise RuntimeError("transform failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("during_prepare", [True, False])
async def test_preparation_and_transform_errors_restore_original_history(
    during_prepare: bool,
) -> None:
    manager = BoundShimManager((_FailingContext(during_prepare=during_prepare),))
    state = RunState(instructions="system", history=["original"], responses=[])
    turn = PreparedTurn(run_state=state, tools=None, model_args=None)

    with pytest.raises(RuntimeError, match="failed"):
        await manager.prepare_turn(turn)
        await manager.transform_messages(
            turn, render_request_messages(state.instructions, state.history)
        )

    assert state.history == ["original"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "alias", ["max_tokens", "max_output_tokens", "max_completion_tokens"]
)
@pytest.mark.parametrize("explicit", [None, 2048])
async def test_default_actual_output_cap_normalizes_null_aliases(
    alias: str, explicit: int | None
) -> None:
    model = SequenceModel([])
    definition = CompactionShim(
        CompactionShimConfig(context_window=32000, default_max_tokens=8192)
    )
    bound = await definition.bind(ShimBindingContext(task=_ModelTask(model)))
    args: dict[str, object] = {
        key: None
        for key in ("max_tokens", "max_output_tokens", "max_completion_tokens")
    }
    args[alias] = explicit
    turn = PreparedTurn(
        run_state=RunState(instructions="system", history=["request"], responses=[]),
        tools=None,
        model_args=args,
    )

    await bound.prepare_turn(turn)

    assert turn.model_args == ({alias: explicit} if explicit else {"max_tokens": 8192})
    assert model.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "alias", ["max_tokens", "max_output_tokens", "max_completion_tokens"]
)
@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "100"])
async def test_malformed_output_caps_fail_before_summary(
    alias: str, value: object
) -> None:
    model = SequenceModel([])
    bound = await _compaction().bind(ShimBindingContext(task=_ModelTask(model)))
    state = RunState(
        instructions="system", history=["request", *_evidence()], responses=[]
    )
    turn = PreparedTurn(run_state=state, tools=None, model_args={alias: value})
    with pytest.raises(ValueError, match="positive integer"):
        await bound.prepare_turn(turn)
    assert model.calls == []
