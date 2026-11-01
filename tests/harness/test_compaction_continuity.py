"""Native continuity and final request validation regressions."""

import asyncio
import re
from collections.abc import AsyncIterator, Sequence
from copy import deepcopy
from typing import Any

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
    SUMMARY_CLOSE_TAG,
    SUMMARY_OPEN_TAG,
    CompactionReport,
    CompactionShim,
    CompactionShimConfig,
    ContextOverflowError,
    DefaultCompactor,
    DefaultCompactorConfig,
    PreservationRequest,
    PreservationSelection,
    is_summary_item,
    render_request_messages,
)
from agentlane.harness.shims import BoundShim, PreparedTurn, ShimBindingContext
from agentlane.harness.shims._manager import BoundShimManager
from agentlane.harness.tools import HarnessToolsShim, plan_tool
from agentlane.models import (
    MessageDict,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    SteeringInput,
    SteeringReceipt,
    Tools,
    internal_message,
    is_internal_message,
)
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine
from agentlane.tracing import Span

from .tools_test_utils import (
    SequenceModel,
    echo_tool,
    make_assistant_response,
    make_tool_call,
    named_tool,
)


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
@pytest.mark.parametrize("shape", ["dict", "string", "multipart"])
@pytest.mark.parametrize("boundary_tags", [False, True])
async def test_compaction_preserves_user_requests_quoting_summary_tags(
    shape: str, boundary_tags: bool
) -> None:
    text = f"{SUMMARY_OPEN_TAG} explain these delimiters [t=2] {SUMMARY_CLOSE_TAG}"
    if not boundary_tags:
        text = f"Debug this: {text}. Keep my exact request."
    request: MessageDict = {"role": "user", "content": text}
    if shape == "multipart":
        request["content"] = [{"type": "text", "text": text}]
    original: RunHistoryItem = text if shape == "string" else request
    model = SequenceModel(
        [make_assistant_response("summary [t=1]"), make_assistant_response("done")]
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="quoted-tags", model=model, shims=(_compaction(),)
        )
    )

    result = await agent.run(
        RunState(
            instructions="system",
            history=["Earlier request", original, *_evidence()],
            responses=[],
        )
    )

    assert request in model.calls[-1]
    assert sum(is_summary_item(message) for message in model.calls[-1]) == 1
    assert result.run_state is not None
    assert result.run_state.shim_state["compaction:latest-user"] == request


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


@pytest.mark.asyncio
async def test_required_request_and_tail_leave_nothing_to_summarize() -> None:
    model = SequenceModel([make_assistant_response("summary [t=1]")] * 2)
    reports: list[CompactionReport] = []
    bound = await _compaction(window=90, trigger=70, reports=reports).bind(
        ShimBindingContext(task=_ModelTask(model))
    )
    manager = BoundShimManager((bound,))
    original: list[RunHistoryItem] = [
        {"role": "user", "content": "large request [t=80]"},
        {"role": "assistant", "content": "recent evidence [t=10]"},
    ]
    state = RunState(instructions="system", history=deepcopy(original), responses=[])
    first = PreparedTurn(run_state=state, tools=None, model_args=None)
    await manager.on_run_start(state, first.transient_state)

    # Required blocks are selected first, so this attempt needs no summary call.
    for _ in range(2):
        turn = PreparedTurn(run_state=state, tools=None, model_args=None)
        await manager.prepare_turn(turn)
        messages = await manager.transform_messages(
            turn, render_request_messages(state.instructions, state.history)
        )
        assert state.history == original
        assert _token_estimator(messages) == 90

    assert len(reports) == 2
    assert all(not report.compacted for report in reports)
    assert all(report.estimated_tokens_after == 90 for report in reports)
    assert all(report.error is None for report in reports)
    assert model.calls == []


class _SteeringModel(SequenceModel):
    def __init__(self, delivery: str) -> None:
        super().__init__(
            [
                make_assistant_response("first evidence [t=20]"),
                make_assistant_response(
                    "second evidence [t=30]",
                    tool_calls=[
                        make_tool_call(
                            tool_id="continue",
                            name="echo",
                            arguments='{"text":"continue"}',
                        )
                    ],
                ),
                make_assistant_response("summary [t=1]"),
                make_assistant_response("done"),
            ]
        )
        self.delivery = delivery
        self.steered = asyncio.Event()
        self.correction: str = ""
        self.stream_calls = 0

    async def steer(self, input: SteeringInput) -> SteeringReceipt:
        assert isinstance(input, str)
        self.correction = input
        self.steered.set()

        return SteeringReceipt(
            input=input, previous_response_id="first", status="accepted"
        )

    async def stream_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: Tools | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
        **kwargs: object,
    ) -> AsyncIterator[ModelStreamEvent]:
        response = await self.get_response(
            messages,
            extra_call_args,
            schema,
            tools,
            cancellation_token,
            parent_span,
            **kwargs,
        )
        self.stream_calls += 1
        inputs: list[dict[str, Any]] | None = None

        # Expose the live control channel before waiting for the caller's correction.
        if self.stream_calls == 1:
            yield ModelStreamEvent(kind=ModelStreamEventKind.TEXT_DELTA, text="working")
            await self.steered.wait()
            if self.delivery != "pending":
                inputs = [{"role": "user", "content": self.correction}]
            if self.delivery == "provider":
                yield ModelStreamEvent(
                    kind=ModelStreamEventKind.PROVIDER, input_messages=inputs
                )
                inputs = None

        yield ModelStreamEvent(
            kind=ModelStreamEventKind.COMPLETED,
            response=response,
            input_messages=inputs,
            continuation="automatic" if self.stream_calls == 1 else None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["completed", "provider", "pending"])
@pytest.mark.parametrize("quote_tags", [False, True])
async def test_native_steering_updates_compaction_request_only_after_delivery(
    delivery: str,
    quote_tags: bool,
) -> None:
    model = _SteeringModel(delivery)
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="steered",
            model=model,
            tools=Tools(tools=(echo_tool("echo"),)),
            shims=(_compaction(),),
        )
    )
    initial = {"role": "user", "content": "Build X [t=1]"}
    correction = {"role": "user", "content": "Stop, build Y instead [t=2]"}
    if quote_tags:
        correction["content"] += f" Explain {SUMMARY_OPEN_TAG} and {SUMMARY_CLOSE_TAG}."
    stream = await agent.run_stream(initial["content"])

    async with asyncio.timeout(5):
        await anext(stream)
        await stream.steer(correction["content"])
        async for _ in stream:
            pass
        result = await stream.result()

    expected = initial if delivery == "pending" else correction
    superseded = correction if delivery == "pending" else initial
    assert expected in model.calls[-1]
    assert superseded not in model.calls[-1]
    assert any(is_summary_item(message) for message in model.calls[-1])
    assert result.run_state is not None
    assert result.run_state.shim_state["compaction:latest-user"] == expected


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
    ("alias", "explicit"),
    [
        ("max_tokens", None),
        ("max_tokens", 2048),
        ("max_output_tokens", 2048),
        ("max_completion_tokens", 2048),
    ],
)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("compact_first", [True, False])
@pytest.mark.parametrize("retain_plan_call", [True, False])
async def test_plan_decision_precedes_summary_in_both_shim_orders(
    compact_first: bool,
    retain_plan_call: bool,
) -> None:
    model = SequenceModel([make_assistant_response("summary [t=1]")])
    context = ShimBindingContext(task=_ModelTask(model))
    compaction = await CompactionShim(
        CompactionShimConfig(context_window=100, trigger_tokens=40, on_failure="skip"),
        DefaultCompactor(
            DefaultCompactorConfig(
                keep_recent_tokens=1,
                keep_recent_messages=3 if retain_plan_call else 1,
                summary_max_tokens=2,
            )
        ),
        estimator=_token_estimator,
    ).bind(context)
    tools = await HarnessToolsShim((plan_tool(),)).bind(context)
    sessions = (compaction, tools) if compact_first else (tools, compaction)
    manager = BoundShimManager(sessions)
    plan = {
        "explanation": None,
        "plan": [{"step": "Verify changes", "status": "in_progress"}],
    }
    call = make_assistant_response(
        None,
        tool_calls=[
            make_tool_call(
                tool_id="plan",
                name="write_plan",
                arguments='{"plan":[{"step":"Verify changes","status":"in_progress"}]}',
            )
        ],
    )
    state = RunState(
        instructions="system",
        turn_count=3,
        responses=[],
        history=[
            *_evidence(),
            call,
            {"role": "tool", "tool_call_id": "plan", "content": "Plan updated [t=1]"},
            {"role": "user", "content": "Continue [t=1]"},
        ],
    )
    state.shim_state["harness-tools:plan"] = plan
    turn = PreparedTurn(run_state=state, tools=None, model_args=None)
    await manager.on_run_start(state, turn.transient_state)
    await manager.prepare_turn(turn)
    prepared_history = deepcopy(state.history)
    messages = await manager.transform_messages(
        turn,
        render_request_messages(state.instructions, state.history),
    )

    assert len(model.calls) == 1
    assert state.history == prepared_history
    projections = [
        item
        for item in state.history
        if is_internal_message(item, origin="agentlane.compaction.context")
    ]
    assert len(projections) == (0 if retain_plan_call else 1)
    assert "Verify changes" in str(messages)
    assert (call in state.history) == retain_plan_call
    if projections:
        assert state.history.index(projections[0]) < len(state.history) - 1


@pytest.mark.asyncio
@pytest.mark.parametrize("retain_latest", [False, True])
async def test_ranked_plan_history_keeps_current_state_across_repeated_compaction(
    retain_latest: bool,
) -> None:
    model = SequenceModel([make_assistant_response("summary [t=1]")] * 2)
    context = ShimBindingContext(task=_ModelTask(model))
    history: list[RunHistoryItem] = [*_evidence()]
    for call_id, status in (
        ("first-a", "pending"),
        ("b", "completed"),
        ("last-a", "pending"),
    ):
        history.extend(
            [
                make_assistant_response(
                    None,
                    tool_calls=[
                        make_tool_call(
                            tool_id=call_id,
                            name="write_plan",
                            arguments='{"plan":[{"step":"Verify changes","status":"'
                            + status
                            + '"}]}',
                        )
                    ],
                ),
                {"role": "tool", "tool_call_id": call_id, "content": "Plan updated"},
            ]
        )
    history.append({"role": "user", "content": "Continue [t=1]"})

    # Keep historical evidence, optionally retaining the final correction too.
    async def select(view: PreservationRequest) -> PreservationSelection:
        selected_calls = {"first-a", "b"}
        if retain_latest:
            selected_calls.add("last-a")

        return PreservationSelection(
            block_ids=tuple(
                block.id
                for block in view.blocks
                if block.required
                or any(
                    message.get("tool_call_id") in selected_calls
                    for message in block.messages
                )
            ),
            context=view.default.context,
        )

    compaction = await CompactionShim(
        CompactionShimConfig(context_window=100, trigger_tokens=40, on_failure="skip"),
        DefaultCompactor(
            DefaultCompactorConfig(
                keep_recent_tokens=1, keep_recent_messages=1, summary_max_tokens=2
            ),
            selector=select,
        ),
        estimator=_token_estimator,
    ).bind(context)
    tools = await HarnessToolsShim((plan_tool(),)).bind(context)
    manager = BoundShimManager((tools, compaction))
    state = RunState(instructions="system", history=history, responses=[])
    state.shim_state["harness-tools:plan"] = {
        "explanation": None,
        "plan": [{"step": "Verify changes", "status": "pending"}],
    }
    turn = PreparedTurn(run_state=state, tools=None, model_args=None)
    await manager.on_run_start(state, turn.transient_state)

    # The second attempt must not promote the surviving first A over the later B.
    for attempt in range(2):
        turn = PreparedTurn(run_state=state, tools=None, model_args=None)
        await manager.prepare_turn(turn)
        await manager.transform_messages(
            turn, render_request_messages(state.instructions, state.history)
        )

        projections = [
            message
            for message in render_request_messages(None, state.history)
            if is_internal_message(message, origin="agentlane.compaction.context")
        ]
        assert len(model.calls) == attempt + 1
        assert len(projections) == (0 if retain_latest else 1)
        assert any(
            isinstance(item, dict) and item.get("tool_call_id") == "b"
            for item in state.history
        )
        assert (
            any(
                isinstance(item, dict) and item.get("tool_call_id") == "last-a"
                for item in state.history
            )
            == retain_latest
        )
        if projections:
            assert '"status": "pending"' in str(projections[0]["content"])
            assert "supersedes historical write_plan calls" in str(
                projections[0]["content"]
            )

        state.history.extend(_evidence())


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_run_cancellation_interrupts_compaction_observer(streaming: bool) -> None:
    entered = asyncio.Event()
    closed = asyncio.Event()
    release = asyncio.Event()

    async def observe(report: CompactionReport) -> None:
        assert report.compacted
        entered.set()
        try:
            await release.wait()
        finally:
            closed.set()

    model = SequenceModel([make_assistant_response("summary [t=1]")])
    compaction = CompactionShim(
        CompactionShimConfig(context_window=100, trigger_tokens=40),
        DefaultCompactor(
            DefaultCompactorConfig(
                keep_recent_tokens=1, keep_recent_messages=1, summary_max_tokens=2
            )
        ),
        estimator=_token_estimator,
        on_compact=observe,
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(name="observer", model=model, shims=(compaction,))
    )
    state = RunState(
        instructions="system", history=["request", *_evidence()], responses=[]
    )
    token = CancellationToken()
    stream = (
        await agent.run_stream(state, cancellation_token=token) if streaming else None
    )
    running = asyncio.create_task(
        stream.result()
        if stream is not None
        else agent.run(state, cancellation_token=token)
    )

    try:
        async with asyncio.timeout(5):
            await entered.wait()
            token.cancel()
            expected_error = asyncio.CancelledError if streaming else RuntimeError
            with pytest.raises(expected_error, match=None if streaming else "canceled"):
                await running
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        if stream is not None:
            await stream.aclose()

    assert closed.is_set()
    assert not release.is_set()
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_selector_cancellation_rolls_back_owner_preparation() -> None:
    model = SequenceModel([])

    async def cancel(request: PreservationRequest) -> PreservationSelection:
        del request
        raise asyncio.CancelledError()

    class Owner(BoundShim):
        async def prepare_turn(self, turn: PreparedTurn) -> None:
            turn.append_history_item({"role": "assistant", "content": "prepared"})

    bound = await CompactionShim(
        CompactionShimConfig(context_window=100, trigger_tokens=40),
        DefaultCompactor(
            DefaultCompactorConfig(
                summary_max_tokens=2, keep_recent_messages=1, keep_recent_tokens=1
            ),
            selector=cancel,
        ),
        estimator=_token_estimator,
    ).bind(ShimBindingContext(task=_ModelTask(model)))
    manager = BoundShimManager((bound, Owner()))
    state = RunState(
        instructions="system", history=["request", *_evidence()], responses=[]
    )
    original = deepcopy(state.history)
    turn = PreparedTurn(run_state=state, tools=None, model_args=None)
    await manager.on_run_start(state, turn.transient_state)

    with pytest.raises(asyncio.CancelledError):
        await manager.prepare_turn(turn)
    assert state.history == original
    assert model.calls == []
