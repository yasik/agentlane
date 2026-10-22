"""Host-owned continuation round trips and execution boundaries."""

import asyncio
import gc
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Self, cast
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, Field

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    DefaultHandoff,
    ExternalTool,
    RunContinuation,
    Runner,
    RunPaused,
    RunResult,
    RunState,
    ToolResultInput,
)
from agentlane.harness._lifecycle import DefaultAgentTool
from agentlane.messaging import AgentId, DeliveryStatus
from agentlane.models import (
    Model,
    ModelBehaviorError,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    Tools,
)
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine

from .tools_test_utils import (
    EchoArgs,
    SequenceModel,
    make_assistant_response,
    make_tool_call,
)

ROOT = AgentId.from_values("test", "root")


class ChangedArgs(BaseModel):
    city: str


def external() -> ExternalTool[EchoArgs]:
    return ExternalTool(
        name="external", description="Get host input.", args_model=EchoArgs
    )


def response(
    *names: str, call_suffix: str = "", arguments: str = '{"text":"needed"}'
) -> ModelResponse:
    calls = [
        make_tool_call(tool_id=name + call_suffix, name=name, arguments=arguments)
        for name in names
    ]

    return make_assistant_response(None if calls else "done", tool_calls=calls or None)


class ScriptedModel(SequenceModel):
    def __init__(self, *responses: ModelResponse, cleanup_error: bool = False) -> None:
        super().__init__(list(responses))
        self.closed = False
        self.cleanup_error = cleanup_error

    @asynccontextmanager
    async def open_run(self) -> AsyncIterator[Self]:
        try:
            yield self
        finally:
            self.closed = True
            if self.cleanup_error:
                raise RuntimeError("cleanup failed")


def agent(
    model: Model[ModelResponse],
    runner: Runner,
    tools: Tools | None = None,
    default_handoff: DefaultHandoff | None = None,
) -> Agent:
    return Agent(
        SingleThreadedRuntimeEngine(),
        runner,
        bind_id=ROOT,
        descriptor=AgentDescriptor(
            name="test",
            model=model,
            tools=tools or Tools(tools=[external()]),
            default_handoff=default_handoff,
        ),
    )


def state() -> RunState:
    return RunState(instructions=None, history=["start"], responses=[])


async def execute(
    runner: Runner,
    model: ScriptedModel,
    run_state: RunState | ToolResultInput,
    tools: Tools | None = None,
    mode: str = "run",
    *,
    default_handoff: DefaultHandoff | None = None,
    cancellation_token: CancellationToken | None = None,
) -> RunResult:
    task = agent(model, runner, tools, default_handoff)

    if mode == "run":
        return await runner.run(task, run_state, cancellation_token=cancellation_token)

    stream = (
        runner.run_events(task, run_state, cancellation_token=cancellation_token)
        if mode == "events"
        else runner.run_stream(task, run_state, cancellation_token=cancellation_token)
    )

    try:
        return await stream.result()
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
async def test_restart_retains_effects_calls_and_saved_budgets(mode: str) -> None:
    effects: list[str] = []

    async def write(text: str) -> str:
        """Record a side effect."""
        effects.append(text)

        return "written"

    tools = Tools(
        tools=[write, external()],
        tool_call_limits={"write": 1, "external": 2},
        max_tool_round_trips=3,
    )
    first = Runner(max_turns=3)
    model = ScriptedModel(response("write"), response("external"))
    with pytest.raises(RunPaused) as paused:
        await execute(first, model, state(), tools, mode)
    saved = RunContinuation.from_json(
        json.loads(json.dumps(paused.value.continuation.to_json()))
    )
    assert model.closed and len(model.calls) == 2
    assert saved.checkpoint.state.turn_count == 2 and saved.checkpoint.revision == 0
    assert saved.tool_call_counts == {"write": 1, "external": 1}
    original = saved.to_json()

    # A new execution can lower limits, but cannot expand the saved budget.
    resumed = Runner(max_turns=999)
    next_model = ScriptedModel(response("external", call_suffix="_again"))
    expanded_tools = Tools(
        tools=[write, external()],
        tool_call_limits={"write": 0, "external": 999},
        max_tool_round_trips=999,
    )
    with pytest.raises(RunPaused) as paused_again:
        await execute(
            resumed,
            next_model,
            ToolResultInput(continuation=saved, result="Prague"),
            expanded_tools,
            mode,
        )
    again = RunContinuation.from_json(paused_again.value.continuation.to_json())
    assert again.tool_call_counts == {"write": 1, "external": 2}
    assert again.tool_call_limits == {"write": 0, "external": 2}
    assert again.max_turns == again.tool_round_trips == 3
    assert again.max_tool_round_trips == 3
    assert effects == ["needed"] and saved.to_json() == original
    assert [m for m in next_model.calls[0] if m["role"] == "user"] == [
        {"role": "user", "content": "start"}
    ]
    assert [
        (m["tool_call_id"], m["content"])
        for m in next_model.calls[0]
        if m["role"] == "tool"
    ] == [("write", "written"), ("external", "Prague")]
    catalog = next_model.call_tools[0]
    assert catalog is not None
    assert [tool.name for tool in catalog.normalized_tools] == ["external"]
    assert catalog.max_tool_round_trips == 3
    assert catalog.tool_call_limits == {"write": 0, "external": 2}

    # A second answer cannot grant another model turn after the saved limit.
    last = Runner(max_turns=999)
    final_model = ScriptedModel()
    with pytest.raises(RuntimeError, match="turn limit"):
        await execute(
            last,
            final_model,
            ToolResultInput(continuation=again, result="answer"),
            expanded_tools,
            mode,
        )
    assert final_model.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
@pytest.mark.parametrize("max_turns,current_limit", [(2, 999), (3, 999), (3, 2)])
async def test_resume_handoff_retains_saved_turn_limit(
    mode: str, max_turns: int, current_limit: int
) -> None:
    with pytest.raises(RunPaused) as paused:
        await execute(
            Runner(max_turns=max_turns),
            ScriptedModel(response("external")),
            state(),
            mode=mode,
        )
    saved = RunContinuation.from_json(paused.value.continuation.to_json())
    child = ScriptedModel(response())
    resumed = Runner(max_turns=current_limit)
    execution = execute(
        resumed,
        ScriptedModel(response("handoff", arguments="{}")),
        ToolResultInput(continuation=saved, result="answer"),
        mode=mode,
        default_handoff=DefaultHandoff(model=child, tools=None),
    )

    # The same handoff succeeds only when the saved budget permits its third turn.
    if min(max_turns, current_limit) == 2:
        with pytest.raises(RuntimeError, match="turn limit"):
            await execution
        assert child.calls == []
    else:
        assert (await execution).final_output == "done"
        assert len(child.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
async def test_resume_preserves_external_argument_defaults(mode: str) -> None:
    factory = Mock(side_effect=uuid4)

    class DefaultArgs(BaseModel):
        request_id: UUID = Field(default_factory=factory, alias="requestId")

    tools = Tools(
        tools=[
            ExternalTool(
                name="external",
                description="Get host input.",
                args_model=DefaultArgs,
                strict=False,
            )
        ]
    )
    with pytest.raises(RunPaused) as paused:
        await execute(
            Runner(),
            ScriptedModel(response("external", arguments="{}")),
            state(),
            tools,
            mode,
        )
    saved = RunContinuation.from_json(
        json.loads(json.dumps(paused.value.continuation.to_json()))
    )
    original = saved.to_json()
    model = ScriptedModel(response())
    result = await execute(
        Runner(),
        model,
        ToolResultInput(continuation=saved, result="answer"),
        tools,
        mode,
    )
    assert result.final_output == "done"
    assert factory.call_count == 1 and saved.to_json() == original
    assert saved.tool_call.function.arguments == "{}"
    assert UUID(str(saved.arguments["request_id"]))
    assert [message for message in model.calls[0] if message["role"] == "user"] == [
        {"role": "user", "content": "start"}
    ]
    assert model.calls[0][-1] == {
        "role": "tool",
        "tool_call_id": saved.tool_call.id,
        "name": "external",
        "content": "answer",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["schema", "identity", "pending"])
async def test_invalid_resume_rejects_before_model_io(invalid: str) -> None:
    runner = Runner()
    with pytest.raises(RunPaused) as paused:
        await execute(runner, ScriptedModel(response("external")), state())
    saved = paused.value.continuation.model_copy(deep=True)
    tools = Tools(tools=[external()])
    if invalid == "schema":
        tools = Tools(
            tools=[
                ExternalTool(
                    name="external", description="changed", args_model=ChangedArgs
                )
            ]
        )
    elif invalid == "identity":
        saved.checkpoint.agent_id = AgentId.from_values("test", "other")
    else:
        saved.checkpoint.state.history.append(
            {"role": "user", "content": "wrong boundary"}
        )
    model = ScriptedModel()
    with pytest.raises(ValueError):
        await execute(
            Runner(), model, ToolResultInput(continuation=saved, result="answer"), tools
        )
    assert model.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["mixed", "cleanup"])
async def test_unsafe_boundary_never_publishes_pause(failure: str) -> None:
    effects: list[str] = []

    async def write(text: str) -> str:
        """Record a side effect."""
        effects.append(text)

        return text

    runner = Runner()
    calls = ("external", "write") if failure == "mixed" else ("external",)
    model = ScriptedModel(response(*calls), cleanup_error=failure == "cleanup")
    expected = {
        "mixed": ModelBehaviorError,
        "cleanup": ExceptionGroup,
    }
    with pytest.raises(expected[failure]):
        await execute(runner, model, state(), Tools(tools=[write, external()]))
    assert effects == []
    assert model.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
async def test_cancelled_input_does_not_consume_continuation_but_execution_does(
    mode: str,
) -> None:
    runner = Runner()
    with pytest.raises(RunPaused) as paused:
        await execute(runner, ScriptedModel(response("external")), state())
    result_input = ToolResultInput(paused.value.continuation, "answer")
    model = ScriptedModel(response())
    token = CancellationToken()
    token.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execute(runner, model, result_input, mode=mode, cancellation_token=token)
    assert (
        await execute(runner, model, result_input, mode=mode)
    ).final_output == "done"
    with pytest.raises(RuntimeError):
        await execute(runner, model, result_input)
    assert len(model.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("max_turns", False),
        ("tool_round_trips", -1),
        ("format_version", True),
        ("continuation_id", None),
        ("checkpoint.revision", -1),
        ("checkpoint.schema_version", True),
        ("checkpoint.state.turn_count", True),
        ("checkpoint.state.shim_state", {"bad": float("nan")}),
        ("unexpected", 1),
    ],
)
async def test_codec_rejects_invalid_counters_and_owns_copies(
    field: str, value: object
) -> None:
    with pytest.raises(RunPaused) as paused:
        await execute(
            Runner(),
            ScriptedModel(response("external")),
            state(),
        )
    saved = paused.value.continuation
    data = saved.to_json()
    result_input = ToolResultInput(continuation=saved, result="answer")
    runner = Runner()
    stream = runner.run_events(agent(ScriptedModel(response()), runner), result_input)
    try:
        # Neither caller can change the payload already owned by the runner.
        saved.arguments["text"] = "first mutation"
        result_input.continuation.arguments["text"] = "second mutation"
        result_input.continuation.checkpoint.state.history.clear()
        assert (await stream.result()).final_output == "done"
    finally:
        await stream.aclose()
    target = data
    *parents, name = field.split(".")
    for parent in parents:
        target = cast(dict[str, object], target[parent])
    if value is None:
        target.pop(name)
    else:
        target[name] = value
    with pytest.raises(ValueError):
        RunContinuation.from_json(data)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["automatic", "explicit", "steering", "async", "session"]
)
async def test_unsupported_provider_controls_reject_before_effects(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    effects: list[str] = []

    async def write(text: str) -> str:
        """Record a side effect."""
        effects.append(text)

        return text

    model = ScriptedModel()
    call = make_tool_call(tool_id="effect", name="write", arguments='{"text":"needed"}')
    event = ModelStreamEvent(
        kind=ModelStreamEventKind.COMPLETED,
        response=response("write"),
    )
    if mode in ("automatic", "explicit"):
        event.continuation = "automatic" if mode == "automatic" else "explicit"
    elif mode == "steering":
        event.input_messages = [{"role": "user", "content": "live input"}]
    elif mode == "async":
        event.async_tool_call = call
    elif mode == "session":
        monkeypatch.setattr(
            ScriptedModel, "requires_streaming", property(lambda self: True)
        )

    async def stream_response(
        *args: object, **kwargs: object
    ) -> AsyncIterator[ModelStreamEvent]:
        yield event

    monkeypatch.setattr(model, "stream_response", stream_response)
    expected = ValueError if mode == "session" else ModelBehaviorError
    with pytest.raises(expected):
        await execute(
            Runner(),
            model,
            state(),
            Tools(tools=[write, external()]),
            "stream",
        )
    assert effects == []
    assert model.closed == (mode != "session")


@pytest.mark.asyncio
async def test_stream_reserves_continuation_before_concurrent_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Runner()
    with pytest.raises(RunPaused) as paused:
        await execute(runner, ScriptedModel(response("external")), state())
    result_input = ToolResultInput(paused.value.continuation, "answer")
    entered, release = asyncio.Event(), asyncio.Event()
    model = ScriptedModel()

    async def blocked_response(*args: object, **kwargs: object) -> ModelResponse:
        entered.set()
        await release.wait()

        return response()

    monkeypatch.setattr(model, "get_response", blocked_response)
    stream = runner.run_events(agent(model, runner), result_input)
    try:
        # Reserve the continuation before the stream task starts.
        with pytest.raises(RuntimeError):
            await runner.run(agent(model, runner), result_input)
        await asyncio.wait_for(entered.wait(), timeout=1)
        release.set()
        assert (await stream.result()).final_output == "done"
    finally:
        release.set()
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
async def test_resume_turn_limit_does_not_change_runner_limit(mode: str) -> None:
    with pytest.raises(RunPaused) as paused:
        await execute(
            Runner(max_turns=1), ScriptedModel(response("external")), state(), mode=mode
        )
    runner = Runner(max_turns=3)
    with pytest.raises(RuntimeError, match="turn limit"):
        await execute(
            runner,
            ScriptedModel(),
            ToolResultInput(paused.value.continuation, "answer"),
            mode=mode,
        )

    async def write(text: str) -> str:
        """Return the supplied text."""
        return text

    model = ScriptedModel(response("write"), response("write"), response())
    result = await execute(runner, model, state(), Tools(tools=[write]), mode)
    assert result.final_output == "done"
    assert len(model.calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
async def test_handoff_cannot_pause_for_external_tool(mode: str) -> None:
    child = ScriptedModel(response("external"))
    expected = RuntimeError if mode == "run" else ValueError
    with pytest.raises(expected, match="child suspension"):
        await execute(
            Runner(),
            ScriptedModel(response("handoff", arguments="{}")),
            state(),
            mode=mode,
            default_handoff=DefaultHandoff(
                model=child, tools=Tools(tools=[external()])
            ),
        )
    assert child.calls == []


@pytest.mark.asyncio
async def test_runtime_resume_commits_current_revision_once() -> None:
    runtime = SingleThreadedRuntimeEngine()
    baseline = RunState(instructions=None, history=["prior"], responses=[], revision=7)
    model = ScriptedModel(response("external"), response())
    task = Agent.bind(
        runtime,
        ROOT,
        runner=Runner(),
        descriptor=AgentDescriptor(
            name="test", model=model, tools=Tools(tools=[external()])
        ),
        run_state=baseline,
    )
    gc.collect()
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled: list[dict[str, Any]] = []

    def record_exception(
        loop: asyncio.AbstractEventLoop, context: dict[str, Any]
    ) -> None:
        del loop
        unhandled.append(context)

    loop.set_exception_handler(record_exception)
    try:
        paused = await runtime.send_message("start", recipient=ROOT)
        assert paused.status == DeliveryStatus.DELIVERED
        assert isinstance(paused.response_payload, RunPaused)
        saved = paused.response_payload.continuation
        assert saved.checkpoint.revision == 7
        assert task.run_state == baseline

        result_input = ToolResultInput(saved, "Prague")
        resumed = await runtime.send_message(result_input, recipient=ROOT)
        assert resumed.status == DeliveryStatus.DELIVERED
        assert isinstance(resumed.response_payload, RunResult)
        assert resumed.response_payload.final_output == "done"
        assert task.run_state is not None and task.run_state.revision == 8
        assert model.calls[-1][-1] == {
            "role": "tool",
            "tool_call_id": saved.tool_call.id,
            "name": "external",
            "content": "Prague",
        }
        assert [
            message for message in model.calls[-1] if message["role"] == "user"
        ] == [
            {"role": "user", "content": "prior"},
            {"role": "user", "content": "start"},
        ]

        stale = await runtime.send_message(result_input, recipient=ROOT)
        assert stale.status == DeliveryStatus.HANDLER_ERROR
        assert stale.error is not None
        assert "Continuation revision" in stale.error.message
        assert task.run_state.revision == 8
        assert len(model.calls) == 2
    finally:
        await runtime.stop_when_idle()
        await asyncio.sleep(0)
        gc.collect()
        loop.set_exception_handler(previous_handler)
    assert unhandled == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
async def test_independent_root_can_pause_and_resume_while_another_root_has_a_child(
    mode: str,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def wait(text: str) -> str:
        """Keep the child active until the independent root completes."""
        entered.set()
        await release.wait()
        return text

    runner = Runner()
    runtime = SingleThreadedRuntimeEngine()
    parent = Agent(
        runtime,
        runner,
        bind_id=AgentId.from_values("test", "other"),
        descriptor=AgentDescriptor(
            name="parent",
            model=ScriptedModel(
                response("agent", arguments='{"name":"Child","task":"Wait."}'),
                response(),
            ),
            tools=Tools(
                tools=[
                    DefaultAgentTool(
                        model=ScriptedModel(response("wait"), response()),
                        tools=Tools(tools=[wait]),
                    )
                ]
            ),
        ),
    )
    running = asyncio.create_task(runner.run(parent, state()))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        with pytest.raises(RunPaused) as paused:
            await execute(
                runner, ScriptedModel(response("external")), state(), mode=mode
            )
        result = await execute(
            runner,
            ScriptedModel(response()),
            ToolResultInput(paused.value.continuation, "answer"),
            mode=mode,
        )
        assert result.final_output == "done"
        assert not running.done()
    finally:
        release.set()
        try:
            assert (await running).final_output == "done"
        finally:
            await runtime.stop_when_idle()
