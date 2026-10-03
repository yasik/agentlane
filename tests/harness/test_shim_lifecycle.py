"""Generic shim cleanup, tool preparation, and execution-policy regressions."""

from contextlib import suppress
from dataclasses import replace
from typing import Any

import pytest

from agentlane.harness import AgentDescriptor, RunResult, RunState
from agentlane.harness._tooling import filter_tools
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.shims import (
    ExcludeToolsShim,
    PreparedTurn,
    Shim,
    ToolNameCollisionError,
)
from agentlane.harness.tools import HarnessToolDefinition, HarnessToolsShim
from agentlane.models import Tools, ToolSpec
from agentlane.models.run import RunContext

from .tools_test_utils import (
    SequenceModel,
    StreamingSequenceModel,
    echo_tool,
    make_assistant_response,
    make_tool_call,
    named_tool,
    run_state,
)


class _ToolSourceShim(Shim):
    def __init__(self, tools: tuple[ToolSpec[Any], ...]) -> None:
        self.tools = tools

    @property
    def name(self) -> str:
        return "tool-source"

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        turn.add_tools(self.tools, require_unique_names=True)


class _LifecycleShim(Shim):
    def __init__(
        self,
        label: str,
        calls: list[str],
        *,
        fail_start: bool = False,
        fail_prepare: bool = False,
        fail_cleanup: bool = False,
    ) -> None:
        self.label = label
        self.calls = calls
        self.fail_start = fail_start
        self.fail_prepare = fail_prepare
        self.fail_cleanup = fail_cleanup

    @property
    def name(self) -> str:
        return self.label

    async def on_run_start(
        self, state: RunState, transient_state: RunContext[Any]
    ) -> None:
        del state, transient_state
        self.calls.append(f"{self.label}:start")
        if self.fail_start:
            raise ValueError("primary startup failure")

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        del turn
        if self.fail_prepare:
            raise ValueError("primary preparation failure")

    async def on_run_end(
        self, result: RunResult | None, transient_state: RunContext[Any]
    ) -> None:
        del result, transient_state
        self.calls.append(f"{self.label}:end")
        if self.fail_cleanup:
            raise RuntimeError("secondary cleanup failure")


async def _run(agent: DefaultAgent, *, stream: bool) -> RunResult:
    if not stream:
        return await agent.run("test")

    events = await agent.run_stream("test")
    try:
        async for _event in events:
            pass
        return await events.result()
    finally:
        await events.aclose()
        with suppress(BaseException):
            await events.result()


def _names(model: SequenceModel, turn: int = 0) -> set[str]:
    tools = model.call_tools[turn]
    return (
        {tool.name for tool in tools.normalized_tools} if tools is not None else set()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_partial_startup_cleans_started_shims_and_preserves_primary_error(
    stream: bool,
) -> None:
    model_type = StreamingSequenceModel if stream else SequenceModel
    model = model_type([make_assistant_response("must not run")])
    calls: list[str] = []
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="fixture",
            model=model,
            shims=(
                _LifecycleShim("first", calls, fail_cleanup=True),
                _LifecycleShim("second", calls),
                _LifecycleShim("failed", calls, fail_start=True, fail_cleanup=True),
                _LifecycleShim("not-started", calls),
            ),
        )
    )
    with pytest.raises((RuntimeError, ValueError), match="primary startup failure"):
        await _run(agent, stream=stream)

    assert calls == [
        "first:start",
        "second:start",
        "failed:start",
        "first:end",
        "second:end",
        "failed:end",
    ]
    assert not model.calls


@pytest.mark.asyncio
async def test_preparation_error_is_not_replaced_by_cleanup_failure() -> None:
    calls: list[str] = []
    model = SequenceModel([make_assistant_response("must not run")])
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="fixture",
            model=model,
            shims=(
                _LifecycleShim("failed", calls, fail_prepare=True, fail_cleanup=True),
                _LifecycleShim("later", calls),
            ),
        )
    )
    with pytest.raises(RuntimeError, match="primary preparation failure"):
        await agent.run("test")

    assert calls == ["failed:start", "later:start", "failed:end", "later:end"]
    assert not model.calls


@pytest.mark.asyncio
async def test_cleanup_failure_after_success_runs_later_cleanup_and_is_reported() -> (
    None
):
    calls: list[str] = []
    model = SequenceModel([make_assistant_response("done")])
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="fixture",
            model=model,
            shims=(
                _LifecycleShim("failed", calls, fail_cleanup=True),
                _LifecycleShim("later", calls),
            ),
        )
    )
    with pytest.raises(RuntimeError, match="Run cleanup failed"):
        await agent.run("test")

    assert calls == ["failed:start", "later:start", "failed:end", "later:end"]
    assert len(model.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("exclude_first", [False, True])
async def test_source_exclusions_apply_after_all_shims_and_keep_settings(
    exclude_first: bool,
) -> None:
    model = SequenceModel([make_assistant_response("done")])
    source = _ToolSourceShim((named_tool("source_read"), named_tool("source_write")))
    exclude = ExcludeToolsShim(names=("source_write", "local"))
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="fixture",
            model=model,
            tools=Tools(
                tools=(named_tool("local"),),
                tool_call_timeout=17,
                tool_call_max_retries=0,
            ),
            shims=(exclude, source) if exclude_first else (source, exclude),
        )
    ).run("test")

    assert _names(model) == {"source_read"}
    visible = model.call_tools[0]
    assert visible is not None
    assert visible.tool_call_timeout == 17
    assert visible.tool_call_max_retries == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["round", "tool"])
@pytest.mark.parametrize("stream", [False, True])
async def test_shim_contributions_obey_final_tool_budgets(
    limit: str, stream: bool
) -> None:
    model_type = StreamingSequenceModel if stream else SequenceModel
    model = model_type(
        [
            make_assistant_response(
                None,
                tool_calls=[
                    make_tool_call(
                        tool_id="read",
                        name="source_read",
                        arguments='{"text":"read once"}',
                    )
                ],
            ),
            make_assistant_response("done"),
        ]
    )
    tools = (
        Tools(tools=(), max_tool_round_trips=1)
        if limit == "round"
        else Tools(tools=(), tool_call_limits={"source_read": 1})
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="fixture",
            model=model,
            tools=tools,
            shims=(
                _ToolSourceShim((echo_tool("source_read"), echo_tool("source_write"))),
            ),
        )
    )
    await _run(agent, stream=stream)

    assert _names(model) == {"source_read", "source_write"}
    assert _names(model, 1) == (set() if limit == "round" else {"source_write"})
    assert "read once" in str(model.calls[1])


@pytest.mark.asyncio
@pytest.mark.parametrize("exclude_first", [False, True])
async def test_empty_final_tools_remove_required_choice(exclude_first: bool) -> None:
    model = SequenceModel([make_assistant_response("done")])
    source = _ToolSourceShim((named_tool("source_read"),))
    exclude = ExcludeToolsShim(names=("source_read",))
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="fixture",
            model=model,
            tools=Tools(tools=(), tool_choice="required"),
            shims=(exclude, source) if exclude_first else (source, exclude),
        )
    ).run("test")

    assert model.call_tools[0] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("source_first", [False, True])
async def test_unique_source_collision_fails_before_model_in_either_order(
    source_first: bool,
) -> None:
    model = SequenceModel([make_assistant_response("must not run")])
    source = _ToolSourceShim((named_tool("source_read"),))
    local = HarnessToolsShim(
        definitions=(HarnessToolDefinition(tool=named_tool("source_read")),)
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="fixture",
            model=model,
            shims=(source, local) if source_first else (local, source),
        )
    )
    with pytest.raises(RuntimeError, match="source_read.*collides"):
        await agent.run("test")

    assert not model.calls


def test_local_tool_precedence_and_execution_settings_stay_unchanged() -> None:
    original = named_tool("local")
    settings = Tools(tools=(original,), tool_call_timeout=17, tool_call_max_retries=0)
    turn = PreparedTurn(run_state=run_state(), tools=settings, model_args=None)
    turn.add_tools((named_tool("local"),))

    assert turn.tools is not None
    assert turn.tools == settings
    assert turn.tools.normalized_tools[0] is original


def test_empty_name_filter_preserves_settings_for_later_source() -> None:
    settings = Tools(
        tools=(named_tool("local"),),
        tool_choice="required",
        parallel_tool_calls=True,
        tool_call_timeout=17,
        tool_call_max_retries=0,
        tool_call_limits={"source_read": 1},
        max_tool_round_trips=2,
    )
    turn = PreparedTurn(
        run_state=run_state(),
        tools=filter_tools(settings, names=frozenset()),
        model_args=None,
    )
    source = named_tool("source_read")
    turn.add_tools((source,))

    assert turn.tools == replace(settings, tools=(source,))


def test_unique_name_reservation_survives_tool_filtering() -> None:
    turn = PreparedTurn(run_state=run_state(), tools=None, model_args=None)
    turn.add_tools((named_tool("source_read"),), require_unique_names=True)
    turn.exclude_tools(frozenset({"source_read"}))

    with pytest.raises(ToolNameCollisionError, match="source_read"):
        turn.add_tools((named_tool("source_read"),))
