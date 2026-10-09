"""MCP integration with the normal agent lifecycle and tool policies."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, TypeGuard, cast

import pytest
from pydantic import BaseModel

from agentlane.harness import (
    INHERIT_TOOLS,
    OVERRIDE_TOOLS,
    RESTRICT_TOOLS,
    AgentDescriptor,
    DefaultAgentTool,
    DefaultHandoff,
    RunnerHooks,
    RunResult,
    RunState,
    Task,
    ToolConfig,
)
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.mcp import (
    MCPAuthorizationContext,
    MCPClientManager,
    MCPDiscoveryError,
    MCPServer,
    MCPStdioTransport,
    MCPToolsShim,
)
from agentlane.harness.mcp._client import MCPClientLease
from agentlane.harness.shims import (
    BoundShim,
    DelegatingBoundShim,
    DelegatingShim,
    ExcludeToolsShim,
    PreparedTurn,
    Shim,
    ShimBindingContext,
    ToolSourceBinding,
)
from agentlane.models import (
    MessageDict,
    ModelResponse,
    Tool,
    ToolCall,
    ToolError,
    ToolExecutionContext,
    ToolFailure,
    Tools,
    ToolSpec,
)
from agentlane.models.run import RunContext
from agentlane.runtime import CancellationToken

from ..tools_test_utils import (
    SequenceModel,
    StreamingSequenceModel,
    make_assistant_response,
    named_tool,
)


class _EmptyArgs(BaseModel):
    pass


class _FakeManager(MCPClientManager):
    def __init__(self) -> None:
        super().__init__()
        self.remote_tools: dict[str, tuple[str, ...]] = {}
        self.failures: set[str] = set()
        self.leases: list[_FakeLease] = []
        self.calls: list[str] = []

    async def _acquire(
        self, server: MCPServer, context: MCPAuthorizationContext
    ) -> MCPClientLease:
        del context
        assert not self.closed
        lease = _FakeLease(self, server)
        self.leases.append(lease)
        return lease

    async def aclose(self) -> None:
        await super().aclose()


class _FakeLease(MCPClientLease):
    """Exercise the shim through its internal manager and lease contract."""

    def __init__(self, manager: _FakeManager, server: MCPServer) -> None:
        self.manager = manager
        self.config = server
        self.released = False
        self.reads = 0

    @property
    def server(self) -> MCPServer:
        return self.config

    async def tools(self) -> tuple[Tool[BaseModel, object], ...]:
        assert not self.released
        assert not self.manager.closed
        self.reads += 1
        if self.server.name in self.manager.failures:
            raise MCPDiscoveryError("Fixture catalog unavailable.")
        return tuple(
            self._tool(name)
            for name in self.manager.remote_tools.get(
                self.server.name, ("read", "write")
            )
        )

    def _tool(self, name: str) -> Tool[BaseModel, object]:
        async def handler(
            args: BaseModel,
            cancellation_token: CancellationToken,
            context: ToolExecutionContext,
        ) -> object:
            del args, cancellation_token, context
            assert not self.released
            assert not self.manager.closed
            self.manager.calls.append(f"{self.server.name}__{name}")
            return "done"

        return Tool(
            name=f"{self.server.name}__{name}",
            description="Fixture remote tool.",
            args_model=_EmptyArgs,
            handler=handler,
        )

    async def release(self) -> None:
        self.released = True


def _mcp(manager: MCPClientManager | None, *, name: str = "remote") -> MCPToolsShim:
    return MCPToolsShim(
        servers=(MCPServer(name=name, transport=MCPStdioTransport(command="fixture")),),
        client_manager=manager,
    )


def _call(name: str, arguments: str = "{}") -> ToolCall:
    return ToolCall.model_validate(
        {
            "id": f"call_{name}",
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
    )


def _names(model: SequenceModel | StreamingSequenceModel, turn: int = 0) -> set[str]:
    tools = model.call_tools[turn]
    return (
        {tool.name for tool in tools.normalized_tools} if tools is not None else set()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "kind", ["generic", "predefined", "handoff", "default_handoff"]
)
@pytest.mark.parametrize(
    "policy", [INHERIT_TOOLS, OVERRIDE_TOOLS, RESTRICT_TOOLS.only("remote__read")]
)
async def test_mcp_inheritance_policies_through_default_agent(
    kind: str, streaming: bool, policy: ToolConfig
) -> None:
    manager = _FakeManager()
    # Subagents are subroutines and use terminal calls even in a streamed run.
    child_streams = streaming and kind in {"handoff", "default_handoff"}
    child_model = (
        StreamingSequenceModel([make_assistant_response("child done")])
        if child_streams
        else SequenceModel([make_assistant_response("child done")])
    )
    child = AgentDescriptor(name="child", model=child_model, tools=policy)
    parent_tools = None
    handoffs = None
    default_handoff = None
    if kind == "generic":
        parent_tools = Tools(tools=[DefaultAgentTool(model=child_model, tools=policy)])
        call = _call("agent", '{"name":"child","task":"test"}')
    elif kind == "predefined":
        parent_tools = Tools(tools=[child.as_tool()])
        call = _call("child")
    elif kind == "handoff":
        handoffs = (child,)
        call = _call("child", '{"task":"test"}')
    else:
        default_handoff = DefaultHandoff(model=child_model, tools=policy)
        call = _call("handoff", '{"task":"test"}')
    outcomes = [make_assistant_response(None, tool_calls=[call])]
    if kind in {"generic", "predefined"}:
        outcomes.append(make_assistant_response("parent done"))
    parent_model = (
        StreamingSequenceModel(outcomes) if streaming else SequenceModel(list(outcomes))
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=parent_tools,
            handoffs=handoffs,
            default_handoff=default_handoff,
            shims=(_mcp(manager),),
        )
    )
    if streaming:
        stream = await agent.run_stream("test")
        async for _ in stream:
            pass
        await stream.result()
    else:
        await agent.run("test")
    expected: set[str] = (
        {"remote__read", "remote__write"}
        if policy is INHERIT_TOOLS
        else set() if policy is OVERRIDE_TOOLS else {"remote__read"}
    )
    assert {
        name for name in _names(child_model) if name.startswith("remote__")
    } == expected
    assert all(lease.released for lease in manager.leases)
    assert manager.closed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "kind", ["generic", "predefined", "handoff", "default_handoff"]
)
@pytest.mark.parametrize("limit", ["round", "tool"])
@pytest.mark.parametrize("child_local", [False, True])
async def test_restricted_mcp_inherits_parent_execution_settings(
    kind: str, streaming: bool, limit: str, child_local: bool
) -> None:
    """Dynamic-only restrictions retain budgets before child discovery runs."""
    manager = _FakeManager()
    child_outcomes = [
        make_assistant_response(None, tool_calls=[_call("remote__read")]),
        make_assistant_response("child done"),
    ]
    child_streams = streaming and kind in {"handoff", "default_handoff"}
    child_model = (
        StreamingSequenceModel(child_outcomes)
        if child_streams
        else SequenceModel(list(child_outcomes))
    )
    policy = RESTRICT_TOOLS.only(
        "remote__read",
        tools=Tools(tools=[named_tool("local")]) if child_local else None,
    )
    child = AgentDescriptor(name="child", model=child_model, tools=policy)
    parent_tools = Tools(
        tools=[],
        parallel_tool_calls=True,
        tool_call_timeout=17,
        tool_call_max_retries=0,
        tool_call_limits={"remote__read": 1} if limit == "tool" else None,
        max_tool_round_trips=1 if limit == "round" else 10,
    )
    handoffs = None
    default_handoff = None
    if kind == "generic":
        parent_tools = replace(
            parent_tools,
            tools=[DefaultAgentTool(model=child_model, tools=policy)],
        )
        call = _call("agent", '{"name":"child","task":"test"}')
    elif kind == "predefined":
        parent_tools = replace(parent_tools, tools=[child.as_tool()])
        call = _call("child")
    elif kind == "handoff":
        handoffs = (child,)
        call = _call("child", '{"task":"test"}')
    else:
        default_handoff = DefaultHandoff(model=child_model, tools=policy)
        call = _call("handoff", '{"task":"test"}')
    parent_outcomes = [make_assistant_response(None, tool_calls=[call])]
    if kind in {"generic", "predefined"}:
        parent_outcomes.append(make_assistant_response("parent done"))
    parent_model = (
        StreamingSequenceModel(parent_outcomes)
        if streaming
        else SequenceModel(list(parent_outcomes))
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=parent_tools,
            handoffs=handoffs,
            default_handoff=default_handoff,
            shims=(_mcp(manager),),
        )
    )
    if streaming:
        stream = await agent.run_stream("test")
        async for _ in stream:
            pass
        await stream.result()
    else:
        await agent.run("test")

    visible = cast(Tools, child_model.call_tools[0])
    assert visible.tool_call_timeout == 17
    assert visible.tool_call_max_retries == 0
    assert visible.parallel_tool_calls is True
    assert visible.tool_call_limits == parent_tools.tool_call_limits
    assert visible.max_tool_round_trips == parent_tools.max_tool_round_trips
    assert _names(child_model) == (
        {"remote__read", "local"} if child_local else {"remote__read"}
    )
    assert _names(child_model, 1) == (
        {"local"} if child_local and limit == "tool" else set()
    )
    assert manager.calls == ["remote__read"]
    assert len(manager.leases) == 2
    assert all(lease.released for lease in manager.leases)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy", [OVERRIDE_TOOLS, RESTRICT_TOOLS.only("remote__read")]
)
async def test_child_local_additions_survive_inherited_mcp_restrictions(
    policy: Any,
) -> None:
    manager = _FakeManager()
    child_model = SequenceModel([make_assistant_response("child done")])
    child = AgentDescriptor(
        name="child",
        model=child_model,
        tools=policy.with_tools(Tools(tools=[named_tool("local")])),
        shims=(_mcp(manager, name="local_source"),),
    )
    parent_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("child")]),
            make_assistant_response("done"),
        ]
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=Tools(tools=[child.as_tool()]),
            shims=(_mcp(manager),),
        )
    ).run("test")
    assert {"local", "local_source__read", "local_source__write"} <= _names(child_model)
    assert "remote__write" not in _names(child_model)


@pytest.mark.asyncio
async def test_inherited_mcp_source_cannot_restore_parent_exclusions() -> None:
    manager = _FakeManager()
    child_model = SequenceModel([make_assistant_response("child done")])
    parent_model = SequenceModel(
        [
            make_assistant_response(
                None, tool_calls=[_call("agent", '{"name":"child","task":"test"}')]
            ),
            make_assistant_response("done"),
        ]
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=Tools(tools=[DefaultAgentTool(model=child_model)]),
            shims=(ExcludeToolsShim(names=("remote__write",)), _mcp(manager)),
        )
    ).run("test")
    assert "remote__read" in _names(child_model)
    assert "remote__write" not in _names(child_model)


@pytest.mark.asyncio
@pytest.mark.parametrize("exclude_first", [False, True])
async def test_mcp_exclusions_apply_after_all_shims(exclude_first: bool) -> None:
    manager = _FakeManager()
    model = SequenceModel([make_assistant_response("done")])
    exclude = ExcludeToolsShim(names=("remote__write", "local"))
    shims: Sequence[Shim] = (
        (exclude, _mcp(manager)) if exclude_first else (_mcp(manager), exclude)
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=model,
            tools=Tools(
                tools=[named_tool("local")],
                tool_call_timeout=17,
                tool_call_max_retries=0,
            ),
            shims=shims,
        )
    ).run("test")
    assert _names(model) == {"remote__read"}
    visible = cast(Tools, model.call_tools[0])
    assert visible.tool_call_timeout == 17
    assert visible.tool_call_max_retries == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["round", "tool"])
async def test_mcp_respects_final_tool_budgets(limit: str) -> None:
    manager = _FakeManager()
    model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("remote__read")]),
            make_assistant_response("done"),
        ]
    )
    tools = (
        Tools(tools=[], max_tool_round_trips=1)
        if limit == "round"
        else Tools(tools=[], tool_call_limits={"remote__read": 1})
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent", model=model, tools=tools, shims=(_mcp(manager),)
        )
    ).run("test")
    assert "remote__read" in _names(model)
    assert "remote__read" not in _names(model, 1)
    assert _names(model, 1) == (set() if limit == "round" else {"remote__write"})
    assert manager.calls == ["remote__read"]


@pytest.mark.asyncio
@pytest.mark.parametrize("exclude_first", [False, True])
async def test_all_excluded_tools_drop_required_choice_before_model(
    exclude_first: bool,
) -> None:
    """An empty final catalog must not send required tool choice to a provider."""
    manager = _FakeManager()
    exclude = ExcludeToolsShim(names=("remote__read", "remote__write"))
    mcp = _mcp(manager)
    model = SequenceModel([make_assistant_response("done")])
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="empty-tools",
            model=model,
            tools=Tools(tools=(), tool_choice="required"),
            shims=(exclude, mcp) if exclude_first else (mcp, exclude),
        )
    ).run("test")
    assert model.call_tools[0] is None


class _ChangeCatalog(Shim):
    def __init__(self, manager: _FakeManager) -> None:
        self.manager = manager

    @property
    def name(self) -> str:
        return "change-catalog"

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        if turn.run_state.turn_count == 2:
            self.manager.remote_tools["remote"] = ("replacement",)


@pytest.mark.asyncio
async def test_mcp_refreshes_catalog_on_each_model_turn() -> None:
    manager = _FakeManager()
    model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("remote__read")]),
            make_assistant_response("done"),
        ]
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent", model=model, shims=(_ChangeCatalog(manager), _mcp(manager))
        )
    ).run("test")
    assert _names(model, 1) == {"remote__replacement"}
    assert manager.leases[0].reads == 3  # Startup and each model turn.


class _StartupFailure(Shim):
    def __init__(self, calls: list[str], *, fail_cleanup: bool = False) -> None:
        self.calls = calls
        self.fail_cleanup = fail_cleanup

    @property
    def name(self) -> str:
        return "startup-failure"

    async def on_run_start(
        self, state: RunState, transient_state: RunContext[Any]
    ) -> None:
        del state, transient_state
        self.calls.append("start")
        raise ValueError("primary startup failure")

    async def on_run_end(
        self, result: object, transient_state: RunContext[Any]
    ) -> None:
        del result, transient_state
        self.calls.append("end")
        if self.fail_cleanup:
            raise RuntimeError("secondary cleanup failure")


@pytest.mark.asyncio
async def test_mcp_startup_failure_releases_leases_and_preserves_primary_error() -> (
    None
):
    manager = _FakeManager()
    model = SequenceModel([make_assistant_response("must not run")])
    calls: list[str] = []
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=model,
            shims=(_mcp(manager), _StartupFailure(calls, fail_cleanup=True)),
        )
    )
    with pytest.raises(RuntimeError, match="primary startup failure"):
        await agent.run("test")
    assert calls == ["start", "end"]
    assert len(manager.leases) == 1
    assert manager.leases[0].released
    assert not model.calls


@pytest.mark.asyncio
async def test_inherited_local_source_owns_an_independent_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managers: list[_FakeManager] = []

    def factory() -> _FakeManager:
        manager = _FakeManager()
        managers.append(manager)
        return manager

    monkeypatch.setattr("agentlane.harness.mcp._shim.MCPClientManager", factory)
    child_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("remote__read")]),
            make_assistant_response("child done"),
        ]
    )
    parent_model = SequenceModel(
        [
            make_assistant_response(
                None, tool_calls=[_call("agent", '{"name":"child","task":"test"}')]
            ),
            make_assistant_response("done"),
        ]
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=Tools(tools=[DefaultAgentTool(model=child_model)]),
            shims=(_mcp(None),),
        )
    ).run("test")
    active_managers = [manager for manager in managers if manager.leases]
    assert len(active_managers) == 2
    assert all(manager.closed for manager in active_managers)
    assert active_managers[0].calls == []
    assert active_managers[1].calls == ["remote__read"]


class _CleanupFailure(Shim):
    @property
    def name(self) -> str:
        return "cleanup-failure"

    async def on_run_end(
        self, result: object, transient_state: RunContext[Any]
    ) -> None:
        del result, transient_state
        raise RuntimeError("first cleanup failure")


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_skip_later_mcp_cleanup() -> None:
    manager = _FakeManager()
    model = SequenceModel([make_assistant_response("done")])
    calls: list[str] = []
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=model,
            shims=(_CleanupFailure(), _mcp(manager), _StartupFailure(calls)),
        )
    )
    with pytest.raises(RuntimeError, match="primary startup failure"):
        await agent.run("test")
    assert manager.leases[0].released
    assert calls == ["start", "end"]


class _GrowCatalog(Shim):
    def __init__(self, manager: _FakeManager) -> None:
        self.manager = manager

    @property
    def name(self) -> str:
        return "grow-catalog"

    async def on_run_start(
        self, state: RunState, transient_state: RunContext[Any]
    ) -> None:
        del state, transient_state
        self.manager.remote_tools["remote"] = ("read", "write", "new")


@pytest.mark.asyncio
async def test_child_catalog_refresh_keeps_parent_visible_name_cap() -> None:
    manager = _FakeManager()
    child_model = SequenceModel([make_assistant_response("child done")])
    child = AgentDescriptor(
        name="child", model=child_model, shims=(_GrowCatalog(manager),)
    )
    parent_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("child")]),
            make_assistant_response("done"),
        ]
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=Tools(tools=[child.as_tool()]),
            shims=(_mcp(manager), ExcludeToolsShim(names=("remote__write",))),
        )
    ).run("test")
    assert "remote__read" in _names(child_model)
    assert "remote__write" not in _names(child_model)
    assert "remote__new" not in _names(child_model)


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [INHERIT_TOOLS, OVERRIDE_TOOLS])
async def test_child_local_name_collision_requires_explicit_override(
    policy: Any,
) -> None:
    manager = _FakeManager()
    child_model = SequenceModel([make_assistant_response("child done")])
    child = AgentDescriptor(
        name="child",
        model=child_model,
        tools=policy.with_tools(Tools(tools=[named_tool("remote__read")])),
    )
    parent_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("child")]),
            make_assistant_response("done"),
        ]
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=Tools(tools=[child.as_tool()]),
            shims=(_mcp(manager),),
        )
    ).run("test")
    if policy is INHERIT_TOOLS:
        assert not child_model.calls
        assert "delegated agent call failed" in str(parent_model.calls[1])
    else:
        assert _names(child_model) == {"remote__read"}


@pytest.mark.asyncio
@pytest.mark.parametrize("required", [False, True])
async def test_required_catalog_failure_stops_run_and_optional_preserves_other_servers(
    required: bool,
) -> None:
    manager = _FakeManager()
    manager.failures.add("unavailable")
    model = SequenceModel([make_assistant_response("done")])
    shim = MCPToolsShim(
        servers=(
            MCPServer(
                name="unavailable",
                transport=MCPStdioTransport(command="fixture"),
                required=required,
            ),
            MCPServer(name="available", transport=MCPStdioTransport(command="fixture")),
        ),
        client_manager=manager,
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(name="parent", model=model, shims=(shim,))
    )
    if required:
        with pytest.raises(RuntimeError, match="Fixture catalog unavailable"):
            await agent.run("test")
        assert not model.calls
    else:
        await agent.run("test")
        assert _names(model) == {"available__read", "available__write"}
    assert all(lease.released for lease in manager.leases)


@dataclass
class _WrapperSession:
    label: str
    task_id: str
    state: RunState | None = None
    transient_state: RunContext[Any] | None = None
    started: int = 0
    ended: int = 0
    prepared: int = 0
    transformed: int = 0
    responses: int = 0
    hook_calls: list[str] = field(default_factory=list[str])
    guarded_calls: list[ToolExecutionContext] = field(
        default_factory=list[ToolExecutionContext]
    )


@dataclass
class _WrapperAudit:
    sessions: list[_WrapperSession] = field(default_factory=list[_WrapperSession])
    denied_agents: frozenset[str] = frozenset()


class _WrapperHooks(RunnerHooks):
    def __init__(self, session: _WrapperSession) -> None:
        self.session = session

    async def on_tool_call_start(self, task: Task, tool_call: ToolCall) -> None:
        assert str(task.task_id) == self.session.task_id
        self.session.hook_calls.append(tool_call.function.name)


def _is_remote_tool(tool: ToolSpec[Any]) -> TypeGuard[Tool[Any, Any]]:
    return isinstance(tool, Tool) and "__" in tool.name


class _AuditedBoundShim(DelegatingBoundShim):
    def __init__(
        self, inner: BoundShim, session: _WrapperSession, audit: _WrapperAudit
    ) -> None:
        super().__init__(inner)
        self.session = session
        self.audit = audit

    async def on_run_start(
        self, state: RunState, transient_state: RunContext[Any]
    ) -> None:
        assert self.session.started == 0
        self.session.started += 1
        self.session.state = state
        self.session.transient_state = transient_state
        await super().on_run_start(state, transient_state)

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        assert self.session.started == 1
        assert self.session.ended == 0
        assert turn.run_state is self.session.state
        assert turn.transient_state is self.session.transient_state
        self.session.prepared += 1
        await super().prepare_turn(turn)
        if turn.tools is not None:
            turn.tools = replace(
                turn.tools,
                tools=tuple(
                    self._guard_tool(tool) if _is_remote_tool(tool) else tool
                    for tool in turn.tools.normalized_tools
                ),
            )

    def _guard_tool(self, tool: Tool[Any, Any]) -> Tool[Any, Any]:
        async def guarded(
            args: BaseModel,
            token: CancellationToken,
            context: ToolExecutionContext,
        ) -> object:
            assert self.session.started == 1
            assert self.session.ended == 0
            assert self.session.prepared > 0
            assert context.run_id == self.session.task_id
            assert context.run_state is not None
            assert str(context.run_state.task_id) == self.session.task_id
            assert tool.name in self.session.hook_calls
            self.session.guarded_calls.append(context)
            if context.agent_name in self.audit.denied_agents:
                return ToolFailure(
                    text="Wrapper denied this remote operation.",
                    error=ToolError(
                        message="Wrapper denied this remote operation.",
                        kind="wrapper_denied",
                    ),
                )
            return await tool.run(args, token, context)

        return tool.replace(handler=guarded)

    async def transform_messages(
        self, turn: PreparedTurn, messages: list[MessageDict]
    ) -> list[MessageDict] | None:
        self.session.transformed += 1
        return await super().transform_messages(turn, messages)

    async def on_model_response(
        self, turn: PreparedTurn, response: ModelResponse
    ) -> None:
        self.session.responses += 1
        await super().on_model_response(turn, response)

    async def on_run_end(
        self, result: RunResult | None, transient_state: RunContext[Any]
    ) -> None:
        assert transient_state is self.session.transient_state
        self.session.ended += 1
        await super().on_run_end(result, transient_state)

    def runner_hooks(self) -> tuple[RunnerHooks, ...]:
        return (*super().runner_hooks(), _WrapperHooks(self.session))


class _AuditedShim(DelegatingShim):
    def __init__(self, inner: Shim, label: str, audit: _WrapperAudit) -> None:
        super().__init__(inner)
        self.label = label
        self.audit = audit

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        session = _WrapperSession(self.label, str(context.task.task_id))
        self.audit.sessions.append(session)
        return _AuditedBoundShim(await self.inner.bind(context), session, self.audit)


def _wrapped_mcp(
    manager: _FakeManager, audit: _WrapperAudit, *, multiple_sources: bool = False
) -> Shim:
    names = ("remote", "unrelated") if multiple_sources else ("remote",)
    source = MCPToolsShim(
        servers=tuple(
            MCPServer(name=name, transport=MCPStdioTransport(command="fixture"))
            for name in names
        ),
        client_manager=manager,
    )
    return _AuditedShim(_AuditedShim(source, "inner", audit), "outer", audit)


def _assert_wrappers_rebound(audit: _WrapperAudit, *, agent_count: int) -> None:
    assert len(audit.sessions) == agent_count * 2
    for label in ("inner", "outer"):
        sessions = [session for session in audit.sessions if session.label == label]
        assert len({session.task_id for session in sessions}) == agent_count
        assert len({id(session.state) for session in sessions}) == agent_count
        assert len({id(session.transient_state) for session in sessions}) == agent_count
        for session in sessions:
            assert session.started == session.ended == 1
            assert session.prepared == session.transformed == session.responses
            assert session.prepared >= 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["generic", "predefined", "handoff", "default_handoff"]
)
async def test_nested_mcp_wrappers_rebind_lifecycle_guards_and_hooks_for_child(
    kind: str,
) -> None:
    manager = _FakeManager()
    audit = _WrapperAudit()
    child_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("remote__read")]),
            make_assistant_response("child done"),
        ]
    )
    child = AgentDescriptor(name="child", model=child_model, tools=INHERIT_TOOLS)
    parent_tools = None
    handoffs = None
    default_handoff = None
    if kind == "generic":
        parent_tools = Tools(tools=[DefaultAgentTool(model=child_model)])
        child_call = _call("agent", '{"name":"child","task":"test"}')
    elif kind == "predefined":
        parent_tools = Tools(tools=[child.as_tool()])
        child_call = _call("child")
    elif kind == "handoff":
        handoffs = (child,)
        child_call = _call("child", '{"task":"test"}')
    else:
        default_handoff = DefaultHandoff(model=child_model)
        child_call = _call("handoff", '{"task":"test"}')
    parent_outcomes = [make_assistant_response(None, tool_calls=[child_call])]
    if kind in {"generic", "predefined"}:
        parent_outcomes.append(
            make_assistant_response(None, tool_calls=[_call("remote__read")])
        )
        parent_outcomes.append(make_assistant_response("parent done"))
    parent_model = SequenceModel(parent_outcomes)
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=parent_tools,
            handoffs=handoffs,
            default_handoff=default_handoff,
            shims=(_wrapped_mcp(manager, audit),),
        )
    ).run("test")

    _assert_wrappers_rebound(audit, agent_count=2)
    assert {"remote__read", "remote__write"} <= _names(child_model)
    expected_calls = 2 if kind in {"generic", "predefined"} else 1
    assert manager.calls == ["remote__read"] * expected_calls
    for label in ("inner", "outer"):
        calls = [
            context
            for session in audit.sessions
            if session.label == label
            for context in session.guarded_calls
        ]
        assert len(calls) == expected_calls
        assert len({context.run_id for context in calls}) == expected_calls
    assert len(manager.leases) == 2
    assert all(lease.released for lease in manager.leases)


@pytest.mark.asyncio
async def test_wrapped_mcp_restriction_prunes_sources_before_child_startup() -> None:
    manager = _FakeManager()
    audit = _WrapperAudit()
    child_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("remote__read")]),
            make_assistant_response("child done"),
        ]
    )
    child = AgentDescriptor(
        name="child",
        model=child_model,
        tools=RESTRICT_TOOLS.only("remote__read"),
    )
    parent_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("child")]),
            make_assistant_response("parent done"),
        ]
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=Tools(tools=[child.as_tool()]),
            shims=(_wrapped_mcp(manager, audit, multiple_sources=True),),
        )
    ).run("test")

    _assert_wrappers_rebound(audit, agent_count=2)
    assert _names(child_model) == {"remote__read"}
    assert [lease.server.name for lease in manager.leases].count("remote") == 2
    assert [lease.server.name for lease in manager.leases].count("unrelated") == 1
    assert manager.calls == ["remote__read"]
    assert sum(bool(session.guarded_calls) for session in audit.sessions) == 2
    assert all(lease.released for lease in manager.leases)


@pytest.mark.asyncio
async def test_wrapped_mcp_override_keeps_only_child_local_tools() -> None:
    manager = _FakeManager()
    audit = _WrapperAudit()
    child_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("remote__read")]),
            make_assistant_response("child done"),
        ]
    )
    child = AgentDescriptor(
        name="child",
        model=child_model,
        tools=OVERRIDE_TOOLS.with_tools(Tools(tools=[named_tool("remote__read")])),
    )
    parent_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("child")]),
            make_assistant_response("parent done"),
        ]
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=Tools(tools=[child.as_tool()]),
            shims=(_wrapped_mcp(manager, audit),),
        )
    ).run("test")

    _assert_wrappers_rebound(audit, agent_count=1)
    assert _names(child_model) == {"remote__read"}
    assert manager.calls == []
    assert len(manager.leases) == 1
    assert manager.leases[0].released
    assert all(not session.guarded_calls for session in audit.sessions)


@pytest.mark.asyncio
async def test_inherited_mcp_wrapper_can_deny_before_remote_call() -> None:
    manager = _FakeManager()
    audit = _WrapperAudit(denied_agents=frozenset({"child"}))
    child_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("remote__read")]),
            make_assistant_response("child done"),
        ]
    )
    child = AgentDescriptor(name="child", model=child_model)
    parent_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("child")]),
            make_assistant_response("parent done"),
        ]
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=Tools(tools=[child.as_tool()]),
            shims=(_wrapped_mcp(manager, audit),),
        )
    ).run("test")

    assert manager.calls == []
    assert "Wrapper denied this remote operation." in str(child_model.calls[1])
    _assert_wrappers_rebound(audit, agent_count=2)
    guards = [session for session in audit.sessions if session.guarded_calls]
    assert len(guards) == 1
    assert guards[0].label == "outer"
    assert guards[0].guarded_calls[0].agent_name == "child"
    assert all(lease.released for lease in manager.leases)


@pytest.mark.asyncio
async def test_nested_mcp_wrappers_preserve_restriction_through_grandchild() -> None:
    manager = _FakeManager()
    audit = _WrapperAudit()
    grandchild_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("remote__read")]),
            make_assistant_response("grandchild done"),
        ]
    )
    grandchild = AgentDescriptor(
        name="grandchild", model=grandchild_model, tools=INHERIT_TOOLS
    )
    child_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("grandchild")]),
            make_assistant_response("child done"),
        ]
    )
    child = AgentDescriptor(
        name="child",
        model=child_model,
        tools=RESTRICT_TOOLS.only(
            "remote__read", tools=Tools(tools=[grandchild.as_tool()])
        ),
    )
    parent_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("child")]),
            make_assistant_response("parent done"),
        ]
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=Tools(tools=[child.as_tool()]),
            shims=(_wrapped_mcp(manager, audit, multiple_sources=True),),
        )
    ).run("test")

    _assert_wrappers_rebound(audit, agent_count=3)
    assert {name for name in _names(grandchild_model) if "__" in name} == {
        "remote__read"
    }
    assert manager.calls == ["remote__read"]
    assert [lease.server.name for lease in manager.leases].count("remote") == 3
    assert [lease.server.name for lease in manager.leases].count("unrelated") == 1
    for label in ("inner", "outer"):
        calls = [
            context
            for session in audit.sessions
            if session.label == label
            for context in session.guarded_calls
        ]
        assert len(calls) == 1
        assert calls[0].agent_name == "grandchild"
    assert all(lease.released for lease in manager.leases)


@pytest.mark.asyncio
async def test_composite_mcp_wrappers_do_not_restore_an_excluded_leaf_source() -> None:
    class CombinedBound(BoundShim):
        def __init__(self, sessions: tuple[BoundShim, ...]) -> None:
            self.sessions = sessions

        async def on_run_start(
            self, state: RunState, transient_state: RunContext[Any]
        ) -> None:
            for session in self.sessions:
                await session.on_run_start(state, transient_state)

        async def prepare_turn(self, turn: PreparedTurn) -> None:
            for session in self.sessions:
                await session.prepare_turn(turn)

        async def on_run_end(
            self, result: RunResult | None, transient_state: RunContext[Any]
        ) -> None:
            for session in self.sessions:
                await session.on_run_end(result, transient_state)

        def inherit_tools(self, names: frozenset[str]) -> tuple[ToolSourceBinding, ...]:
            return tuple(
                binding
                for session in self.sessions
                for binding in session.inherit_tools(names) or ()
            )

    manager = _FakeManager()
    audit = _WrapperAudit()
    included = _mcp(manager, name="remote")
    excluded = _mcp(manager, name="unrelated")

    class CombinedMCPShim(Shim):
        @property
        def name(self) -> str:
            return "combined-mcp"

        async def bind(self, context: ShimBindingContext) -> BoundShim:
            return CombinedBound(
                (await included.bind(context), await excluded.bind(context))
            )

    child_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("remote__read")]),
            make_assistant_response("child done"),
        ]
    )
    child = AgentDescriptor(
        name="child",
        model=child_model,
        tools=RESTRICT_TOOLS.only("remote__read"),
    )
    parent_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("child")]),
            make_assistant_response("parent done"),
        ]
    )
    wrapped = _AuditedShim(
        _AuditedShim(CombinedMCPShim(), "inner", audit), "outer", audit
    )
    await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=parent_model,
            tools=Tools(tools=[child.as_tool()]),
            shims=(wrapped,),
        )
    ).run("test")

    _assert_wrappers_rebound(audit, agent_count=2)
    assert {"remote__read", "unrelated__read"} <= _names(parent_model)
    assert _names(child_model) == {"remote__read"}
    assert [lease.server.name for lease in manager.leases].count("remote") == 2
    assert [lease.server.name for lease in manager.leases].count("unrelated") == 1
    assert manager.calls == ["remote__read"]
    assert sum(bool(session.guarded_calls) for session in audit.sessions) == 2
    assert all(lease.released for lease in manager.leases)
