"""Regression coverage for MCP source ownership and turn composition."""

import asyncio
from collections import Counter
from dataclasses import dataclass, replace
from typing import cast

import pytest
from pydantic import BaseModel

from agentlane.harness import AgentDescriptor, RunState, Task
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.mcp import (
    MCPAuthorizationContext,
    MCPAuthorizationError,
    MCPClientManager,
    MCPDiscoveryError,
    MCPServer,
    MCPStdioTransport,
    MCPToolsShim,
)
from agentlane.harness.mcp._client import MCPClientLease
from agentlane.harness.mcp._shim import (
    _BoundMCPToolsShim,  # pyright: ignore[reportPrivateUsage]
)
from agentlane.harness.shims import (
    BoundShim,
    PreparedTurn,
    ShimBindingContext,
    ToolNameCollisionError,
)
from agentlane.harness.tools import HarnessToolDefinition, HarnessToolsShim
from agentlane.models import Tool, ToolExecutionContext, Tools
from agentlane.models.run import RunContext
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine

from ..tools_test_utils import (
    EmptyToolArgs,
    SequenceModel,
    make_assistant_response,
    named_tool,
)


class _SourceLease(MCPClientLease):
    def __init__(self, server: MCPServer) -> None:
        self.config = server
        self.released = False

    @property
    def server(self) -> MCPServer:
        return self.config

    async def tools(self) -> tuple[Tool[BaseModel, object], ...]:
        async def handler(
            args: BaseModel,
            cancellation_token: CancellationToken,
            context: ToolExecutionContext,
        ) -> object:
            del args, cancellation_token, context
            assert not self.released
            return "remote result"

        return (
            Tool(
                name=f"{self.server.name}__read",
                description="Source-owned test tool.",
                args_model=EmptyToolArgs,
                handler=handler,
            ),
        )

    async def release(self) -> None:
        self.released = True


class _SourceManager(MCPClientManager):
    """Control source acquisition, not MCP transport or catalog internals."""

    def __init__(self) -> None:
        super().__init__()
        self.attempts: Counter[str] = Counter()
        self.failures: dict[str, Exception] = {}
        self.leases: list[_SourceLease] = []
        self.active = 0
        self.peak_active = 0

    async def _acquire(
        self, server: MCPServer, context: MCPAuthorizationContext
    ) -> MCPClientLease:
        del context
        self.attempts[server.name] += 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            await asyncio.sleep(0)
            if failure := self.failures.get(server.name):
                raise failure
            lease = _SourceLease(server)
            self.leases.append(lease)
            return lease
        finally:
            self.active -= 1


def _server(name: str, *, required: bool = True) -> MCPServer:
    return MCPServer(
        name=name, transport=MCPStdioTransport(command="fixture"), required=required
    )


def _state() -> RunState:
    return RunState(instructions=None, history=[], responses=[])


def _binding() -> ShimBindingContext:
    return ShimBindingContext(task=Task(engine=SingleThreadedRuntimeEngine()))


@dataclass(init=False)
class _EqualMCPToolsShim(MCPToolsShim):
    """Distinct definitions can compare equal and have no hash."""


@pytest.mark.asyncio
async def test_inherited_binding_scope_omits_unselected_mcp_source() -> None:
    manager = _SourceManager()
    included = _EqualMCPToolsShim(
        servers=(_server("included"),), client_manager=manager
    )
    excluded = _EqualMCPToolsShim(
        servers=(_server("excluded"),), client_manager=manager
    )
    assert included == excluded
    assert type(included).__hash__ is None
    context = RunContext[object](context=None)
    parent = await included.bind(_binding())
    children: list[BoundShim] = []
    try:
        await parent.on_run_start(_state(), context)
        bindings = parent.inherit_tools(frozenset({"included__read"}))
        assert bindings
        child_context = replace(_binding(), tool_source_bindings=bindings)
        for source in (included, excluded):
            child = await source.bind(child_context)
            children.append(child)
            await child.on_run_start(_state(), context)
        turn = PreparedTurn(run_state=_state(), tools=None, model_args=None)
        for child in children:
            await child.prepare_turn(turn)
        assert turn.tools is not None
        assert {tool.name for tool in turn.tools.normalized_tools} == {"included__read"}
        assert manager.attempts == {"included": 2}
    finally:
        for child in children:
            await child.on_run_end(None, context)
        await parent.on_run_end(None, context)
        await manager.aclose()
    assert all(lease.released for lease in manager.leases)


@pytest.mark.asyncio
@pytest.mark.parametrize("mcp_first", [False, True])
async def test_mcp_harness_collision_fails_before_model_in_both_orders(
    mcp_first: bool,
) -> None:
    manager = _SourceManager()
    model = SequenceModel([make_assistant_response("must not run")])
    remote = MCPToolsShim(servers=(_server("remote"),), client_manager=manager)
    local = HarnessToolsShim(
        definitions=(HarnessToolDefinition(tool=named_tool("remote__read")),)
    )
    shims = (remote, local) if mcp_first else (local, remote)
    agent = DefaultAgent(
        descriptor=AgentDescriptor(name="fixture", model=model, shims=shims)
    )
    with pytest.raises(RuntimeError, match="remote__read.*collides"):
        await agent.run("test")
    assert not model.calls
    assert all(lease.released for lease in manager.leases)
    await manager.aclose()


@pytest.mark.asyncio
async def test_initial_optional_connection_failure_recovers_after_backoff() -> None:
    manager = _SourceManager()
    manager.failures["optional"] = ConnectionError("temporary transport failure")
    bound = cast(
        _BoundMCPToolsShim,
        await MCPToolsShim(
            servers=(_server("optional", required=False), _server("healthy")),
            client_manager=manager,
        ).bind(_binding()),
    )
    state = _state()
    context = RunContext[object](context=None)
    try:
        await bound.on_run_start(state, context)
        assert {tool.name for tool in bound.tools} == {"healthy__read"}
        assert manager.attempts["optional"] == 1
        manager.failures.clear()
        turn = PreparedTurn(run_state=state, tools=None, model_args=None)
        await bound.prepare_turn(turn)
        assert manager.attempts["optional"] == 1
        bound.sources[0].retry_at = 0.0
        await bound.prepare_turn(
            PreparedTurn(run_state=state, tools=None, model_args=None)
        )
        assert manager.attempts["optional"] == 2
        assert {tool.name for tool in bound.tools} == {
            "optional__read",
            "healthy__read",
        }
    finally:
        await bound.on_run_end(None, context)
        await manager.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [MCPAuthorizationError("denied"), MCPDiscoveryError("invalid catalog")],
)
async def test_optional_nontransient_failure_is_not_retried(
    failure: Exception,
) -> None:
    manager = _SourceManager()
    manager.failures["optional"] = failure
    bound = await MCPToolsShim(
        servers=(_server("optional", required=False),), client_manager=manager
    ).bind(_binding())
    state = _state()
    context = RunContext[object](context=None)
    try:
        await bound.on_run_start(state, context)
        manager.failures.clear()
        for _ in range(3):
            await bound.prepare_turn(
                PreparedTurn(run_state=state, tools=None, model_args=None)
            )
        assert manager.attempts["optional"] == 1
    finally:
        await bound.on_run_end(None, context)
        await manager.aclose()


@pytest.mark.asyncio
async def test_inherited_tools_acquire_only_their_owning_servers() -> None:
    manager = _SourceManager()
    parent = await MCPToolsShim(
        servers=(_server("needed"), _server("unrelated")), client_manager=manager
    ).bind(_binding())
    context = RunContext[object](context=None)
    child = None
    try:
        await parent.on_run_start(_state(), context)
        manager.failures["unrelated"] = ConnectionError("no child access")
        inherited = parent.inherit_tools(frozenset({"needed__read"}))
        assert inherited is not None and len(inherited) == 1
        child = await inherited[0].bind(_binding())
        await child.on_run_start(_state(), context)
        turn = PreparedTurn(run_state=_state(), tools=None, model_args=None)
        await child.prepare_turn(turn)
        assert turn.tools is not None
        assert {tool.name for tool in turn.tools.normalized_tools} == {"needed__read"}
        assert manager.attempts == {"needed": 2, "unrelated": 1}
    finally:
        if child is not None:
            await child.on_run_end(None, context)
        await parent.on_run_end(None, context)
        await manager.aclose()
    assert all(lease.released for lease in manager.leases)


@pytest.mark.asyncio
async def test_source_startup_concurrency_is_bounded() -> None:
    manager = _SourceManager()
    bound = await MCPToolsShim(
        servers=tuple(_server(f"source_{index}") for index in range(12)),
        client_manager=manager,
        max_concurrent_discoveries=3,
    ).bind(_binding())
    context = RunContext[object](context=None)
    try:
        await bound.on_run_start(_state(), context)
        assert manager.peak_active == 3
        assert len(manager.leases) == 12
    finally:
        await bound.on_run_end(None, context)
        await manager.aclose()


@pytest.mark.asyncio
async def test_required_failure_cancels_pending_discovery_and_releases_leases() -> None:
    started = asyncio.Event()
    canceled = asyncio.Event()

    class BlockingManager(_SourceManager):
        async def _acquire(
            self, server: MCPServer, context: MCPAuthorizationContext
        ) -> MCPClientLease:
            if server.name == "blocked":
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    canceled.set()
                    raise
            if server.name == "failed":
                await started.wait()
                raise ConnectionError("required server unavailable")
            return await super()._acquire(server, context)

    manager = BlockingManager()
    bound = await MCPToolsShim(
        servers=tuple(_server(name) for name in ("healthy", "blocked", "failed")),
        client_manager=manager,
        max_concurrent_discoveries=2,
    ).bind(_binding())
    try:
        async with asyncio.timeout(1):
            with pytest.raises(ConnectionError, match="required server"):
                await bound.on_run_start(_state(), RunContext[object](context=None))
        assert canceled.is_set()
        assert manager.leases and all(lease.released for lease in manager.leases)
    finally:
        await manager.aclose()


def test_local_tool_precedence_and_execution_settings_stay_unchanged() -> None:
    original = named_tool("local")
    turn = PreparedTurn(
        run_state=_state(),
        tools=Tools(tools=(original,), tool_call_timeout=17, tool_call_max_retries=0),
        model_args=None,
    )
    turn.add_tools((named_tool("local"),))
    assert turn.tools is not None
    assert turn.tools.normalized_tools == (original,)
    assert turn.tools.tool_call_timeout == 17
    assert turn.tools.tool_call_max_retries == 0


def test_unique_name_reservation_survives_tool_filtering() -> None:
    turn = PreparedTurn(run_state=_state(), tools=None, model_args=None)
    turn.add_tools((named_tool("remote__read"),), require_unique_names=True)
    turn.exclude_tools(frozenset({"remote__read"}))
    with pytest.raises(ToolNameCollisionError, match="remote__read"):
        turn.add_tools((named_tool("remote__read"),))


@pytest.mark.parametrize("limit", [0, -1, True])
def test_discovery_concurrency_requires_a_positive_integer(limit: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        MCPToolsShim(servers=(_server("remote"),), max_concurrent_discoveries=limit)
