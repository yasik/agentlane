"""Resource bounds and transport-generation regressions for the MCP pool."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Literal

import httpx2
import pytest
from mcp import types

from agentlane.harness.mcp import (
    MCPAccessToken,
    MCPAuthorizationContext,
    MCPClientLimits,
    MCPClientManager,
    MCPPoolCapacityError,
    MCPServer,
    MCPShutdownTimeoutError,
    MCPStdioTransport,
    MCPStreamableHTTPTransport,
)
from agentlane.harness.mcp import _connection as mcp_connection
from agentlane.harness.mcp._connection import MCPConnection
from agentlane.harness.mcp._sdk import load_mcp_dependencies
from agentlane.models import ToolFailure
from agentlane.runtime import CancellationToken

from .helpers import ConnectionInstaller, acquire_lease

_FIXTURE_VALUE = "fixture-access-token"


@dataclass
class _Peer:
    opened: list[MCPConnection] = field(default_factory=list[MCPConnection])
    calls: list[int] = field(default_factory=list[int])
    fail_refresh: bool = False
    opening_gate: asyncio.Event | None = None
    call_gate: asyncio.Event | None = None
    call_started: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _Client:
    peer: _Peer
    generation: int

    async def list_tools(
        self, *, cursor: str | None, cache_mode: Literal["bypass"]
    ) -> types.ListToolsResult:
        del cursor, cache_mode
        if self.peer.fail_refresh and self.generation > 0:
            raise TimeoutError("temporary discovery timeout")
        return types.ListToolsResult(
            tools=[types.Tool(name="write", input_schema={"type": "object"})]
        )

    async def call_tool(
        self, name: str, arguments: dict[str, Any], *, read_timeout_seconds: float
    ) -> types.CallToolResult:
        del name, arguments, read_timeout_seconds
        self.peer.calls.append(self.generation)
        self.peer.call_started.set()
        if self.peer.call_gate is not None:
            await self.peer.call_gate.wait()
        return types.CallToolResult(content=[types.TextContent(text="done")])

    @asynccontextmanager
    async def listen(
        self, *, tools_list_changed: bool
    ) -> AsyncIterator[AsyncIterator[object]]:
        del tools_list_changed

        async def events() -> AsyncIterator[object]:
            yield object()

        yield events()


@pytest.fixture(name="peer")
def fixture_peer(install_connection: ConnectionInstaller) -> _Peer:
    state = _Peer()

    async def prepare(connection: MCPConnection) -> None:
        generation = len(state.opened)
        state.opened.append(connection)
        if state.opening_gate is not None:
            await state.opening_gate.wait()
        connection.client = _Client(state, generation)

    install_connection(prepare)
    return state


def _server() -> MCPServer:
    return MCPServer(name="peer", transport=MCPStdioTransport(command="fixture"))


def _pool_size(manager: MCPClientManager) -> int:
    connections = manager._connections  # pyright: ignore[reportPrivateUsage]
    retiring = manager._retiring  # pyright: ignore[reportPrivateUsage]
    return len(connections) + len(retiring)


@pytest.mark.asyncio
async def test_pool_evicts_idle_identities_before_opening_replacements(
    peer: _Peer,
) -> None:
    async with MCPClientManager(MCPClientLimits(max_connections=2)) as manager:
        for index in range(8):
            lease = await acquire_lease(
                manager, _server(), MCPAuthorizationContext(key=str(index))
            )
            assert _pool_size(manager) <= 2
            assert sum(not item.closing for item in peer.opened) <= 2
            await lease.release()
        assert len(peer.opened) == 8
        assert _pool_size(manager) == 2
    assert all(
        item.owner_task is not None and item.owner_task.done() for item in peer.opened
    )
    assert _pool_size(manager) == 0


@pytest.mark.asyncio
async def test_pool_rejects_capacity_without_evicting_active_lease(peer: _Peer) -> None:
    async with MCPClientManager(MCPClientLimits(max_connections=1)) as manager:
        active = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="active")
        )
        with pytest.raises(MCPPoolCapacityError) as caught:
            await acquire_lease(
                manager, _server(), MCPAuthorizationContext(key="other")
            )
        assert caught.value.retryable
        assert not peer.opened[0].closing
        assert len(await active.tools()) == 1
        await active.release()


@pytest.mark.asyncio
async def test_pool_counts_connections_while_startup_is_pending(peer: _Peer) -> None:
    peer.opening_gate = asyncio.Event()
    async with MCPClientManager(MCPClientLimits(max_connections=1)) as manager:
        pending = asyncio.create_task(
            acquire_lease(manager, _server(), MCPAuthorizationContext(key="opening"))
        )
        try:
            async with asyncio.timeout(1):
                while not peer.opened:
                    await asyncio.sleep(0)
            with pytest.raises(MCPPoolCapacityError):
                await acquire_lease(
                    manager, _server(), MCPAuthorizationContext(key="other")
                )
            peer.opening_gate.set()
            lease = await pending
            await lease.release()
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_pool_idle_deadline_releases_resources_without_new_acquisition(
    peer: _Peer,
) -> None:
    async with MCPClientManager(MCPClientLimits(idle_timeout_seconds=0.02)) as manager:
        lease = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="idle")
        )
        await lease.release()
        async with asyncio.timeout(1):
            while _pool_size(manager):
                await asyncio.sleep(0.005)
        assert peer.opened[0].owner_task is not None
        assert peer.opened[0].owner_task.done()


@pytest.mark.asyncio
async def test_cancelled_release_still_returns_its_pool_capacity(peer: _Peer) -> None:
    async with MCPClientManager(MCPClientLimits(max_connections=1)) as manager:
        lease = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="first")
        )
        async with manager._lock:  # pyright: ignore[reportPrivateUsage]
            releasing = asyncio.create_task(lease.release())
            await asyncio.sleep(0)
            releasing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await releasing
        await lease.release()
        replacement = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="second")
        )
        assert len(peer.opened) == 2
        assert peer.opened[0].closing
        await replacement.release()


@pytest.mark.asyncio
async def test_pool_does_not_retire_a_call_after_its_lease_is_released(
    peer: _Peer,
) -> None:
    peer.call_gate = asyncio.Event()
    async with MCPClientManager(
        MCPClientLimits(max_connections=1, idle_timeout_seconds=0.01)
    ) as manager:
        lease = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="active")
        )
        tool = (await lease.tools())[0]
        call = asyncio.create_task(tool.run(tool.args_type()(), CancellationToken()))
        await peer.call_started.wait()
        await lease.release()
        await asyncio.sleep(0.03)
        assert not peer.opened[0].closing
        with pytest.raises(MCPPoolCapacityError):
            await acquire_lease(
                manager, _server(), MCPAuthorizationContext(key="other")
            )
        peer.call_gate.set()
        assert not isinstance(await call, ToolFailure)
        async with asyncio.timeout(1):
            while _pool_size(manager):
                await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_stale_catalog_dispatches_once_on_live_replacement(peer: _Peer) -> None:
    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="identity")
        )
        await lease.tools()
        peer.opened[0].failed = True
        peer.fail_refresh = True
        tool = (await lease.tools())[0]
        assert len(peer.opened) == 2
        result = await tool.run(tool.args_type()(), CancellationToken())
        assert not isinstance(result, ToolFailure)
        assert peer.calls == [1]
        assert peer.opened[0].closing
        await lease.release()


@pytest.mark.asyncio
async def test_cancelled_reconnect_waiter_does_not_cancel_shared_startup(
    peer: _Peer,
) -> None:
    async with MCPClientManager() as manager:
        context = MCPAuthorizationContext(key="identity")
        first = await acquire_lease(manager, _server(), context)
        second = await acquire_lease(manager, _server(), context)
        tool = (await first.tools())[0]
        peer.opened[0].failed = True
        peer.opening_gate = asyncio.Event()
        token = CancellationToken()
        call = asyncio.create_task(tool.run(tool.args_type()(), token))
        async with asyncio.timeout(1):
            while len(peer.opened) < 2:
                await asyncio.sleep(0)
        other_discovery = asyncio.create_task(second.tools())
        token.cancel()
        result = await asyncio.wait_for(call, timeout=0.2)
        assert isinstance(result, ToolFailure)
        assert result.error.kind == "cancelled"
        assert peer.calls == []
        assert not peer.opened[1].closing
        peer.opening_gate.set()
        assert len(await asyncio.wait_for(other_discovery, timeout=1)) == 1
        await first.release()
        await second.release()


@pytest.mark.asyncio
async def test_tool_timeout_includes_reconnect_without_cancelling_shared_owner(
    peer: _Peer,
) -> None:
    async with MCPClientManager() as manager:
        server = replace(_server(), tool_timeout_seconds=0.02)
        lease = await acquire_lease(
            manager, server, MCPAuthorizationContext(key="identity")
        )
        tool = (await lease.tools())[0]
        peer.opened[0].failed = True
        peer.opening_gate = asyncio.Event()
        result = await asyncio.wait_for(
            tool.run(tool.args_type()(), CancellationToken()), timeout=0.3
        )
        assert isinstance(result, ToolFailure)
        assert result.error.kind == "timeout"
        assert len(peer.opened) == 2
        assert not peer.opened[1].closing
        peer.opening_gate.set()
        assert len(await lease.tools()) == 1
        assert peer.calls == []
        await lease.release()


@dataclass
class _ClosingProvider:
    closing: bool = False
    blocked: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)

    async def get_access_token(
        self, server: MCPServer, context: MCPAuthorizationContext
    ) -> MCPAccessToken:
        del server, context
        if self.closing:
            self.blocked.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
        return MCPAccessToken(token=_FIXTURE_VALUE)

    async def invalidate_access_token(
        self, server: MCPServer, context: MCPAuthorizationContext, token: MCPAccessToken
    ) -> None:
        del server, context, token


@pytest.mark.asyncio
async def test_official_sdk_shutdown_bounds_blocked_cleanup_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _ClosingProvider()
    clients: list[httpx2.AsyncClient] = []
    methods: list[str] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET":
            return httpx2.Response(405, request=request)
        payload: dict[str, Any] = json.loads(request.content) if request.content else {}
        method = payload.get("method", "")
        methods.append(method)
        if method == "initialize":
            return httpx2.Response(
                200,
                request=request,
                headers={"Mcp-Session-Id": "fixture-session"},
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "serverInfo": {"name": "fixture", "version": "1"},
                    },
                },
            )
        return httpx2.Response(202, request=request)

    original_client = httpx2.AsyncClient

    def client_factory(**kwargs: Any) -> httpx2.AsyncClient:
        client = original_client(transport=httpx2.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    sdk = replace(
        load_mcp_dependencies(),
        http=SimpleNamespace(
            AsyncClient=client_factory,
            Timeout=httpx2.Timeout,
            Auth=httpx2.Auth,
        ),
    )
    monkeypatch.setattr(mcp_connection, "load_mcp_dependencies", lambda: sdk)
    manager = MCPClientManager(MCPClientLimits(shutdown_timeout_seconds=0.05))
    lease = await acquire_lease(
        manager,
        MCPServer(
            name="cleanup",
            transport=MCPStreamableHTTPTransport(url="https://fixture.test/mcp"),
            authorization=provider,
        ),
        MCPAuthorizationContext(key="identity"),
    )
    await lease.release()
    provider.closing = True
    try:
        await asyncio.wait_for(manager.aclose(), timeout=1)
    except MCPShutdownTimeoutError:
        await asyncio.wait_for(provider.cancelled.wait(), timeout=1)
        await asyncio.wait_for(manager.aclose(), timeout=1)
    assert "initialize" in methods
    assert provider.blocked.is_set()
    assert provider.cancelled.is_set()
    assert clients and all(client.is_closed for client in clients)
    assert _pool_size(manager) == 0
