"""Shared catalog waits consume each lease's discovery deadline."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import pytest
from mcp import types
from mcp.server import Server, ServerRequestContext

from agentlane.harness.mcp import (
    MCPAuthorizationContext,
    MCPClientManager,
    MCPDiscoveryError,
    MCPServer,
    MCPStreamableHTTPTransport,
)
from agentlane.harness.mcp import (
    _client as mcp_client,  # pyright: ignore[reportPrivateUsage]
)
from agentlane.harness.mcp import (
    _connection as mcp_connection,  # pyright: ignore[reportPrivateUsage]
)

from .helpers import acquire_lease, http_server


@dataclass
class _DiscoveryPeer:
    delay_seconds: float = 0.0
    tool_name: str = "initial"
    list_calls: int = 0


@asynccontextmanager
async def _discovery_peer(
    port: int, peer: _DiscoveryPeer, *, discovery_timeout_seconds: float
) -> AsyncIterator[MCPServer]:
    async def list_tools(
        context: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        del context, params
        peer.list_calls += 1
        await asyncio.sleep(peer.delay_seconds)
        return types.ListToolsResult(
            tools=[types.Tool(name=peer.tool_name, input_schema={"type": "object"})],
            ttl_ms=0,
        )

    sdk_server = Server("shared-discovery", on_list_tools=list_tools)
    app = sdk_server.streamable_http_app(stateless_http=True, host="127.0.0.1")
    async with http_server(app, port):
        yield MCPServer(
            name="notes",
            transport=MCPStreamableHTTPTransport(
                url=f"http://127.0.0.1:{port}/mcp", allow_insecure_http=True
            ),
            discovery_timeout_seconds=discovery_timeout_seconds,
        )


def _connection(lease: mcp_client.MCPClientLease) -> mcp_connection.MCPConnection:
    connection = lease._entry.connection  # pyright: ignore[reportPrivateUsage]
    assert connection is not None
    return connection


@pytest.mark.asyncio
async def test_http_shared_catalog_waits_use_each_lease_discovery_deadline(
    unused_tcp_port: int,
) -> None:
    peer = _DiscoveryPeer(delay_seconds=0.12)
    async with (
        _discovery_peer(unused_tcp_port, peer, discovery_timeout_seconds=0.2) as server,
        MCPClientManager() as manager,
        asyncio.timeout(5),
    ):
        leases = await asyncio.gather(
            *(
                acquire_lease(manager, server, MCPAuthorizationContext(key="user"))
                for _ in range(4)
            )
        )
        connection = _connection(leases[0])
        assert all(_connection(lease) is connection for lease in leases)
        finished_at: list[float] = []
        started_at = asyncio.get_running_loop().time()

        async def discover(lease: mcp_client.MCPClientLease) -> tuple[str, ...]:
            try:
                return tuple(tool.name for tool in await lease.tools())
            finally:
                finished_at.append(asyncio.get_running_loop().time() - started_at)

        results = await asyncio.gather(
            *(discover(lease) for lease in leases), return_exceptions=True
        )
        assert results.count(("notes__initial",)) == 1, finished_at
        errors = [result for result in results if isinstance(result, MCPDiscoveryError)]
        assert len(errors) == 3
        assert all(
            error.failure_kind == "timeout" and error.retryable for error in errors
        )
        assert max(finished_at) < 0.35, finished_at
        assert peer.list_calls == 2
        assert not connection.failed

        peer.tool_name = "recovered"
        assert [tool.name for tool in await leases[-1].tools()] == ["notes__recovered"]
        assert _connection(leases[-1]) is connection
        assert peer.list_calls == 3
        await asyncio.gather(*(lease.release() for lease in leases))


@pytest.mark.asyncio
async def test_shared_lock_timeout_keeps_only_the_calling_lease_stale_catalog(
    unused_tcp_port: int,
) -> None:
    peer = _DiscoveryPeer()
    async with (
        _discovery_peer(
            unused_tcp_port, peer, discovery_timeout_seconds=0.05
        ) as server,
        MCPClientManager() as manager,
        asyncio.timeout(5),
    ):
        owner = await acquire_lease(
            manager, server, MCPAuthorizationContext(key="user")
        )
        assert [tool.name for tool in await owner.tools()] == ["notes__initial"]
        newcomer = await acquire_lease(
            manager, server, MCPAuthorizationContext(key="user")
        )
        connection = _connection(owner)
        assert _connection(newcomer) is connection

        # Hold the real catalog lock to isolate timeout policy from HTTP latency.
        async with connection.catalog_lock:
            calls = tuple(
                asyncio.create_task(lease.tools()) for lease in (owner, newcomer)
            )
            try:
                _, pending = await asyncio.wait(calls, timeout=0.3)
                assert (
                    not pending
                ), "Catalog lock waits exceeded the discovery deadline."
                retained = await calls[0]
                assert [tool.name for tool in retained] == ["notes__initial"]
                with pytest.raises(MCPDiscoveryError) as caught:
                    await calls[1]
                assert caught.value.failure_kind == "timeout"
                assert caught.value.retryable
                assert peer.list_calls == 1
                assert not connection.failed
            finally:
                for call in calls:
                    call.cancel()
                await asyncio.gather(*calls, return_exceptions=True)

        peer.tool_name = "recovered"
        for lease in (owner, newcomer):
            assert [tool.name for tool in await lease.tools()] == ["notes__recovered"]
            assert _connection(lease) is connection
        assert peer.list_calls == 3
        await asyncio.gather(owner.release(), newcomer.release())
