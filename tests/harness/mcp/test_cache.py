"""Catalog freshness follows each server page's lifetime and local limits."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from mcp import types
from mcp_types.version import LATEST_HANDSHAKE_VERSION, LATEST_MODERN_VERSION

from agentlane.harness.mcp import (
    MCPAuthorizationContext,
    MCPClientManager,
    MCPServer,
    MCPStdioTransport,
)
from agentlane.harness.mcp import (
    _catalog as mcp_catalog,  # pyright: ignore[reportPrivateUsage]
)

from .helpers import ConnectionInstaller, acquire_lease


def _tool(name: str = "read") -> types.Tool:
    return types.Tool(name=name, input_schema={"type": "object"})


@dataclass
class _CatalogServer:
    now: float = 100.0
    protocol_version: str | None = LATEST_MODERN_VERSION
    pages: dict[str | None, types.ListToolsResult] = field(
        default_factory=lambda: {None: types.ListToolsResult(tools=[_tool()])}
    )
    response_delays: dict[str | None, float] = field(
        default_factory=dict[str | None, float]
    )
    list_calls: int = 0
    opened: list[Any] = field(default_factory=list[Any])

    def monotonic(self) -> float:
        return self.now

    async def list_tools(
        self, *, cursor: str | None, cache_mode: str
    ) -> types.ListToolsResult:
        assert cache_mode == "bypass"
        self.list_calls += 1
        self.now += self.response_delays.get(cursor, 0.0)
        return self.pages[cursor]


@pytest.fixture(name="catalog_server")
def fixture_catalog_server(
    monkeypatch: pytest.MonkeyPatch, install_connection: ConnectionInstaller
) -> _CatalogServer:
    state = _CatalogServer()

    async def prepare(connection: Any) -> None:
        state.opened.append(connection)
        connection.protocol_version = state.protocol_version
        connection.client = SimpleNamespace(
            list_tools=state.list_tools,
            session=SimpleNamespace(protocol_version=state.protocol_version),
        )

    install_connection(prepare)
    monkeypatch.setattr(mcp_catalog, "time", SimpleNamespace(monotonic=state.monotonic))
    return state


def _server(local_ttl: float = 300.0) -> MCPServer:
    return MCPServer(
        name="notes",
        transport=MCPStdioTransport(command="fixture"),
        catalog_ttl_seconds=local_ttl,
    )


@pytest.mark.asyncio
async def test_catalog_explicit_zero_ttl_refetches_on_next_access(
    catalog_server: _CatalogServer,
) -> None:
    catalog_server.pages[None] = types.ListToolsResult(tools=[_tool()], ttl_ms=0)
    assert "ttl_ms" in catalog_server.pages[None].model_fields_set

    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="user")
        )
        assert [tool.name for tool in await lease.tools()] == ["notes__read"]
        assert [tool.name for tool in await lease.tools()] == ["notes__read"]
        assert catalog_server.list_calls == 2
        await lease.release()


@pytest.mark.parametrize(
    ("remote_ttl_ms", "local_ttl", "effective_ttl"),
    [(5000, 300.0, 5.0), (10000, 2.0, 2.0)],
)
@pytest.mark.asyncio
async def test_catalog_uses_shorter_remote_or_local_ttl(
    catalog_server: _CatalogServer,
    remote_ttl_ms: int,
    local_ttl: float,
    effective_ttl: float,
) -> None:
    catalog_server.pages[None] = types.ListToolsResult(
        tools=[_tool()], ttl_ms=remote_ttl_ms
    )

    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(local_ttl), MCPAuthorizationContext(key="user")
        )
        await lease.tools()
        catalog_server.now = 100.0 + effective_ttl - 0.001
        await lease.tools()
        assert catalog_server.list_calls == 1
        catalog_server.now = 100.0 + effective_ttl
        await lease.tools()
        assert catalog_server.list_calls == 2
        await lease.release()


@pytest.mark.asyncio
async def test_catalog_omitted_remote_ttl_uses_configured_fallback(
    catalog_server: _CatalogServer,
) -> None:
    assert "ttl_ms" not in catalog_server.pages[None].model_fields_set

    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(local_ttl=4.0), MCPAuthorizationContext(key="user")
        )
        await lease.tools()
        catalog_server.now = 103.999
        await lease.tools()
        assert catalog_server.list_calls == 1
        catalog_server.now = 104.0
        await lease.tools()
        assert catalog_server.list_calls == 2
        await lease.release()


@pytest.mark.parametrize(
    "protocol_version", [LATEST_HANDSHAKE_VERSION, None, "unknown"]
)
@pytest.mark.parametrize("remote_ttl_ms", [0, 1000])
@pytest.mark.asyncio
async def test_catalog_non_modern_protocol_uses_configured_fallback(
    catalog_server: _CatalogServer,
    protocol_version: str | None,
    remote_ttl_ms: int,
) -> None:
    catalog_server.protocol_version = protocol_version
    catalog_server.pages[None] = types.ListToolsResult(
        tools=[_tool()], ttl_ms=remote_ttl_ms
    )

    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(local_ttl=4.0), MCPAuthorizationContext(key="user")
        )
        await lease.tools()
        catalog_server.now = 103.999
        await lease.tools()
        assert catalog_server.list_calls == 1
        catalog_server.now = 104.0
        await lease.tools()
        assert catalog_server.list_calls == 2
        await lease.release()


@pytest.mark.parametrize(
    ("first_ttl_ms", "second_ttl_ms", "expires_at"),
    [(2000, 10000, 102.0), (10000, 2000, 103.0)],
)
@pytest.mark.asyncio
async def test_catalog_pages_use_earliest_deadline_from_page_receipt(
    catalog_server: _CatalogServer,
    first_ttl_ms: int,
    second_ttl_ms: int,
    expires_at: float,
) -> None:
    catalog_server.pages = {
        None: types.ListToolsResult(
            tools=[_tool()], ttl_ms=first_ttl_ms, next_cursor="second"
        ),
        "second": types.ListToolsResult(tools=[_tool("write")], ttl_ms=second_ttl_ms),
    }
    catalog_server.response_delays = {"second": 1.0}

    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="user")
        )
        assert [tool.name for tool in await lease.tools()] == [
            "notes__read",
            "notes__write",
        ]
        assert catalog_server.now == 101.0
        assert catalog_server.list_calls == 2
        catalog_server.now = expires_at - 0.001
        await lease.tools()
        assert catalog_server.list_calls == 2
        catalog_server.now = expires_at
        await lease.tools()
        assert catalog_server.list_calls == 4
        await lease.release()


@pytest.mark.asyncio
async def test_catalog_list_changed_overrides_positive_server_ttl(
    catalog_server: _CatalogServer,
) -> None:
    catalog_server.pages[None] = types.ListToolsResult(tools=[_tool()], ttl_ms=10000)

    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="user")
        )
        await lease.tools()
        catalog_server.pages[None] = types.ListToolsResult(
            tools=[_tool("updated")], ttl_ms=10000
        )
        catalog_server.opened[0].catalog_revision += 1
        assert [tool.name for tool in await lease.tools()] == ["notes__updated"]
        assert catalog_server.list_calls == 2
        await lease.release()


@pytest.mark.asyncio
async def test_catalog_empty_with_positive_server_ttl_is_cached(
    catalog_server: _CatalogServer,
) -> None:
    catalog_server.pages[None] = types.ListToolsResult(tools=[], ttl_ms=2000)

    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="user")
        )
        assert await lease.tools() == ()
        catalog_server.now = 101.999
        assert await lease.tools() == ()
        assert catalog_server.list_calls == 1
        catalog_server.now = 102.0
        assert await lease.tools() == ()
        assert catalog_server.list_calls == 2
        await lease.release()
