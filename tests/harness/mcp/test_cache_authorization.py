"""Catalogs must follow current authorization, including HTTP token refresh."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import pytest
from mcp import types
from mcp.server import Server, ServerRequestContext
from starlette.requests import Request
from starlette.types import Message, Receive, Scope, Send

from agentlane.harness.mcp import (
    MCPAccessToken,
    MCPAuthorizationContext,
    MCPAuthorizationError,
    MCPClientManager,
    MCPDiscoveryError,
    MCPServer,
    MCPStreamableHTTPTransport,
)
from agentlane.harness.mcp import (
    _client as mcp_client,  # pyright: ignore[reportPrivateUsage]
)
from agentlane.harness.mcp._catalog import get_catalog
from agentlane.models import ToolFailure
from agentlane.runtime import CancellationToken

from .helpers import ConnectionInstaller, acquire_lease, http_server


def _token(label: str) -> str:
    return f"cache-test-credential-{label}"


@dataclass
class _Provider:
    token: str = field(default_factory=lambda: _token("initial"))
    scopes: tuple[str, ...] = ("read", "write")
    failure: bool = False
    rotate_on_invalidation: bool = True
    mint_each_lookup: bool = False
    lookups: int = 0
    lookup_started: asyncio.Event | None = None
    lookup_release: asyncio.Event | None = None
    lookup_cancelled: bool = False
    invalidated: list[str] = field(default_factory=list[str])

    async def get_access_token(
        self, server: MCPServer, context: MCPAuthorizationContext
    ) -> MCPAccessToken:
        del server, context
        if self.lookup_started is not None and self.lookup_release is not None:
            self.lookup_started.set()
            try:
                await self.lookup_release.wait()
            except asyncio.CancelledError:
                self.lookup_cancelled = True
                raise
        if self.failure:
            raise RuntimeError(_token("provider-error"))
        self.lookups += 1
        if self.mint_each_lookup:
            self.token = _token(f"minted-{self.lookups}")
        return MCPAccessToken(token=self.token, scopes=self.scopes)

    async def invalidate_access_token(
        self,
        server: MCPServer,
        context: MCPAuthorizationContext,
        token: MCPAccessToken,
    ) -> None:
        del server, context
        self.invalidated.append(token.token)
        if self.rotate_on_invalidation:
            self.token = _token("rotated")


@dataclass
class _FakePeer:
    name: str = "read_initial"
    list_calls: int = 0
    open_calls: int = 0
    list_error: Exception | None = None
    call_error: Exception | None = None
    open_error: Exception | None = None

    async def list_tools(self, **kwargs: Any) -> types.ListToolsResult:
        del kwargs
        self.list_calls += 1
        if self.list_error is not None:
            raise self.list_error
        return types.ListToolsResult(
            tools=[types.Tool(name=self.name, input_schema={"type": "object"})],
            ttl_ms=300_000,
            cache_scope="private",
        )

    async def call_tool(self, *args: Any, **kwargs: Any) -> types.CallToolResult:
        del args, kwargs
        if self.call_error is not None:
            raise self.call_error
        return types.CallToolResult(content=[])


@pytest.fixture(name="fake_peer")
def fixture_fake_peer(install_connection: ConnectionInstaller) -> _FakePeer:
    peer = _FakePeer()

    async def prepare(connection: Any) -> None:
        peer.open_calls += 1
        if peer.open_error is not None:
            raise peer.open_error
        connection.client = SimpleNamespace(
            list_tools=peer.list_tools,
            call_tool=peer.call_tool,
            session=SimpleNamespace(protocol_version="2026-07-28"),
        )
        connection.protocol_version = "2026-07-28"

    install_connection(prepare)
    return peer


def _server(provider: _Provider, port: int | None = None) -> MCPServer:
    transport = (
        MCPStreamableHTTPTransport(
            url=f"http://127.0.0.1:{port}/mcp", allow_insecure_http=True
        )
        if port is not None
        else MCPStreamableHTTPTransport(url="https://example.test/mcp")
    )
    return MCPServer(
        name="notes",
        transport=transport,
        authorization=provider,
        discovery_timeout_seconds=2,
        tool_timeout_seconds=2,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["token", "scopes"])
async def test_fresh_catalog_is_refetched_when_authorization_changes(
    fake_peer: _FakePeer, change: str
) -> None:
    provider = _Provider()
    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(provider), MCPAuthorizationContext(key="u")
        )
        assert [tool.name for tool in await lease.tools()] == ["notes__read_initial"]
        if change == "token":
            provider.token = _token("rotated")
        else:
            provider.scopes = ("read",)
        fake_peer.name = "read_rotated"
        assert [tool.name for tool in await lease.tools()] == ["notes__read_rotated"]
        assert fake_peer.list_calls == 2
        assert fake_peer.open_calls == 1
        await lease.release()


@pytest.mark.asyncio
async def test_scope_order_does_not_invalidate_same_authorization(
    fake_peer: _FakePeer,
) -> None:
    provider = _Provider()
    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(provider), MCPAuthorizationContext(key="u")
        )
        await lease.tools()
        provider.scopes = tuple(reversed(provider.scopes))
        await lease.tools()
        assert fake_peer.list_calls == 1
        await lease.release()


@pytest.mark.asyncio
async def test_provider_error_blocks_fresh_catalog_and_later_stale_fallback(
    fake_peer: _FakePeer,
) -> None:
    provider = _Provider()
    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(provider), MCPAuthorizationContext(key="u")
        )
        await lease.tools()
        provider.failure = True
        with pytest.raises(MCPAuthorizationError) as caught:
            await lease.tools()
        assert _token("provider-error") not in str(caught.value)
        provider.failure = False
        fake_peer.list_error = ConnectionError("temporarily offline")
        with pytest.raises(MCPDiscoveryError):
            await lease.tools()
        await lease.release()


@pytest.mark.asyncio
async def test_rotated_authorization_cannot_use_stale_catalog_after_refresh_failure(
    fake_peer: _FakePeer,
) -> None:
    provider = _Provider()
    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(provider), MCPAuthorizationContext(key="u")
        )
        await lease.tools()
        provider.token = _token("rotated")
        fake_peer.list_error = TimeoutError("temporarily offline")
        with pytest.raises(MCPDiscoveryError):
            await lease.tools()
        fake_peer.list_error = None
        fake_peer.name = "read_rotated"
        assert [tool.name for tool in await lease.tools()] == ["notes__read_rotated"]
        await lease.release()


@pytest.mark.asyncio
async def test_reconnect_failure_checks_authorization_before_stale_fallback(
    fake_peer: _FakePeer,
) -> None:
    provider = _Provider()
    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(provider), MCPAuthorizationContext(key="u")
        )
        tool = (await lease.tools())[0]
        fake_peer.call_error = ConnectionError("disconnected")
        result = await tool.run(tool.args_type()(), CancellationToken())
        assert isinstance(result, ToolFailure)
        assert result.error.kind == "mcp_transport"
        provider.token = _token("rotated")
        fake_peer.open_error = ConnectionError("reconnect failed")
        with pytest.raises(MCPDiscoveryError):
            await lease.tools()
        assert fake_peer.open_calls == 2
        await lease.release()


@pytest.mark.asyncio
async def test_authorization_change_before_discovery_publication_rejects_catalog(
    fake_peer: _FakePeer, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _Provider()
    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(provider), MCPAuthorizationContext(key="u")
        )
        catalog = get_catalog

        async def invalidate_after_catalog(
            connection: Any, *, expected_authorization_generation: int
        ) -> Any:
            snapshot = await catalog(
                connection,
                expected_authorization_generation=expected_authorization_generation,
            )
            # Run after the discovery task finishes, before its caller resumes.
            asyncio.get_running_loop().call_soon(connection.authorization.reject)
            return snapshot

        monkeypatch.setattr(mcp_client, "get_catalog", invalidate_after_catalog)
        with pytest.raises(MCPAuthorizationError):
            await lease.tools()
        assert fake_peer.list_calls == 1

        monkeypatch.setattr(mcp_client, "get_catalog", catalog)
        assert [tool.name for tool in await lease.tools()] == ["notes__read_initial"]
        assert fake_peer.list_calls == 2
        await lease.release()


@pytest.mark.asyncio
async def test_shutdown_cancels_and_joins_catalog_authorization_lookup(
    fake_peer: _FakePeer,
) -> None:
    del fake_peer
    provider = _Provider()
    manager = MCPClientManager()
    lease = await acquire_lease(
        manager, _server(provider), MCPAuthorizationContext(key="u")
    )
    provider.lookup_started, provider.lookup_release = asyncio.Event(), asyncio.Event()
    discovery = asyncio.create_task(lease.tools())
    try:
        await asyncio.wait_for(provider.lookup_started.wait(), timeout=2)
        await asyncio.wait_for(manager.aclose(), timeout=2)
        assert provider.lookup_cancelled
        assert discovery.done()
        with pytest.raises(asyncio.CancelledError):
            await discovery
    finally:
        discovery.cancel()
        await asyncio.gather(discovery, return_exceptions=True)
        await lease.release()
        await manager.aclose()


@dataclass
class _HTTPPeer:
    provider: _Provider = field(default_factory=_Provider)
    paginated: bool = False
    reject_initial_calls: bool = False
    reject_initial_second_page: bool = False
    call_status: int | None = None
    list_failure: bool = False
    ttl_ms: int = 300_000
    list_started: asyncio.Event | None = None
    list_release: asyncio.Event | None = None
    denied: int = 0
    calls: int = 0
    listed: list[tuple[str, str | None]] = field(
        default_factory=list[tuple[str, str | None]]
    )


@asynccontextmanager
async def _http_peer(port: int, peer: _HTTPPeer) -> AsyncIterator[MCPServer]:
    async def list_tools(
        context: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        headers = cast(Request, context.request).headers
        identity = (
            "initial"
            if headers.get("authorization") == f"Bearer {_token('initial')}"
            else "rotated"
        )
        cursor = params.cursor if params is not None else None
        peer.listed.append((identity, cursor))
        if peer.list_started is not None and peer.list_release is not None:
            started, release = peer.list_started, peer.list_release
            peer.list_started = None
            started.set()
            await release.wait()
        name = f"read_{identity}"
        if peer.paginated:
            name = f"{'first' if cursor is None else 'second'}_{identity}"
        return types.ListToolsResult(
            tools=[types.Tool(name=name, input_schema={"type": "object"})],
            next_cursor="next" if peer.paginated and cursor is None else None,
            ttl_ms=peer.ttl_ms,
            cache_scope="private",
        )

    async def call_tool(
        context: ServerRequestContext[Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        del context, params
        return types.CallToolResult(content=[types.TextContent(text="done")])

    sdk_server = Server("cache-auth", on_list_tools=list_tools, on_call_tool=call_tool)
    inner = sdk_server.streamable_http_app(stateless_http=True, host="127.0.0.1")

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await inner(scope, receive, send)
            return
        buffered: list[Message] = []
        while True:
            message = await receive()
            buffered.append(message)
            if not message.get("more_body", False):
                break
        payload = json.loads(b"".join(message.get("body", b"") for message in buffered))
        method = payload.get("method")
        if method == "tools/call":
            peer.calls += 1
        initial = (
            dict(scope.get("headers", [])).get(b"authorization")
            == f"Bearer {_token('initial')}".encode()
        )
        reject = initial and (
            (peer.reject_initial_calls and method == "tools/call")
            or (
                peer.reject_initial_second_page
                and method == "tools/list"
                and payload.get("params", {}).get("cursor") == "next"
            )
        )
        status = None
        if method == "tools/list" and peer.list_failure:
            status = 503
        elif method == "tools/call" and peer.call_status is not None:
            status = peer.call_status
        elif reject:
            status = 401
        if status is not None:
            peer.denied += 1
            await send({"type": "http.response.start", "status": status, "headers": []})
            await send({"type": "http.response.body", "body": b"Request rejected"})
            return

        async def replay() -> Message:
            return buffered.pop(0) if buffered else await receive()

        await inner(scope, replay, send)

    async with http_server(app, port):
        yield _server(peer.provider, port)


@pytest.mark.asyncio
async def test_http_successful_token_retry_invalidates_fresh_catalog(
    unused_tcp_port: int,
) -> None:
    peer = _HTTPPeer(reject_initial_calls=True)
    async with (
        _http_peer(unused_tcp_port, peer) as server,
        MCPClientManager() as manager,
    ):
        lease = await acquire_lease(manager, server, MCPAuthorizationContext(key="u"))
        tool = (await lease.tools())[0]
        assert tool.name == "notes__read_initial"
        result = await tool.run(tool.args_type()(), CancellationToken())
        assert not isinstance(result, ToolFailure)
        assert peer.calls == 2
        assert peer.denied == 1
        assert peer.provider.invalidated == [_token("initial")]
        assert [tool.name for tool in await lease.tools()] == ["notes__read_rotated"]
        assert peer.listed == [("initial", None), ("rotated", None)]
        await lease.release()


@pytest.mark.asyncio
async def test_http_single_page_accepts_provider_minting_each_lookup(
    unused_tcp_port: int,
) -> None:
    peer = _HTTPPeer(provider=_Provider(mint_each_lookup=True))
    async with (
        _http_peer(unused_tcp_port, peer) as server,
        MCPClientManager() as manager,
    ):
        lease = await acquire_lease(manager, server, MCPAuthorizationContext(key="u"))
        assert [tool.name for tool in await lease.tools()] == ["notes__read_rotated"]
        assert [tool.name for tool in await lease.tools()] == ["notes__read_rotated"]
        assert peer.listed == [("rotated", None), ("rotated", None)]
        assert peer.provider.lookups >= 4
        await lease.release()


@pytest.mark.asyncio
async def test_http_token_retry_during_pagination_never_publishes_mixed_catalog(
    unused_tcp_port: int,
) -> None:
    peer = _HTTPPeer(paginated=True, reject_initial_second_page=True)
    async with (
        _http_peer(unused_tcp_port, peer) as server,
        MCPClientManager() as manager,
    ):
        lease = await acquire_lease(manager, server, MCPAuthorizationContext(key="u"))
        try:
            tools = await lease.tools()
        except MCPAuthorizationError:
            # A client may stop the changed discovery or restart this read-only
            # operation; neither path may expose tools from both credentials.
            tools = await lease.tools()
        assert [tool.name for tool in tools] == [
            "notes__first_rotated",
            "notes__second_rotated",
        ]
        assert peer.denied == 1
        assert peer.provider.invalidated == [_token("initial")]
        assert ("rotated", None) in peer.listed
        await lease.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_http_authorization_rejection_forbids_later_stale_fallback(
    unused_tcp_port: int, status: int
) -> None:
    peer = _HTTPPeer(provider=_Provider(rotate_on_invalidation=False))
    async with (
        _http_peer(unused_tcp_port, peer) as server,
        MCPClientManager() as manager,
    ):
        lease = await acquire_lease(manager, server, MCPAuthorizationContext(key="u"))
        tool = (await lease.tools())[0]
        peer.call_status = status
        result = await tool.run(tool.args_type()(), CancellationToken())
        assert isinstance(result, ToolFailure)
        assert result.error.kind == "mcp_authorization"
        assert peer.calls == (2 if status == 401 else 1)
        peer.call_status = None
        peer.list_failure = True
        with pytest.raises(MCPDiscoveryError):
            await lease.tools()
        await lease.release()


@pytest.mark.asyncio
async def test_http_concurrent_token_retry_discards_inflight_old_catalog(
    unused_tcp_port: int,
) -> None:
    peer = _HTTPPeer(reject_initial_calls=True, ttl_ms=0)
    async with (
        _http_peer(unused_tcp_port, peer) as server,
        MCPClientManager() as manager,
    ):
        lease = await acquire_lease(manager, server, MCPAuthorizationContext(key="u"))
        tool = (await lease.tools())[0]
        started, release = asyncio.Event(), asyncio.Event()
        peer.list_started, peer.list_release = started, release
        discovery = asyncio.create_task(lease.tools())
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            result = await tool.run(tool.args_type()(), CancellationToken())
            assert not isinstance(result, ToolFailure)
            release.set()
            try:
                tools = await asyncio.wait_for(discovery, timeout=2)
            except MCPAuthorizationError:
                tools = await lease.tools()
            assert [tool.name for tool in tools] == ["notes__read_rotated"]
            assert peer.calls == 2
            assert peer.listed[-1] == ("rotated", None)
        finally:
            release.set()
            discovery.cancel()
            await asyncio.gather(discovery, return_exceptions=True)
        await lease.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["token", "scopes", "mint", "none"])
async def test_http_authorization_is_checked_before_first_tool_request(
    unused_tcp_port: int, change: str
) -> None:
    peer = _HTTPPeer()
    async with (
        _http_peer(unused_tcp_port, peer) as server,
        MCPClientManager() as manager,
    ):
        lease = await acquire_lease(manager, server, MCPAuthorizationContext(key="u"))
        tool = (await lease.tools())[0]
        if change == "token":
            peer.provider.token = _token("rotated")
        elif change == "scopes":
            peer.provider.scopes = ("read",)
        elif change == "mint":
            peer.provider.mint_each_lookup = True

        result = await tool.run(tool.args_type()(), CancellationToken())
        if change == "none":
            assert not isinstance(result, ToolFailure)
            assert peer.calls == 1
        else:
            assert isinstance(result, ToolFailure)
            assert result.error.kind == "mcp_authorization"
            assert peer.calls == 0
        assert peer.listed == [("initial", None)]
        await lease.release()
