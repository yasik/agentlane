"""Shared transports preserve each caller's opaque authorization context."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, cast

import pytest
from mcp import types
from mcp.server import Server, ServerRequestContext
from starlette.requests import Request
from starlette.types import Message, Receive, Scope, Send

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.mcp import (
    MCPAccessToken,
    MCPAuthorizationContext,
    MCPClientManager,
    MCPServer,
    MCPStreamableHTTPTransport,
    MCPToolsShim,
)
from agentlane.harness.mcp._client import (
    _PoolEntry,  # pyright: ignore[reportPrivateUsage]
)
from agentlane.harness.mcp._connection import MCPConnection
from agentlane.harness.mcp._operation import mcp_operation
from agentlane.models import ToolFailure
from agentlane.runtime import CancellationToken

from ..tools_test_utils import SequenceModel, make_assistant_response, make_tool_call
from .helpers import acquire_lease, http_server


def _token(value: str, revision: int = 0) -> str:
    return f"lease-context-test-{value}-{revision}"


@dataclass
class _ContextProvider:
    shared_token: bool = False
    lookups: list[str] = field(default_factory=list[str])
    invalidations: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    revisions: dict[str, int] = field(default_factory=dict[str, int])

    async def get_access_token(
        self, server: MCPServer, context: MCPAuthorizationContext
    ) -> MCPAccessToken:
        del server
        assert isinstance(context.value, str)
        self.lookups.append(context.value)
        value = "shared" if self.shared_token else context.value
        return MCPAccessToken(token=_token(value, self.revisions.get(value, 0)))

    async def invalidate_access_token(
        self, server: MCPServer, context: MCPAuthorizationContext, token: MCPAccessToken
    ) -> None:
        del server
        assert isinstance(context.value, str)
        self.invalidations.append((context.value, token.token))
        value = "shared" if self.shared_token else context.value
        self.revisions[value] = self.revisions.get(value, 0) + 1


@dataclass
class _ContextPeer:
    provider: _ContextProvider = field(default_factory=_ContextProvider)
    legacy: bool = False
    reject_calls: int = 0
    fail_calls: bool = False
    requests: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    calls: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    listed: list[str] = field(default_factory=list[str])


@asynccontextmanager
async def _context_peer(port: int, peer: _ContextPeer) -> AsyncIterator[MCPServer]:
    async def list_tools(
        context: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        del params
        credential = cast(Request, context.request).headers["authorization"]
        peer.listed.append(credential)
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name="read",
                    description="catalog " + credential.rsplit("-", 2)[-2],
                    input_schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                )
            ],
            ttl_ms=300_000,
        )

    async def call_tool(
        context: ServerRequestContext[Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        credential = cast(Request, context.request).headers["authorization"]
        assert params.arguments is not None
        peer.calls.append((str(params.arguments["value"]), credential))
        return types.CallToolResult(content=[types.TextContent(text="done")])

    sdk_server = Server(
        "lease-context", on_list_tools=list_tools, on_call_tool=call_tool
    )
    inner = sdk_server.streamable_http_app(
        stateless_http=not peer.legacy, host="127.0.0.1"
    )

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await inner(scope, receive, send)
            return
        method = scope["method"]
        credential = dict(scope["headers"]).get(b"authorization", b"").decode()
        buffered: list[Message] = []
        if method == "POST":
            while True:
                message = await receive()
                buffered.append(message)
                if not message.get("more_body", False):
                    break
            payload = json.loads(
                b"".join(message.get("body", b"") for message in buffered)
            )
            method = payload.get("method", "POST")
        peer.requests.append((method, credential))
        status = None
        if peer.legacy and method == "server/discover":
            status = 404
        elif method == "tools/call" and peer.reject_calls:
            peer.reject_calls -= 1
            status = 401
        elif method == "tools/call" and peer.fail_calls:
            status = 503
        if status is not None:
            await send({"type": "http.response.start", "status": status, "headers": []})
            await send({"type": "http.response.body", "body": b"rejected"})
            return

        async def replay() -> Message:
            return buffered.pop(0) if buffered else await receive()

        await inner(scope, replay, send)

    async with http_server(app, port):
        yield MCPServer(
            name="notes",
            transport=MCPStreamableHTTPTransport(
                url=f"http://127.0.0.1:{port}/mcp", allow_insecure_http=True
            ),
            authorization=peer.provider,
        )


def _context(value: str) -> MCPAuthorizationContext:
    return MCPAuthorizationContext(key="same-user", value=value)


async def _run_agent(server: MCPServer, manager: MCPClientManager, value: str) -> None:
    model = SequenceModel(
        [
            make_assistant_response(
                None,
                tool_calls=[
                    make_tool_call(
                        tool_id=f"call-{value}",
                        name="notes__read",
                        arguments=json.dumps({"value": value}),
                    )
                ],
            ),
            make_assistant_response("complete"),
        ]
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            model=model,
            shims=(
                MCPToolsShim(
                    servers=(server,),
                    client_manager=manager,
                    authorization_context=_context(value),
                ),
            ),
        )
    )
    result = await agent.run("Read the record")
    assert result.final_output == "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent", [False, True])
async def test_shared_http_agents_use_each_lease_context(
    unused_tcp_port: int, concurrent: bool
) -> None:
    # Concurrent callers share valid credentials; distinct tokens would correctly
    # invalidate each other's prepared catalogs. Sequential callers rotate them.
    peer = _ContextPeer(provider=_ContextProvider(shared_token=concurrent))
    async with (
        _context_peer(unused_tcp_port, peer) as server,
        MCPClientManager() as manager,
    ):
        if concurrent:
            await asyncio.gather(
                *(_run_agent(server, manager, value) for value in ("A", "B"))
            )
        else:
            await _run_agent(server, manager, "A")
            await _run_agent(server, manager, "B")
        assert set(peer.provider.lookups) == {"A", "B"}
        assert sorted(peer.calls) == [
            (value, f"Bearer {_token('shared' if concurrent else value)}")
            for value in ("A", "B")
        ]
        assert set(peer.listed) == {
            f"Bearer {_token('shared' if concurrent else value)}"
            for value in ("A", "B")
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("rejections", [1, 2])
async def test_http_retry_invalidates_tokens_with_the_requesting_lease_context(
    unused_tcp_port: int, rejections: int
) -> None:
    peer = _ContextPeer()
    async with (
        _context_peer(unused_tcp_port, peer) as server,
        MCPClientManager() as manager,
    ):
        first = await acquire_lease(manager, server, _context("A"))
        await first.tools()
        await first.release()
        second = await acquire_lease(manager, server, _context("B"))
        tool = (await second.tools())[0]
        peer.reject_calls = rejections
        result = await tool.run(
            tool.args_type().model_validate({"value": "B"}), CancellationToken()
        )
        assert isinstance(result, ToolFailure) is (rejections == 2)
        assert peer.provider.invalidations == [
            ("B", _token("B", revision)) for revision in range(rejections)
        ]
        await second.release()


@pytest.mark.asyncio
async def test_http_lifecycle_keeps_opener_context_after_other_lease_requests(
    unused_tcp_port: int,
) -> None:
    peer = _ContextPeer(legacy=True)
    async with _context_peer(unused_tcp_port, peer) as server:
        async with MCPClientManager() as manager:
            await _run_agent(server, manager, "A")
            await _run_agent(server, manager, "B")
        lifecycle = [
            (method, token)
            for method, token in peer.requests
            if method in {"initialize", "GET", "DELETE"}
        ]
        assert {method for method, _ in lifecycle} == {"initialize", "GET", "DELETE"}
        assert all(token == f"Bearer {_token('A')}" for _, token in lifecycle)
        assert peer.calls == [
            (value, f"Bearer {_token(value)}") for value in ("A", "B")
        ]


def test_nested_mcp_operations_inherit_only_authorization_context() -> None:
    context = _context("A")
    with mcp_operation(authorization_context=context) as outer:
        outer.http_status = 401
        outer.authorization_generation = 7
        outer.expected_authorization_generation = 6
        outer.retry_reason = "unauthorized"
        outer.secrets.add(_token("A"))
        with mcp_operation() as inner:
            assert inner.authorization_context is context
            assert inner.http_status is None
            assert inner.authorization_generation is None
            assert inner.expected_authorization_generation is None
            assert inner.retry_reason is None
            assert inner.secrets == set()
        with mcp_operation(authorization_context=_context("B")) as explicit:
            assert explicit.authorization_context is not context


@pytest.mark.asyncio
async def test_cached_http_catalog_still_checks_the_current_lease_context(
    unused_tcp_port: int,
) -> None:
    peer = _ContextPeer(provider=_ContextProvider(shared_token=True))
    async with (
        _context_peer(unused_tcp_port, peer) as server,
        MCPClientManager() as manager,
    ):
        first = await acquire_lease(manager, server, _context("A"))
        await first.tools()
        await first.release()
        second = await acquire_lease(manager, server, _context("B"))
        before = len(peer.provider.lookups)
        await second.tools()
        assert peer.provider.lookups[before:] == ["B"]
        assert peer.listed == [f"Bearer {_token('shared')}"]
        await second.release()


@pytest.mark.asyncio
async def test_cached_catalog_matches_the_waiting_leases_preflight_generation(
    unused_tcp_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    peer = _ContextPeer()
    async with (
        _context_peer(unused_tcp_port, peer) as server,
        MCPClientManager() as manager,
    ):
        first = await acquire_lease(manager, server, _context("A"))
        second = await acquire_lease(manager, server, _context("B"))
        await first.tools()
        preflight_finished = asyncio.Event()
        resume = asyncio.Event()
        connect = manager._connection  # pyright: ignore[reportPrivateUsage]

        async def hold_second(
            entry: _PoolEntry, *, authorization_context: MCPAuthorizationContext
        ) -> MCPConnection:
            if authorization_context.value == "B":
                preflight_finished.set()
                await resume.wait()
            return await connect(entry, authorization_context=authorization_context)

        monkeypatch.setattr(manager, "_connection", hold_second)
        discovery = asyncio.create_task(second.tools())
        try:
            await asyncio.wait_for(preflight_finished.wait(), timeout=1)
            # A refreshes while B is between preflight and cache access. B must
            # not publish A's newly current generation as its own cached result.
            await first.tools()
            resume.set()
            tool = (await asyncio.wait_for(discovery, timeout=1))[0]
            assert tool.description.endswith("catalog B")
            assert peer.listed == [
                f"Bearer {_token(value)}" for value in ("A", "A", "B")
            ]
        finally:
            resume.set()
            discovery.cancel()
            await asyncio.gather(discovery, return_exceptions=True)
            await first.release()
            await second.release()


@pytest.mark.asyncio
async def test_replacement_transport_captures_its_own_opening_context(
    unused_tcp_port: int,
) -> None:
    peer = _ContextPeer(legacy=True)
    async with _context_peer(unused_tcp_port, peer) as server:
        async with MCPClientManager() as manager:
            first = await acquire_lease(manager, server, _context("A"))
            await first.tools()
            await first.release()
            second = await acquire_lease(manager, server, _context("B"))
            tool = (await second.tools())[0]
            peer.fail_calls = True
            result = await tool.run(
                tool.args_type().model_validate({"value": "B"}), CancellationToken()
            )
            assert isinstance(result, ToolFailure)
            assert result.error.kind == "mcp_transport"
            peer.fail_calls = False
            assert (await second.tools())[0].description.endswith("catalog B")
            await second.release()
        assert [token for method, token in peer.requests if method == "initialize"] == [
            f"Bearer {_token(value)}" for value in ("A", "B")
        ]
        assert [token for method, token in peer.requests if method == "DELETE"] == [
            f"Bearer {_token(value)}" for value in ("A", "B")
        ]
