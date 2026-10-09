"""Connection and catalog regressions at real and controlled SDK boundaries."""

import asyncio
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mcp import types

from agentlane.harness.mcp import (
    MCPAuthorizationContext,
    MCPAuthorizationError,
    MCPClientManager,
    MCPDiscoveryError,
    MCPError,
    MCPResultPolicy,
    MCPServer,
    MCPStdioTransport,
    MCPToolFilter,
)
from agentlane.runtime import CancellationToken

from .helpers import ConnectionInstaller, acquire_lease


@dataclass
class _FakeState:
    tools: list[types.Tool] = field(
        default_factory=lambda: [
            types.Tool(name="read", input_schema={"type": "object"})
        ]
    )
    list_error: Exception | None = None
    list_calls: int = 0
    opened: list[Any] = field(default_factory=list[Any])
    gate: asyncio.Event | None = None

    async def list_tools(self, **kwargs: object) -> types.ListToolsResult:
        del kwargs
        self.list_calls += 1
        if self.list_error is not None:
            raise self.list_error
        return types.ListToolsResult(tools=self.tools)


@pytest.fixture(name="fake_mcp")
def fixture_fake_mcp(install_connection: ConnectionInstaller) -> _FakeState:
    state = _FakeState()

    async def prepare(connection: Any) -> None:
        state.opened.append(connection)
        if state.gate is not None:
            await state.gate.wait()
        connection.client = SimpleNamespace(
            list_tools=state.list_tools,
            session=SimpleNamespace(protocol_version="test"),
        )

    install_connection(prepare)
    return state


def _server(**kwargs: Any) -> MCPServer:
    return MCPServer(
        name="notes", transport=MCPStdioTransport(command="fixture"), **kwargs
    )


@pytest.mark.asyncio
async def test_long_tool_aliases_fit_provider_limits_and_call_original_names(
    fake_mcp: _FakeState,
) -> None:
    remote_names = ["x" * 64 + suffix for suffix in ("first", "second")]
    fake_mcp.tools = [
        types.Tool(name=name, input_schema={"type": "object"}) for name in remote_names
    ]
    calls: list[str] = []

    async def call_tool(
        name: str, arguments: dict[str, Any], **kwargs: Any
    ) -> types.CallToolResult:
        del arguments, kwargs
        calls.append(name)
        return types.CallToolResult(content=[types.TextContent(text="done")])

    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="user")
        )
        fake_mcp.opened[0].client.call_tool = call_tool
        tools = await lease.tools()
        names = [tool.name for tool in tools]
        assert all(len(name) <= 64 for name in names)
        assert len(set(names)) == 2
        assert all(name.startswith("notes__") for name in names)
        assert [tool.name for tool in await lease.tools()] == names
        for tool in tools:
            await tool.run(tool.args_type()(), CancellationToken())
        assert calls == remote_names
        await lease.release()


@pytest.mark.asyncio
async def test_acquire_concurrent_runs_share_one_connection(
    fake_mcp: _FakeState,
) -> None:
    async with MCPClientManager() as manager:
        server = _server()
        context = MCPAuthorizationContext(key="user")
        first, second = await asyncio.gather(
            acquire_lease(manager, server, context),
            acquire_lease(manager, server, context),
        )
        assert len(fake_mcp.opened) == 1
        await first.release()
        await first.release()
        assert [tool.name for tool in await second.tools()] == ["notes__read"]
        assert not fake_mcp.opened[0].closing
        await second.release()


@pytest.mark.parametrize(
    "change",
    [
        {"tools": MCPToolFilter(exclude=("read",))},
        {"result_policy": MCPResultPolicy(max_text_chars=128)},
        {"required": False},
        {"connect_timeout_seconds": 1},
        {"discovery_timeout_seconds": 1},
        {"tool_timeout_seconds": 1},
        {"catalog_ttl_seconds": 1},
    ],
)
@pytest.mark.asyncio
async def test_acquire_different_policies_have_separate_entries(
    fake_mcp: _FakeState, change: dict[str, Any]
) -> None:
    async with MCPClientManager() as manager:
        server = _server()
        context = MCPAuthorizationContext(key="user")
        first = await acquire_lease(manager, server, context)
        second = await acquire_lease(manager, replace(server, **change), context)
        assert len(fake_mcp.opened) == 2
        assert second.server == replace(server, **change)
        await first.release()
        await second.release()


@pytest.mark.asyncio
async def test_acquire_cancel_one_waiter_preserves_other_waiter(
    fake_mcp: _FakeState,
) -> None:
    fake_mcp.gate = asyncio.Event()
    async with MCPClientManager() as manager:
        server = _server()
        context = MCPAuthorizationContext(key="user")
        first = asyncio.create_task(acquire_lease(manager, server, context))
        second = asyncio.create_task(acquire_lease(manager, server, context))
        while not fake_mcp.opened:
            await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not fake_mcp.opened[0].ready.cancelled()
        fake_mcp.gate.set()
        lease = await asyncio.wait_for(second, 1)
        assert len(fake_mcp.opened) == 1
        await lease.release()


@pytest.mark.asyncio
async def test_acquire_cancel_only_waiter_stops_owner(fake_mcp: _FakeState) -> None:
    fake_mcp.gate = asyncio.Event()
    async with MCPClientManager() as manager:
        acquire = asyncio.create_task(
            acquire_lease(manager, _server(), MCPAuthorizationContext(key="u"))
        )
        while not fake_mcp.opened:
            await asyncio.sleep(0)
        acquire.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(acquire, 1)
        assert fake_mcp.opened[0].owner_task.done()


@pytest.mark.asyncio
async def test_catalog_empty_result_is_cached(fake_mcp: _FakeState) -> None:
    fake_mcp.tools = []
    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="u")
        )
        assert await lease.tools() == ()
        assert await lease.tools() == ()
        assert fake_mcp.list_calls == 1
        await lease.release()


@pytest.mark.asyncio
async def test_catalog_invalidation_refreshes_same_lease(fake_mcp: _FakeState) -> None:
    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="u")
        )
        assert [tool.name for tool in await lease.tools()] == ["notes__read"]
        fake_mcp.tools = [types.Tool(name="updated", input_schema={"type": "object"})]
        fake_mcp.opened[0].catalog_revision += 1
        assert [tool.name for tool in await lease.tools()] == ["notes__updated"]
        assert fake_mcp.list_calls == 2
        await lease.release()


@pytest.mark.asyncio
async def test_catalog_notification_during_refresh_remains_invalidated(
    fake_mcp: _FakeState, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="u")
        )
        connection = fake_mcp.opened[0]
        original = fake_mcp.list_tools

        async def changing_list(**kwargs: object) -> types.ListToolsResult:
            result = await original(**kwargs)
            connection.catalog_revision += 1
            return result

        monkeypatch.setattr(connection.client, "list_tools", changing_list)
        await lease.tools()
        await lease.tools()
        assert fake_mcp.list_calls == 2
        await lease.release()


@pytest.mark.asyncio
async def test_catalog_transient_fallback_belongs_only_to_existing_lease(
    fake_mcp: _FakeState,
) -> None:
    async with MCPClientManager() as manager:
        server, context = _server(), MCPAuthorizationContext(key="u")
        first = await acquire_lease(manager, server, context)
        await first.tools()
        fake_mcp.opened[0].catalog_revision += 1
        fake_mcp.list_error = OSError("disconnected")
        assert [tool.name for tool in await first.tools()] == ["notes__read"]
        second = await acquire_lease(manager, server, context)
        with pytest.raises(MCPDiscoveryError):
            await second.tools()
        await first.release()
        await second.release()


@pytest.mark.asyncio
async def test_catalog_authorization_failure_does_not_reuse_stale_tools(
    fake_mcp: _FakeState,
) -> None:
    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="u")
        )
        await lease.tools()
        fake_mcp.opened[0].catalog_revision += 1
        fake_mcp.list_error = MCPAuthorizationError("revoked")
        with pytest.raises(MCPAuthorizationError):
            await lease.tools()
        await lease.release()


def _hanging_server(pid_file: Path, timeout: float) -> MCPServer:
    return MCPServer(
        name="hanging",
        transport=MCPStdioTransport(
            command=sys.executable,
            args=(str(Path(__file__).parent / "fixtures/hanging_server.py"),),
            env={"MCP_TEST_PID": str(pid_file)},
        ),
        connect_timeout_seconds=timeout,
    )


def _assert_process_stopped(pid_file: Path) -> None:
    if pid_file.exists():
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)


@pytest.mark.asyncio
async def test_stdio_handshake_timeout_stops_process(tmp_path: Path) -> None:
    pid_file = tmp_path / "server.pid"
    async with MCPClientManager() as manager:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                acquire_lease(
                    manager,
                    _hanging_server(pid_file, 0.15),
                    MCPAuthorizationContext(key="u"),
                ),
                3,
            )
    _assert_process_stopped(pid_file)


@pytest.mark.asyncio
async def test_shutdown_pending_stdio_handshake_joins_process(tmp_path: Path) -> None:
    pid_file = tmp_path / "server.pid"
    manager = MCPClientManager()
    acquire = asyncio.create_task(
        acquire_lease(
            manager, _hanging_server(pid_file, 20), MCPAuthorizationContext(key="u")
        )
    )
    try:
        for _ in range(100):
            if pid_file.exists():
                break
            await asyncio.sleep(0.01)
        assert pid_file.exists()
        await asyncio.wait_for(asyncio.gather(manager.aclose(), manager.aclose()), 3)
        with pytest.raises(MCPError):
            await acquire
        _assert_process_stopped(pid_file)
    finally:
        await manager.aclose()
        await asyncio.gather(acquire, return_exceptions=True)


@pytest.mark.asyncio
async def test_stdio_crash_reconnects_on_discovery_without_replaying_tool(
    tmp_path: Path,
) -> None:
    call_count = tmp_path / "calls"
    server = MCPServer(
        name="crash",
        transport=MCPStdioTransport(
            command=sys.executable,
            args=(str(Path(__file__).parent / "fixtures/crash_server.py"),),
            env={"MCP_CALL_COUNT": str(call_count)},
        ),
    )
    async with MCPClientManager() as manager:
        lease = await acquire_lease(manager, server, MCPAuthorizationContext(key="u"))
        tools = {tool.name: tool for tool in await lease.tools()}
        crash = tools["crash__crash"]
        result = await crash.run(crash.args_type()(), CancellationToken())
        assert result.error.kind == "mcp_transport"
        updated = {tool.name: tool for tool in await lease.tools()}
        ping = updated["crash__ping"]
        assert "pong" in await ping.run(ping.args_type()(), CancellationToken())
        assert call_count.read_text() == "1"
        await lease.release()


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.asyncio
async def test_tool_timeout_and_cancellation_stop_one_call(
    fake_mcp: _FakeState, cancel: bool
) -> None:
    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager,
            _server(tool_timeout_seconds=0.03),
            MCPAuthorizationContext(key="u"),
        )
        entered, finished = asyncio.Event(), asyncio.Event()
        calls = 0

        async def wait(*args: object, **kwargs: object) -> None:
            nonlocal calls
            del args, kwargs
            calls += 1
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()

        fake_mcp.opened[0].client.call_tool = wait
        tool = (await lease.tools())[0]
        token = CancellationToken()
        call = asyncio.create_task(tool.run(tool.args_type()(), token))
        await entered.wait()
        if cancel:
            token.cancel()
        result = await asyncio.wait_for(call, 1)
        assert result.error.kind == ("cancelled" if cancel else "timeout")
        assert finished.is_set()
        assert calls == 1
        await lease.release()


def test_stdio_environment_is_copied_before_pool_identity() -> None:
    environment = {"APP": "original"}
    config = MCPStdioTransport(command="fixture", env=environment)
    environment["APP"] = "mutated"
    assert config.env == {"APP": "original"}


def _waiting_server(tmp_path: Path) -> MCPServer:
    return MCPServer(
        name="waiting",
        transport=MCPStdioTransport(
            command=sys.executable,
            args=(str(Path(__file__).parent / "fixtures/waiting_server.py"),),
            env={
                "MCP_TEST_PID": str(tmp_path / "server.pid"),
                "MCP_WAIT_STARTED": str(tmp_path / "started"),
            },
        ),
    )


async def _wait_for_started(tmp_path: Path) -> None:
    async with asyncio.timeout(3):
        while not (tmp_path / "started").exists():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_stdio_release_one_run_preserves_concurrent_calls(tmp_path: Path) -> None:
    async with MCPClientManager() as manager:
        server, context = _waiting_server(tmp_path), MCPAuthorizationContext(key="u")
        first = await acquire_lease(manager, server, context)
        second = await acquire_lease(manager, server, context)
        tools = {tool.name: tool for tool in await second.tools()}
        wait = tools["waiting__wait"]
        token = CancellationToken()
        pending = asyncio.create_task(wait.run(wait.args_type()(), token))
        try:
            await _wait_for_started(tmp_path)
            await first.release()
            assert not pending.done()
            ping = tools["waiting__ping"]
            assert "pong" in await ping.run(ping.args_type()(), CancellationToken())
            token.cancel()
            result = await asyncio.wait_for(pending, 2)
            assert result.error.kind == "cancelled"
            await second.release()
        finally:
            token.cancel()
            await asyncio.gather(pending, return_exceptions=True)
    _assert_process_stopped(tmp_path / "server.pid")


@pytest.mark.asyncio
async def test_stdio_shutdown_cancels_active_call_and_joins_process(
    tmp_path: Path,
) -> None:
    manager = MCPClientManager()
    lease = await acquire_lease(
        manager, _waiting_server(tmp_path), MCPAuthorizationContext(key="u")
    )
    tools = {tool.name: tool for tool in await lease.tools()}
    wait = tools["waiting__wait"]
    pending = asyncio.create_task(wait.run(wait.args_type()(), CancellationToken()))
    try:
        await _wait_for_started(tmp_path)
        await asyncio.wait_for(manager.aclose(), 3)
        result = await asyncio.wait_for(pending, 1)
        assert result.error.kind == "mcp_transport"
        _assert_process_stopped(tmp_path / "server.pid")
    finally:
        await manager.aclose()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_catalog_token_redaction_preserves_schema_property_names(
    fake_mcp: _FakeState,
) -> None:
    canary = "catalog-canary-canary"
    schema = {
        "type": "object",
        "properties": {"access_token": {"type": "string", "default": canary}},
    }
    fake_mcp.tools = [
        types.Tool(name="read", description=f"Echo {canary}", input_schema=schema)
    ]
    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="u")
        )
        fake_mcp.opened[0].secrets.add(canary)
        tool = (await lease.tools())[0]
        assert canary not in str(tool.schema)
        assert (
            tool.schema["parameters"]["properties"]["access_token"]["type"] == "string"
        )
        assert fake_mcp.tools[0].input_schema == schema
        await lease.release()


@pytest.mark.parametrize("in_name", [False, True])
@pytest.mark.asyncio
async def test_catalog_credential_in_tool_or_schema_name_is_rejected(
    fake_mcp: _FakeState, in_name: bool
) -> None:
    canary = "catalog-canary-canary"
    fake_mcp.tools = [
        types.Tool(
            name=canary if in_name else "read",
            input_schema={
                "type": "object",
                "properties": {"safe" if in_name else canary: {"type": "string"}},
            },
        )
    ]
    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="u")
        )
        fake_mcp.opened[0].secrets.add(canary)
        with pytest.raises(MCPDiscoveryError, match="credential") as error:
            await lease.tools()
        assert canary not in str(error.value)
        await lease.release()


@pytest.mark.asyncio
async def test_stdio_server_stderr_is_not_forwarded_to_application(
    capfd: pytest.CaptureFixture[str],
) -> None:
    canary = "STDIO_STDERR_CANARY_7"
    server = MCPServer(
        name="stderr",
        transport=MCPStdioTransport(
            command=sys.executable,
            args=(str(Path(__file__).parent / "fixtures/crash_server.py"),),
            env={"MCP_STDERR_CANARY": canary},
        ),
    )
    async with MCPClientManager() as manager:
        lease = await acquire_lease(manager, server, MCPAuthorizationContext(key="u"))
        assert await lease.tools()
        await lease.release()
    captured = capfd.readouterr()
    assert canary not in captured.out + captured.err
