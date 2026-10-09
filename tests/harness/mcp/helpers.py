"""Shared lifecycle support for MCP tests."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import pytest
import uvicorn
from starlette.types import ASGIApp

from agentlane.harness.mcp import (
    MCPAuthorizationContext,
    MCPClientManager,
    MCPError,
    MCPServer,
)
from agentlane.harness.mcp import (
    _client as mcp_client,  # pyright: ignore[reportPrivateUsage]
)
from agentlane.harness.mcp._client import MCPClientLease
from agentlane.harness.mcp._connection import MCPConnection


async def acquire_lease(
    manager: MCPClientManager,
    server: MCPServer,
    context: MCPAuthorizationContext,
) -> MCPClientLease:
    """Access the internal lease boundary in connection and transport tests."""
    return await manager._acquire(  # pyright: ignore[reportPrivateUsage]
        server, context
    )


class ConnectionInstaller:
    """Replace SDK startup while keeping the connection owner's lifecycle."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch

    def __call__(self, prepare: Callable[[MCPConnection], Awaitable[None]]) -> None:
        async def run_connection(connection: MCPConnection) -> None:
            try:
                await prepare(connection)
                connection.ready.set_result(None)
                await connection.stop_event.wait()
            except BaseException as exc:
                connection.failed = True
                if not connection.ready.done():
                    error = (
                        MCPError("Cancelled")
                        if isinstance(exc, asyncio.CancelledError)
                        else exc
                    )
                    connection.ready.set_exception(error)
                    connection.ready.exception()
            finally:
                connection.closing = True

        self._monkeypatch.setattr(mcp_client, "_run_connection", run_connection)


@asynccontextmanager
async def http_server(app: ASGIApp, port: int) -> AsyncIterator[None]:
    """Run a local ASGI peer with bounded startup and shutdown."""
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="error", lifespan="on"
        )
    )
    task = asyncio.create_task(server.serve())
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if task.done():
                    await task
                    raise AssertionError("The local MCP HTTP peer stopped at startup.")
                await asyncio.sleep(0.01)
        yield
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
