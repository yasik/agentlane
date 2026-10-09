"""The official SDK must finish process cleanup after a manager deadline."""

import asyncio
import os
import signal
import sys
from pathlib import Path

import pytest

from agentlane.harness.mcp import (
    MCPAuthorizationContext,
    MCPClientLimits,
    MCPClientManager,
    MCPServer,
    MCPShutdownTimeoutError,
    MCPStdioTransport,
)

from .helpers import acquire_lease


@pytest.mark.skipif(os.name != "posix", reason="The fixture uses POSIX process groups.")
@pytest.mark.asyncio
async def test_shutdown_deadline_preserves_sdk_process_kill(tmp_path: Path) -> None:
    pid_file = tmp_path / "server.pid"
    server = MCPServer(
        name="stubborn",
        transport=MCPStdioTransport(
            command=sys.executable,
            args=(str(Path(__file__).parent / "fixtures/stubborn_stdio_server.py"),),
            env={"MCP_TEST_PID": str(pid_file)},
        ),
    )
    manager = MCPClientManager(MCPClientLimits(shutdown_timeout_seconds=0.05))
    try:
        lease = await acquire_lease(
            manager, server, MCPAuthorizationContext(key="test")
        )
        # Stop the caller's wait without interrupting the SDK's bounded
        # graceful-exit and forced-kill stages in the connection owner task.
        with pytest.raises(MCPShutdownTimeoutError):
            await asyncio.wait_for(manager.aclose(), 0.5)
        assert manager.closed
        os.kill(int(pid_file.read_text()), 0)

        await _wait_for_shutdown(manager, timeout=8)
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)

        await lease.release()
    finally:
        # Keep this regression safe even when the behavior under test leaks
        # the fixture. The recorded PID is its SDK-created process-group ID.
        if pid_file.exists():
            try:
                os.killpg(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass

        await _wait_for_shutdown(manager, timeout=3)


async def _wait_for_shutdown(manager: MCPClientManager, *, timeout: float) -> None:
    async with asyncio.timeout(timeout):
        while True:
            try:
                await manager.aclose()
            except MCPShutdownTimeoutError:
                await asyncio.sleep(0.05)
            else:
                return
