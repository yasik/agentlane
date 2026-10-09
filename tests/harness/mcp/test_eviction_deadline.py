"""Pool eviction bounds the caller while SDK cleanup retains its capacity."""

import asyncio
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from agentlane.harness import RunState, Task
from agentlane.harness.mcp import (
    MCPAuthorizationContext,
    MCPClientLimits,
    MCPClientManager,
    MCPPoolCapacityError,
    MCPServer,
    MCPShutdownTimeoutError,
    MCPStdioTransport,
    MCPToolsShim,
)
from agentlane.harness.mcp import (
    _client as mcp_client,  # pyright: ignore[reportPrivateUsage]
)
from agentlane.harness.mcp._client import (
    _PoolEntry,  # pyright: ignore[reportPrivateUsage]
)
from agentlane.harness.mcp._client import (  # pyright: ignore[reportPrivateUsage]
    MCPClientLease,
)
from agentlane.harness.mcp._connection import MCPConnection
from agentlane.harness.shims import ShimBindingContext
from agentlane.models.run import RunContext
from agentlane.runtime import SingleThreadedRuntimeEngine

from .helpers import acquire_lease


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="The fixture uses POSIX process groups.")
async def test_optional_startup_stops_waiting_for_idle_stdio_eviction(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "peer.pid"
    server = MCPServer(
        name="stubborn",
        transport=MCPStdioTransport(
            command=sys.executable,
            args=(str(Path(__file__).parent / "fixtures/stubborn_stdio_server.py"),),
            env={"MCP_TEST_PID": str(pid_path)},
        ),
        required=False,
        connect_timeout_seconds=2,
        discovery_timeout_seconds=0.1,
    )
    manager = MCPClientManager(
        MCPClientLimits(max_connections=1, shutdown_timeout_seconds=0.03)
    )
    binding = ShimBindingContext(task=Task(engine=SingleThreadedRuntimeEngine()))
    state = RunState(instructions=None, history=[], responses=[])
    context = RunContext[object](context=None)
    try:
        first = await MCPToolsShim(
            servers=(server,),
            client_manager=manager,
            authorization_context=MCPAuthorizationContext(key="first"),
        ).bind(binding)
        await first.on_run_start(state, context)
        await first.on_run_end(None, context)
        second = await MCPToolsShim(
            servers=(server,),
            client_manager=manager,
            authorization_context=MCPAuthorizationContext(key="second"),
        ).bind(binding)
        started = time.monotonic()
        await asyncio.wait_for(second.on_run_start(state, context), timeout=0.5)
        assert time.monotonic() - started < 0.4
        await second.on_run_end(None, context)
        os.kill(int(pid_path.read_text()), 0)

        required = MCPServer(
            name="other", transport=MCPStdioTransport(command="must-not-start")
        )
        third = await MCPToolsShim(servers=(required,), client_manager=manager).bind(
            binding
        )
        with pytest.raises(MCPPoolCapacityError):
            await third.on_run_start(state, context)
    finally:
        try:
            async with asyncio.timeout(8):
                while True:
                    try:
                        await manager.aclose()
                    except MCPShutdownTimeoutError:
                        await asyncio.sleep(0.03)
                    else:
                        break
            with pytest.raises(ProcessLookupError):
                os.kill(int(pid_path.read_text()), 0)
        finally:
            if pid_path.exists():
                try:
                    os.killpg(int(pid_path.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


@dataclass
class _EvictionPeer:
    closing: dict[str, asyncio.Event] = field(default_factory=dict[str, asyncio.Event])
    gates: dict[str, asyncio.Event] = field(default_factory=dict[str, asyncio.Event])
    cancelled: list[str] = field(default_factory=list[str])


@pytest.fixture(name="eviction_peer")
def fixture_eviction_peer(monkeypatch: pytest.MonkeyPatch) -> _EvictionPeer:
    peer = _EvictionPeer()

    async def run_connection(connection: MCPConnection) -> None:
        key = connection.lifecycle_context.key
        peer.closing[key] = asyncio.Event()
        peer.gates[key] = asyncio.Event()
        connection.ready.set_result(None)
        try:
            await connection.stop_event.wait()
        finally:
            peer.closing[key].set()
            try:
                await peer.gates[key].wait()
            except asyncio.CancelledError:
                peer.cancelled.append(key)
                raise

    monkeypatch.setattr(mcp_client, "_run_connection", run_connection)
    return peer


def _server() -> MCPServer:
    return MCPServer(name="peer", transport=MCPStdioTransport(command="fixture"))


@pytest.mark.asyncio
async def test_eviction_uses_one_deadline_when_another_acquirer_takes_capacity(
    eviction_peer: _EvictionPeer, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = MCPClientManager(
        MCPClientLimits(max_connections=2, shutdown_timeout_seconds=0.08)
    )
    competitors: list[asyncio.Task[MCPClientLease]] = []
    deadlines: list[float] = []
    acquiring: asyncio.Task[MCPClientLease] | None = None
    retire = manager._retire_locked  # pyright: ignore[reportPrivateUsage]
    wait = mcp_client._wait_for_retirements  # pyright: ignore[reportPrivateUsage]

    def start_competitor(task: asyncio.Task[None]) -> None:
        del task
        competitors.append(
            asyncio.create_task(
                acquire_lease(manager, _server(), MCPAuthorizationContext(key="D"))
            )
        )

    def retire_with_competitor(entry: _PoolEntry) -> asyncio.Task[None]:
        task = retire(entry)
        if entry.key.authorization_context_key == "A":
            # Register before the acquisition waiter, so D occupies the freed
            # slot before C resumes and selects the next idle victim.
            task.add_done_callback(start_competitor)
        return task

    async def record_wait(
        tasks: tuple[asyncio.Task[None], ...], *, deadline: float
    ) -> bool:
        if asyncio.current_task() is acquiring:
            deadlines.append(deadline)
        return await wait(tasks, deadline=deadline)

    try:
        for key in ("A", "B"):
            lease = await acquire_lease(
                manager, _server(), MCPAuthorizationContext(key=key)
            )
            await lease.release()
        monkeypatch.setattr(manager, "_retire_locked", retire_with_competitor)
        monkeypatch.setattr(mcp_client, "_wait_for_retirements", record_wait)
        acquiring = asyncio.create_task(
            acquire_lease(manager, _server(), MCPAuthorizationContext(key="C"))
        )
        await asyncio.wait_for(eviction_peer.closing["A"].wait(), timeout=1)
        await asyncio.sleep(0.02)
        eviction_peer.gates["A"].set()
        with pytest.raises(MCPPoolCapacityError):
            await asyncio.wait_for(acquiring, timeout=0.5)
        assert len(deadlines) == 2
        assert deadlines[0] == deadlines[1]
        assert eviction_peer.closing["B"].is_set()
        assert not eviction_peer.cancelled
    finally:
        for gate in eviction_peer.gates.values():
            gate.set()
        if acquiring is not None:
            acquiring.cancel()
            await asyncio.gather(acquiring, return_exceptions=True)
        for competitor in competitors:
            lease = await competitor
            await lease.release()
        await manager.aclose()


@pytest.mark.asyncio
async def test_cancelled_eviction_keeps_capacity_until_owned_cleanup_finishes(
    eviction_peer: _EvictionPeer,
) -> None:
    manager = MCPClientManager(
        MCPClientLimits(max_connections=1, shutdown_timeout_seconds=0.1)
    )
    acquiring: asyncio.Task[MCPClientLease] | None = None
    try:
        first = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="A")
        )
        await first.release()
        acquiring = asyncio.create_task(
            acquire_lease(manager, _server(), MCPAuthorizationContext(key="B"))
        )
        await asyncio.wait_for(eviction_peer.closing["A"].wait(), timeout=1)
        acquiring.cancel()
        with pytest.raises(asyncio.CancelledError):
            await acquiring
        with pytest.raises(MCPPoolCapacityError):
            await acquire_lease(manager, _server(), MCPAuthorizationContext(key="B"))
        assert not eviction_peer.cancelled
        eviction_peer.gates["A"].set()
        async with asyncio.timeout(1):
            while manager._retiring:  # pyright: ignore[reportPrivateUsage]
                await asyncio.sleep(0)
        second = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="B")
        )
        await second.release()
    finally:
        for gate in eviction_peer.gates.values():
            gate.set()
        if acquiring is not None:
            acquiring.cancel()
            await asyncio.gather(acquiring, return_exceptions=True)
        await manager.aclose()
    assert not eviction_peer.cancelled
