"""Caller deadlines must not cancel connection-owner cleanup."""

import asyncio
import gc
import weakref
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import pytest
from mcp import types

from agentlane.harness import RunState, Task
from agentlane.harness.mcp import (
    MCPAuthorizationContext,
    MCPClientLimits,
    MCPClientManager,
    MCPServer,
    MCPShutdownTimeoutError,
    MCPStdioTransport,
    MCPToolsShim,
)
from agentlane.harness.mcp import (
    _client as mcp_client,  # pyright: ignore[reportPrivateUsage]
)
from agentlane.harness.mcp._connection import MCPConnection
from agentlane.harness.mcp._shim import (
    _BoundMCPToolsShim,  # pyright: ignore[reportPrivateUsage]
)
from agentlane.harness.shims import ShimBindingContext
from agentlane.models.run import RunContext
from agentlane.runtime import SingleThreadedRuntimeEngine

from .helpers import acquire_lease


class _ShutdownClient:
    async def list_tools(
        self, *, cursor: str | None, cache_mode: Literal["bypass"]
    ) -> types.ListToolsResult:
        del cursor, cache_mode
        return types.ListToolsResult(
            tools=[types.Tool(name="read", input_schema={"type": "object"})]
        )

    async def call_tool(
        self, name: str, arguments: dict[str, Any], *, read_timeout_seconds: float
    ) -> object:
        raise AssertionError("Shutdown tests must not execute tools.")

    def listen(
        self, *, tools_list_changed: bool
    ) -> AbstractAsyncContextManager[AsyncIterator[object]]:
        raise AssertionError("The controlled owner does not start a listener.")


@dataclass
class _ShutdownPeer:
    connections: dict[str, MCPConnection] = field(
        default_factory=dict[str, MCPConnection]
    )
    closing: dict[str, asyncio.Event] = field(default_factory=dict[str, asyncio.Event])
    opened: asyncio.Event = field(default_factory=asyncio.Event)
    cleanup_gate: asyncio.Event = field(default_factory=asyncio.Event)
    cleanup_cancelled: bool = False
    startup: Literal["ready", "failed", "pending"] = "ready"
    startup_error: ConnectionError = field(
        default_factory=lambda: ConnectionError("fixture startup failed")
    )


@pytest.fixture(name="shutdown_peer")
def fixture_shutdown_peer(monkeypatch: pytest.MonkeyPatch) -> _ShutdownPeer:
    peer = _ShutdownPeer()

    async def run_connection(connection: MCPConnection) -> None:
        name = connection.server.name
        peer.connections[name] = connection
        peer.closing[name] = asyncio.Event()
        connection.client = _ShutdownClient()
        if peer.startup == "ready":
            connection.ready.set_result(None)
        elif peer.startup == "failed":
            connection.failed = True
            connection.ready.set_exception(peer.startup_error)
            connection.ready.exception()
        peer.opened.set()
        try:
            await connection.stop_event.wait()
        finally:
            peer.closing[name].set()
            try:
                if name == "slow":
                    await peer.cleanup_gate.wait()
            except asyncio.CancelledError:
                peer.cleanup_cancelled = True
                raise
            finally:
                connection.closing = True

    monkeypatch.setattr(mcp_client, "_run_connection", run_connection)
    return peer


def _server(name: str = "slow") -> MCPServer:
    return MCPServer(name=name, transport=MCPStdioTransport(command="fixture"))


@pytest.mark.asyncio
@pytest.mark.parametrize("owns_manager", [False, True])
async def test_run_cleanup_bounds_failed_owners_and_starts_other_cleanup(
    shutdown_peer: _ShutdownPeer, owns_manager: bool
) -> None:
    manager = MCPClientManager(MCPClientLimits(shutdown_timeout_seconds=0.02))
    bound = cast(
        _BoundMCPToolsShim,
        await MCPToolsShim(
            servers=(_server(), _server("fast")), client_manager=manager
        ).bind(ShimBindingContext(task=Task(engine=SingleThreadedRuntimeEngine()))),
    )
    bound.owns_manager = owns_manager
    context = RunContext[object](context=None)
    cleanup: asyncio.Task[None] | None = None
    try:
        await bound.on_run_start(
            RunState(instructions=None, history=[], responses=[]), context
        )
        for connection in shutdown_peer.connections.values():
            connection.failed = True

        cleanup = asyncio.create_task(bound.on_run_end(None, context))
        with pytest.raises(BaseExceptionGroup) as caught:
            await asyncio.wait_for(cleanup, timeout=0.5)
        assert all(
            isinstance(error, MCPShutdownTimeoutError)
            for error in caught.value.exceptions
        )
        assert shutdown_peer.closing["slow"].is_set()
        assert shutdown_peer.closing["fast"].is_set()
        assert manager.closed is owns_manager
        slow_owner = shutdown_peer.connections["slow"].owner_task
        fast_owner = shutdown_peer.connections["fast"].owner_task
        assert slow_owner is not None and not slow_owner.done()
        assert fast_owner is not None and fast_owner.done()
        assert not shutdown_peer.cleanup_cancelled
        assert bound.sources == ()
        assert bound.tools == ()
        if not owns_manager:
            await bound.on_run_end(None, context)
    finally:
        shutdown_peer.cleanup_gate.set()
        if cleanup is not None:
            await asyncio.gather(cleanup, return_exceptions=True)
        await asyncio.wait_for(manager.aclose(), timeout=1)
    assert slow_owner.done()
    assert not shutdown_peer.cleanup_cancelled
    await bound.on_run_end(None, context)


@pytest.mark.asyncio
async def test_shared_cleanup_releases_other_leases_before_slow_release_finishes(
    shutdown_peer: _ShutdownPeer, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = MCPClientManager()
    bound = cast(
        _BoundMCPToolsShim,
        await MCPToolsShim(
            servers=(_server(), _server("fast")), client_manager=manager
        ).bind(ShimBindingContext(task=Task(engine=SingleThreadedRuntimeEngine()))),
    )
    context = RunContext[object](context=None)
    release_started = asyncio.Event()
    release_gate = asyncio.Event()
    cleanup: asyncio.Task[None] | None = None
    try:
        await bound.on_run_start(
            RunState(instructions=None, history=[], responses=[]), context
        )
        slow, fast = (source.lease for source in bound.sources)
        assert slow is not None and fast is not None
        release_slow = slow.release

        async def gated_release() -> None:
            release_started.set()
            await release_gate.wait()
            await release_slow()

        monkeypatch.setattr(slow, "release", gated_release)
        cleanup = asyncio.create_task(bound.on_run_end(None, context))
        await asyncio.wait_for(release_started.wait(), timeout=1)
        async with asyncio.timeout(1):
            while fast._entry.leases:  # pyright: ignore[reportPrivateUsage]
                await asyncio.sleep(0)
        assert not cleanup.done()
        assert not manager.closed
    finally:
        release_gate.set()
        shutdown_peer.cleanup_gate.set()
        if cleanup is not None:
            await asyncio.wait_for(cleanup, timeout=1)
        await asyncio.wait_for(manager.aclose(), timeout=1)


@pytest.mark.asyncio
async def test_failed_lease_release_timeout_is_idempotent_and_preserves_cleanup(
    shutdown_peer: _ShutdownPeer,
) -> None:
    manager = MCPClientManager(MCPClientLimits(shutdown_timeout_seconds=0.02))
    try:
        lease = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="u")
        )
        entry = lease._entry  # pyright: ignore[reportPrivateUsage]
        shutdown_peer.connections["slow"].failed = True
        for _ in range(2):
            with pytest.raises(MCPShutdownTimeoutError):
                await asyncio.wait_for(lease.release(), timeout=0.5)
            assert entry.leases == 0
        owner = shutdown_peer.connections["slow"].owner_task
        assert owner is not None and not owner.done()
        assert not shutdown_peer.cleanup_cancelled
    finally:
        shutdown_peer.cleanup_gate.set()
        await asyncio.wait_for(manager.aclose(), timeout=1)
    assert owner.done()
    assert not shutdown_peer.cleanup_cancelled


@pytest.mark.asyncio
async def test_cancelled_run_cleanup_starts_every_shared_lease_release(
    shutdown_peer: _ShutdownPeer,
) -> None:
    manager = MCPClientManager()
    bound = cast(
        _BoundMCPToolsShim,
        await MCPToolsShim(
            servers=(_server(), _server("fast")), client_manager=manager
        ).bind(ShimBindingContext(task=Task(engine=SingleThreadedRuntimeEngine()))),
    )
    context = RunContext[object](context=None)

    async def cancel_at_cleanup_start() -> None:
        cleanup = asyncio.current_task()
        assert cleanup is not None
        asyncio.get_running_loop().call_soon(cleanup.cancel)
        await bound.on_run_end(None, context)

    try:
        await bound.on_run_start(
            RunState(instructions=None, history=[], responses=[]), context
        )
        leases = tuple(source.lease for source in bound.sources)
        with pytest.raises(BaseExceptionGroup) as caught:
            await asyncio.create_task(cancel_at_cleanup_start())
        assert len(caught.value.exceptions) == 1
        assert isinstance(caught.value.exceptions[0], asyncio.CancelledError)
        assert bound.sources == ()
        for lease in leases:
            assert lease is not None
            task = lease._release_task  # pyright: ignore[reportPrivateUsage]
            async with asyncio.timeout(0.5):
                while task is None:
                    await asyncio.sleep(0)
                    task = lease._release_task  # pyright: ignore[reportPrivateUsage]
            await asyncio.wait_for(task, timeout=0.5)
            assert lease._released  # pyright: ignore[reportPrivateUsage]
            assert lease._entry.leases == 0  # pyright: ignore[reportPrivateUsage]
        assert not manager.closed
    finally:
        shutdown_peer.cleanup_gate.set()
        await asyncio.wait_for(manager.aclose(), timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_acquire_preserves_primary_failure_when_cleanup_times_out(
    shutdown_peer: _ShutdownPeer, cancel: bool
) -> None:
    shutdown_peer.startup = "pending" if cancel else "failed"
    manager = MCPClientManager(MCPClientLimits(shutdown_timeout_seconds=0.02))
    acquiring = asyncio.create_task(
        acquire_lease(manager, _server(), MCPAuthorizationContext(key="u"))
    )
    try:
        await asyncio.wait_for(shutdown_peer.opened.wait(), timeout=1)
        if cancel:
            acquiring.cancel()
        with pytest.raises(
            asyncio.CancelledError if cancel else ConnectionError
        ) as caught:
            await asyncio.wait_for(acquiring, timeout=0.5)
        if not cancel:
            assert caught.value is shutdown_peer.startup_error
        assert any("cleanup" in note.lower() for note in caught.value.__notes__)
        assert not shutdown_peer.cleanup_cancelled
    finally:
        shutdown_peer.cleanup_gate.set()
        await asyncio.gather(acquiring, return_exceptions=True)
        await asyncio.wait_for(manager.aclose(), timeout=1)
    assert not shutdown_peer.cleanup_cancelled


@pytest.mark.asyncio
async def test_cancelled_release_observes_its_background_timeout(
    shutdown_peer: _ShutdownPeer,
) -> None:
    manager = MCPClientManager(MCPClientLimits(shutdown_timeout_seconds=0.02))
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled: list[dict[str, Any]] = []

    def record_exception(
        loop: asyncio.AbstractEventLoop, context: dict[str, Any]
    ) -> None:
        del loop
        unhandled.append(context)

    loop.set_exception_handler(record_exception)
    try:
        lease = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="u")
        )
        shutdown_peer.connections["slow"].failed = True
        releasing = asyncio.create_task(lease.release())
        await asyncio.wait_for(shutdown_peer.closing["slow"].wait(), timeout=1)
        releasing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await releasing
        release_task = lease._release_task  # pyright: ignore[reportPrivateUsage]
        assert release_task is not None
        completed, pending = await asyncio.wait((release_task,), timeout=0.5)
        assert not pending
        reference = weakref.ref(release_task)
        del lease, release_task, releasing, completed
        await asyncio.sleep(0)
        gc.collect()
        assert reference() is None
        assert unhandled == []
        assert not shutdown_peer.cleanup_cancelled
    finally:
        shutdown_peer.cleanup_gate.set()
        await asyncio.wait_for(manager.aclose(), timeout=1)
        loop.set_exception_handler(previous_handler)
