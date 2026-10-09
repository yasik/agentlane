"""Harness shim that exposes discovered MCP tools to model turns."""

import asyncio
import time
from dataclasses import dataclass
from functools import partial

import structlog
from pydantic import BaseModel

from agentlane.models import Tool
from agentlane.models.run import RunContext

from .._run import RunResult, RunState
from ..shims import (
    BoundShim,
    PreparedTurn,
    Shim,
    ShimBindingContext,
    ToolSourceBinding,
)
from ._client import MCPClientLease, MCPClientManager
from ._errors import MCPDiscoveryError
from ._types import MCPAuthorizationContext, MCPServer

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class _SourceState:
    """Keep retry and ownership state even before the first successful lease."""

    server: MCPServer
    lease: MCPClientLease | None = None
    tools: tuple[Tool[BaseModel, object], ...] = ()
    failures: int = 0
    retry_at: float | None = 0.0


@dataclass(slots=True)
class _BoundMCPToolsShim(BoundShim):
    source: object
    servers: tuple[MCPServer, ...]
    authorization_context: MCPAuthorizationContext
    manager: MCPClientManager
    owns_manager: bool
    max_concurrent_discoveries: int = 8
    inherited_names: frozenset[str] | None = None
    sources: tuple[_SourceState, ...] = ()
    tools: tuple[Tool[BaseModel, object], ...] = ()

    async def on_run_start(
        self,
        state: RunState,
        transient_state: RunContext[object],
    ) -> None:
        del state, transient_state
        self.sources = tuple(_SourceState(server) for server in self.servers)
        self.tools = ()
        if self.owns_manager and self.manager.closed:
            self.manager = MCPClientManager()
        try:
            self.tools = await self._refresh_tools()
        except BaseException as exc:
            try:
                await self._release_resources()
            except BaseException:
                exc.add_note("MCP resource cleanup also failed.")
            raise

    async def _refresh_tools(self) -> tuple[Tool[BaseModel, object], ...]:
        """Refresh sources with bounded concurrency and publish one snapshot."""
        semaphore = asyncio.Semaphore(self.max_concurrent_discoveries)
        refreshes = tuple(
            asyncio.create_task(self._refresh_source(source, semaphore))
            for source in self.sources
        )
        try:
            await asyncio.gather(*refreshes)
        except BaseException:
            # A required failure must not wait for all queued server batches.
            # Drain canceled work before run cleanup releases acquired leases.
            for refresh in refreshes:
                refresh.cancel()
            await asyncio.gather(*refreshes, return_exceptions=True)
            raise
        native_tools: list[Tool[BaseModel, object]] = []
        visible_names: set[str] = set()
        for source in self.sources:
            for tool in source.tools:
                if (
                    self.inherited_names is not None
                    and tool.name not in self.inherited_names
                ):
                    continue
                if tool.name in visible_names:
                    raise MCPDiscoveryError(
                        f"Multiple MCP servers expose the visible tool name {tool.name!r}."
                    )
                visible_names.add(tool.name)
                native_tools.append(tool)
        return tuple(native_tools)

    async def _refresh_source(
        self, source: _SourceState, semaphore: asyncio.Semaphore
    ) -> None:
        if source.retry_at is None or time.monotonic() < source.retry_at:
            return
        async with semaphore:
            try:
                if source.lease is None:
                    source.lease = await self.manager._acquire(  # pyright: ignore[reportPrivateUsage]
                        source.server, self.authorization_context
                    )
                source.tools = await source.lease.tools()
            except Exception as exc:
                source.tools = ()
                if source.server.required:
                    raise
                retryable = (
                    exc.retryable
                    if isinstance(exc, MCPDiscoveryError)
                    else isinstance(exc, (TimeoutError, ConnectionError))
                )
                source.failures += 1
                source.retry_at = (
                    time.monotonic() + min(2 ** min(source.failures - 1, 5), 30)
                    if retryable
                    else None
                )
                logger.warning(
                    "mcp_optional_source_unavailable",
                    server=source.server.name,
                    retryable=retryable,
                )
            else:
                source.failures = 0
                source.retry_at = 0.0

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        self.tools = await self._refresh_tools()
        turn.add_tools(self.tools, require_unique_names=True)

    def inherit_tools(
        self, names: frozenset[str]
    ) -> tuple[ToolSourceBinding, ...] | None:
        allowed_names = names.intersection(tool.name for tool in self.tools)
        if not allowed_names:
            return ()
        bind = partial(
            _bind_mcp_tools,
            source=self.source,
            servers=tuple(
                source.server
                for source in self.sources
                if any(tool.name in allowed_names for tool in source.tools)
            ),
            authorization_context=self.authorization_context,
            client_manager=None if self.owns_manager else self.manager,
            inherited_names=allowed_names,
            max_concurrent_discoveries=self.max_concurrent_discoveries,
        )
        return (ToolSourceBinding(source=self.source, bind=bind),)

    async def on_run_end(
        self,
        result: RunResult | None,
        transient_state: RunContext[object],
    ) -> None:
        del result, transient_state
        await self._release_resources()

    async def _release_resources(self) -> None:
        sources, self.sources = self.sources, ()
        self.tools = ()
        errors: list[BaseException] = []
        if self.owns_manager:
            try:
                await self.manager.aclose()
            except BaseException as exc:
                errors.append(exc)
        else:
            try:
                # Start every release even if cancellation reaches the batch
                # before its lease coroutines get their first event-loop turn.
                results = await asyncio.shield(
                    asyncio.gather(
                        *(
                            source.lease.release()
                            for source in sources
                            if source.lease is not None
                        ),
                        return_exceptions=True,
                    )
                )
            except BaseException as exc:
                errors.append(exc)
            else:
                errors.extend(
                    result for result in results if isinstance(result, BaseException)
                )
        if errors:
            raise BaseExceptionGroup("MCP resource cleanup failed.", errors)


class MCPToolsShim(Shim):
    """Discover MCP tools and add them to every prepared model turn."""

    def __init__(
        self,
        *,
        servers: tuple[MCPServer, ...],
        authorization_context: MCPAuthorizationContext | None = None,
        client_manager: MCPClientManager | None = None,
        max_concurrent_discoveries: int = 8,
    ) -> None:
        if not servers:
            raise ValueError("MCPToolsShim requires at least one server.")
        if (
            isinstance(max_concurrent_discoveries, bool)
            or max_concurrent_discoveries < 1
        ):
            raise ValueError("max_concurrent_discoveries must be a positive integer.")
        self._servers = servers
        self._authorization_context = authorization_context or MCPAuthorizationContext(
            key="anonymous"
        )
        self._client_manager = client_manager
        self._max_concurrent_discoveries = max_concurrent_discoveries

    @property
    def name(self) -> str:
        return "mcp-tools"

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        for binding in context.tool_source_bindings:
            if binding.source is self:
                return await binding.bind(context)
        if context.tool_source_bindings:
            return BoundShim()
        return await _bind_mcp_tools(
            context,
            source=self,
            servers=self._servers,
            authorization_context=self._authorization_context,
            client_manager=self._client_manager,
            max_concurrent_discoveries=self._max_concurrent_discoveries,
        )


async def _bind_mcp_tools(
    context: ShimBindingContext,
    *,
    source: object,
    servers: tuple[MCPServer, ...],
    authorization_context: MCPAuthorizationContext,
    client_manager: MCPClientManager | None,
    max_concurrent_discoveries: int,
    inherited_names: frozenset[str] | None = None,
) -> BoundShim:
    """Create a fresh source session with its complete inheritance limits."""
    del context
    return _BoundMCPToolsShim(
        source=source,
        servers=servers,
        authorization_context=authorization_context,
        manager=client_manager if client_manager is not None else MCPClientManager(),
        owns_manager=client_manager is None,
        max_concurrent_discoveries=max_concurrent_discoveries,
        inherited_names=inherited_names,
    )
