"""Owner-task lifecycle for official MCP client transports."""

import asyncio
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import cast

import structlog

from ._auth import MCPAuthorizationState, product_bearer_auth, record_http_failure
from ._errors import (
    MCPAuthorizationError,
    MCPDependencyError,
    MCPDiscoveryError,
    MCPError,
)
from ._operation import MCPOperation, mcp_operation
from ._sdk import (
    MCPClientProtocol,
    MCPShutdownScope,
    exception_kind,
    load_mcp_dependencies,
    shutdown_scope,
)
from ._types import (
    MCPAuthorizationContext,
    MCPCatalog,
    MCPServer,
    MCPStreamableHTTPTransport,
)
from ._validation import validate_redirect

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class MCPConnection:
    server: MCPServer
    lifecycle_context: MCPAuthorizationContext
    """Full context of the lease that opened this transport generation."""
    ready: asyncio.Future[None]
    authorization: MCPAuthorizationState = field(default_factory=MCPAuthorizationState)
    client: MCPClientProtocol | None = None
    protocol_version: str | None = None
    secrets: set[str] = field(default_factory=set[str], repr=False)
    owner_task: asyncio.Task[None] | None = None
    close_task: asyncio.Task[None] | None = None
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    catalog: MCPCatalog | None = None
    catalog_revision: int = 0
    catalog_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_calls: set[asyncio.Task[object]] = field(
        default_factory=set[asyncio.Task[object]]
    )
    shutdown_timeout_seconds: float = 10.0
    failed: bool = False
    closing: bool = False

    def get_client(self) -> MCPClientProtocol:
        if self.client is None:
            raise MCPError("MCP connection is not ready.")
        return self.client


async def run_connection(connection: MCPConnection) -> None:
    """Keep the cancellation scope outside every task-local SDK context."""
    try:
        load_mcp_dependencies()
        with shutdown_scope() as scope:
            await _own_connection(connection, scope)
    except BaseException as exc:
        connection.failed = True
        if not connection.ready.done():
            connection.ready.set_exception(
                exc
                if isinstance(exc, MCPDependencyError)
                else MCPError("MCP connection startup failed.")
            )
            connection.ready.exception()
        raise


async def _own_connection(
    connection: MCPConnection, shutdown: MCPShutdownScope
) -> None:
    """Enter and exit every SDK context in one task, even during cancellation."""
    listener: asyncio.Task[None] | None = None
    operation: MCPOperation | None = None
    stack = AsyncExitStack()
    try:
        with mcp_operation(
            authorization_context=connection.lifecycle_context
        ) as operation:
            operation.secrets = connection.secrets
            async with asyncio.timeout(connection.server.connect_timeout_seconds):
                sdk = load_mcp_dependencies()
                server = connection.server

                async def message_handler(message: object) -> None:
                    if (
                        getattr(message, "method", None)
                        == "notifications/tools/list_changed"
                    ):
                        connection.catalog_revision += 1
                    elif (
                        isinstance(message, Exception)
                        and exception_kind(message) == "transport"
                    ):
                        connection.failed = True

                if isinstance(server.transport, MCPStreamableHTTPTransport):
                    config = server.transport
                    http_client = await stack.enter_async_context(
                        sdk.http.AsyncClient(
                            auth=(
                                product_bearer_auth(
                                    sdk.http,
                                    server,
                                    connection.lifecycle_context,
                                    connection.authorization,
                                )
                                if server.authorization is not None
                                else None
                            ),
                            timeout=sdk.http.Timeout(
                                connect=config.connect_timeout_seconds,
                                read=config.read_timeout_seconds,
                                write=config.connect_timeout_seconds,
                                pool=config.connect_timeout_seconds,
                            ),
                            trust_env=False,
                            follow_redirects=True,
                            event_hooks={
                                "response": [validate_redirect, record_http_failure]
                            },
                        )
                    )
                    transport = sdk.streamable_http.streamable_http_client(
                        config.url, http_client=http_client
                    )
                else:
                    transport = sdk.stdio.stdio_client(
                        sdk.stdio.StdioServerParameters(
                            command=server.transport.command,
                            args=list(server.transport.args),
                            env=(
                                dict(server.transport.env)
                                if server.transport.env is not None
                                else None
                            ),
                            cwd=server.transport.cwd,
                        ),
                        errlog=stack.enter_context(open(os.devnull, "w")),
                    )
                client = sdk.client(
                    transport, cache=None, message_handler=message_handler
                )
                sdk_client = await stack.enter_async_context(client)
                connection.client = cast(MCPClientProtocol, sdk_client)
                connection.protocol_version = cast(
                    str | None, sdk_client.session.protocol_version
                )
            listener = asyncio.create_task(
                _listen_for_tool_changes(connection),
                name=f"agentlane-mcp-listen-{server.name}",
            )
            connection.ready.set_result(None)
            await connection.stop_event.wait()
    except BaseException as exc:
        connection.failed = True
        if not connection.ready.done():
            if isinstance(exc, asyncio.CancelledError):
                connection.ready.set_exception(
                    MCPError("MCP connection startup was cancelled.")
                )
            elif isinstance(exc, MCPDependencyError):
                connection.ready.set_exception(exc)
            else:
                kind = (
                    operation.failure_kind if operation is not None else None
                ) or exception_kind(exc)
                error: Exception
                if kind == "authorization":
                    error = MCPAuthorizationError(
                        "MCP connection authorization failed."
                    )
                elif kind == "timeout":
                    error = TimeoutError("MCP connection timed out.")
                elif kind == "transport":
                    error = ConnectionError("MCP connection transport failed.")
                else:
                    error = MCPDiscoveryError("MCP connection protocol failed.")
                connection.ready.set_exception(error)
            # An abandoned acquisition may have no waiter left to retrieve this.
            connection.ready.exception()
    finally:
        connection.closing = True
        shutdown.deadline = (
            asyncio.get_running_loop().time() + connection.shutdown_timeout_seconds
        )
        try:
            if listener is not None:
                listener.cancel()
                await asyncio.gather(listener, return_exceptions=True)
            calls = tuple(connection.active_calls)
            for call in calls:
                call.cancel()
            await asyncio.gather(*calls, return_exceptions=True)
        finally:
            # A native asyncio timeout defeats the SDK's bounded stdio process
            # cleanup shield. The outer AnyIO scope cancels HTTP auth on DELETE
            # but lets the SDK finish bounded process-tree termination first.
            with mcp_operation(authorization_context=connection.lifecycle_context):
                await stack.aclose()


async def close_connection(connection: MCPConnection) -> None:
    if connection.close_task is None:
        connection.closing = True
        connection.close_task = asyncio.create_task(_finish_close(connection))
    await asyncio.shield(connection.close_task)


async def _finish_close(connection: MCPConnection) -> None:
    connection.closing = True
    connection.stop_event.set()
    owner = connection.owner_task
    if owner is not None:
        if not connection.ready.done():
            owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)
    if not connection.ready.done():
        connection.ready.set_exception(
            MCPError("MCP connection startup was cancelled.")
        )
        connection.ready.exception()


async def _listen_for_tool_changes(connection: MCPConnection) -> None:
    with mcp_operation(authorization_context=connection.lifecycle_context):
        try:
            async with connection.get_client().listen(
                tools_list_changed=True
            ) as subscription:
                # Changes before acknowledgment may have no notification.
                connection.catalog_revision += 1
                async for _event in subscription:
                    connection.catalog_revision += 1
        except Exception as exc:
            if exception_kind(exc) == "transport":
                connection.failed = True
            elif type(exc).__name__ != "ListenNotSupportedError":
                logger.debug(
                    "mcp_subscription_ended",
                    server=connection.server.name,
                    status="unavailable",
                )
        else:
            # Normal completion also requires a new notification stream.
            connection.failed = True
