"""Typed lazy boundary for the optional official MCP SDK."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from typing import Any, Literal, Protocol, cast

from ._errors import (
    MCPAuthorizationError,
    MCPDependencyError,
    MCPDiscoveryError,
    MCPFailureKind,
)
from ._types import MCPRemoteTool


class MCPClientProtocol(Protocol):
    async def list_tools(
        self, *, cursor: str | None, cache_mode: Literal["bypass"]
    ) -> object: ...

    async def call_tool(
        self, name: str, arguments: dict[str, Any], *, read_timeout_seconds: float
    ) -> object: ...

    def listen(
        self, *, tools_list_changed: bool
    ) -> AbstractAsyncContextManager[AsyncIterator[object]]: ...


class MCPShutdownScope(Protocol):
    deadline: float
    cancel_called: bool


def shutdown_scope() -> AbstractContextManager[MCPShutdownScope]:
    """Create an outer scope without eagerly importing the optional runtime."""
    return cast(
        AbstractContextManager[MCPShutdownScope],
        import_module("anyio").CancelScope(),
    )


@dataclass(frozen=True, slots=True)
class _ToolPage:
    tools: tuple[MCPRemoteTool, ...]
    next_cursor: str | None
    ttl_ms: int | None


class _SDKTool(Protocol):
    name: str
    description: str | None
    input_schema: dict[str, Any]


class _SDKToolPage(Protocol):
    tools: list[_SDKTool]
    next_cursor: str | None
    ttl_ms: int
    model_fields_set: set[str]


def read_tool_page(raw_result: object) -> _ToolPage:
    """Normalize SDK-validated values at the optional dependency boundary."""
    result = cast(_SDKToolPage, raw_result)
    return _ToolPage(
        tools=tuple(
            MCPRemoteTool(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
            )
            for tool in result.tools
        ),
        next_cursor=result.next_cursor,
        ttl_ms=result.ttl_ms if "ttl_ms" in result.model_fields_set else None,
    )


@dataclass(frozen=True, slots=True)
class MCPDependencies:
    """Dynamic imports remain confined to SDK transport construction."""

    client: Any
    stdio: Any
    streamable_http: Any
    subscriptions: Any
    http: Any
    types: Any
    modern_protocol_versions: tuple[str, ...]


@cache
def load_mcp_dependencies() -> MCPDependencies:
    """Load optional protocol dependencies at the connection boundary."""
    try:
        return MCPDependencies(
            client=import_module("mcp").Client,
            stdio=import_module("mcp.client.stdio"),
            streamable_http=import_module("mcp.client.streamable_http"),
            subscriptions=import_module("mcp.client.subscriptions"),
            http=import_module("httpx2"),
            types=import_module("mcp.types"),
            modern_protocol_versions=import_module(
                "mcp_types.version"
            ).MODERN_PROTOCOL_VERSIONS,
        )
    except ModuleNotFoundError as exc:
        raise MCPDependencyError(
            "MCP support requires the optional dependency: pip install 'agentlane[mcp]'."
        ) from exc


def exception_kind(exc: BaseException) -> MCPFailureKind:
    if isinstance(exc, BaseExceptionGroup):
        nested = cast(
            tuple[BaseException, ...],
            exc.exceptions,  # pyright: ignore[reportUnknownMemberType]
        )
        kinds = {exception_kind(item) for item in nested}
        return next(
            (
                kind
                for kind in (
                    MCPFailureKind.AUTHORIZATION,
                    MCPFailureKind.TRANSPORT,
                    MCPFailureKind.TIMEOUT,
                )
                if kind in kinds
            ),
            MCPFailureKind.PROTOCOL,
        )
    if isinstance(exc, MCPDiscoveryError):
        return exc.failure_kind
    if isinstance(exc, MCPAuthorizationError):
        return MCPFailureKind.AUTHORIZATION
    if isinstance(exc, TimeoutError):
        return MCPFailureKind.TIMEOUT
    if type(exc).__module__.startswith("httpx2") and isinstance(
        exc, load_mcp_dependencies().http.TimeoutException
    ):
        return MCPFailureKind.TIMEOUT
    if type(exc).__module__.startswith(("httpx2", "anyio")) or isinstance(exc, OSError):
        return MCPFailureKind.TRANSPORT
    if type(exc).__module__.startswith("mcp"):
        sdk = load_mcp_dependencies()
        # A lost subscription requires a fresh catalog and notification stream.
        if isinstance(exc, sdk.subscriptions.SubscriptionLost):
            return MCPFailureKind.TRANSPORT

        types = sdk.types
        code = getattr(exc, "code", None)
        if code == types.CONNECTION_CLOSED:
            return MCPFailureKind.TRANSPORT
        if code == types.REQUEST_TIMEOUT:
            return MCPFailureKind.TIMEOUT
    return MCPFailureKind.PROTOCOL
