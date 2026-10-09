"""Public configuration and authorization contracts for MCP tools."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from math import isfinite
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from ._validation import validate_http_url


@dataclass(frozen=True, slots=True)
class MCPClientLimits:
    """Resource limits for one application-owned MCP client manager."""

    max_connections: int = 64
    idle_timeout_seconds: float = 300.0
    shutdown_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if type(self.max_connections) is not int or self.max_connections < 1:
            raise ValueError("MCP max_connections must be at least 1.")
        _require_positive_timeout(self.idle_timeout_seconds, "idle timeout")
        _require_positive_timeout(self.shutdown_timeout_seconds, "shutdown timeout")


@dataclass(frozen=True, slots=True)
class MCPRemoteTool:
    name: str
    description: str | None
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MCPCatalog:
    tools: tuple[MCPRemoteTool, ...]
    revision: int
    expires_at: float
    authorization_generation: int


@dataclass(frozen=True, slots=True)
class MCPAccessToken:
    """One short-lived bearer token supplied by the host application."""

    token: str = field(repr=False)
    expires_at: datetime | None = None
    scopes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.token:
            raise ValueError("MCP access token must not be empty.")
        if self.expires_at is not None and self.expires_at.tzinfo is None:
            raise ValueError("MCP access token expiry must include a timezone.")

    @property
    def is_expired(self) -> bool:
        """Return whether the token has passed its advertised expiry."""
        return self.expires_at is not None and self.expires_at <= datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class MCPAuthorizationContext:
    """Opaque application identity used to isolate MCP connections.

    `key` is a stable, non-secret cache partition, such as a practitioner ID.
    `value` is passed back to the authorization provider and is never persisted.
    """

    key: str
    value: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("MCP authorization context key must not be empty.")


# Keep the server/provider type cycle lazy.
type _MCPServerRef = MCPServer


@runtime_checkable
class MCPAuthorizationProvider(Protocol):
    """Product-owned source of short-lived MCP access tokens."""

    async def get_access_token(
        self,
        server: _MCPServerRef,
        context: MCPAuthorizationContext,
    ) -> MCPAccessToken:
        """Return a current bearer token for one server and identity."""
        ...

    async def invalidate_access_token(
        self,
        server: _MCPServerRef,
        context: MCPAuthorizationContext,
        token: MCPAccessToken,
    ) -> None:
        """Invalidate a token rejected by the MCP server."""
        ...


@dataclass(frozen=True, slots=True)
class MCPStreamableHTTPTransport:
    """Configuration for a remote Streamable HTTP MCP server."""

    url: str
    allow_insecure_http: bool = False
    connect_timeout_seconds: float = 30.0
    read_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        validate_http_url(self.url, self.allow_insecure_http)
        _require_positive_timeout(self.connect_timeout_seconds, "connect timeout")
        _require_positive_timeout(self.read_timeout_seconds, "read timeout")


@dataclass(frozen=True, slots=True)
class MCPStdioTransport:
    """Configuration for a local stdio MCP server process."""

    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    cwd: str | None = None

    def __post_init__(self) -> None:
        if not self.command.strip():
            raise ValueError("MCP stdio command must not be empty.")
        if self.env is not None:
            object.__setattr__(self, "env", MappingProxyType(dict(self.env)))


type MCPTransport = MCPStreamableHTTPTransport | MCPStdioTransport


@dataclass(frozen=True, slots=True)
class MCPToolFilter:
    """Include and exclude model-visible MCP tools by original tool name."""

    include: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ()

    def allows(self, name: str) -> bool:
        """Return whether one original MCP tool name passes this filter."""
        included = any(fnmatchcase(name, pattern) for pattern in self.include)
        excluded = any(fnmatchcase(name, pattern) for pattern in self.exclude)
        return included and not excluded


@dataclass(frozen=True, slots=True)
class MCPResultPolicy:
    """Limits for MCP content returned to the model conversation."""

    max_content_blocks: int = 32
    max_text_chars: int = 50 * 1024
    include_structured_content: bool = True

    def __post_init__(self) -> None:
        if self.max_content_blocks < 1:
            raise ValueError("MCP result max_content_blocks must be at least 1.")
        if self.max_text_chars < 128:
            raise ValueError("MCP result max_text_chars must be at least 128.")


@dataclass(frozen=True, slots=True)
class MCPServer:
    """One MCP server exposed to a harness agent."""

    name: str
    transport: MCPTransport
    authorization: MCPAuthorizationProvider | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    tools: MCPToolFilter = field(default_factory=MCPToolFilter)
    result_policy: MCPResultPolicy = field(default_factory=MCPResultPolicy)
    required: bool = True
    connect_timeout_seconds: float = 30.0
    discovery_timeout_seconds: float = 30.0
    tool_timeout_seconds: float = 120.0
    catalog_ttl_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("MCP server name must not be empty.")
        if (
            isinstance(self.transport, MCPStdioTransport)
            and self.authorization is not None
        ):
            raise ValueError("MCP authorization is supported only for HTTP servers.")
        _require_positive_timeout(self.connect_timeout_seconds, "connection timeout")
        _require_positive_timeout(self.discovery_timeout_seconds, "discovery timeout")
        _require_positive_timeout(self.tool_timeout_seconds, "tool timeout")
        _require_positive_timeout(self.catalog_ttl_seconds, "catalog TTL")


def _require_positive_timeout(value: float, label: str) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"MCP {label} must be finite and greater than zero.")
