"""Errors raised at MCP lifecycle and authorization boundaries."""

from strenum import LowercaseStrEnum


class MCPFailureKind(LowercaseStrEnum):
    """Failure categories used by MCP connection and discovery policy."""

    AUTHORIZATION = "authorization"
    TRANSPORT = "transport"
    TIMEOUT = "timeout"
    CAPACITY = "capacity"
    PROTOCOL = "protocol"


MCP_TRANSPORT_TOOL_ERROR_KIND = f"mcp_{MCPFailureKind.TRANSPORT.value}"
"""Identifies tool failures caused by an unavailable MCP transport."""

MCP_AUTHORIZATION_TOOL_ERROR_KIND = f"mcp_{MCPFailureKind.AUTHORIZATION.value}"
"""Identifies tool failures caused by invalid or changed MCP authorization."""

MCP_SERVER_TOOL_ERROR_KIND = "mcp_server"
"""Identifies tool results that the MCP server marks as errors."""


class MCPError(RuntimeError):
    """Base error for MCP discovery and lifecycle failures."""


class MCPDependencyError(MCPError):
    """The optional MCP SDK is not installed."""


class MCPShutdownTimeoutError(MCPError, TimeoutError):
    """The close deadline passed while owned cleanup continues in the background."""


class MCPDiscoveryError(MCPError):
    """An MCP server tool catalog could not be discovered."""

    def __init__(
        self, message: str, *, failure_kind: MCPFailureKind = MCPFailureKind.PROTOCOL
    ) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind

    @property
    def retryable(self) -> bool:
        """Return whether a later read-only discovery attempt can recover."""
        return self.failure_kind in {
            MCPFailureKind.TRANSPORT,
            MCPFailureKind.TIMEOUT,
            MCPFailureKind.CAPACITY,
        }


class MCPPoolCapacityError(MCPDiscoveryError):
    """All connection capacity is held by active or closing connections."""

    def __init__(self) -> None:
        super().__init__(
            "MCP connection capacity is in use.", failure_kind=MCPFailureKind.CAPACITY
        )


class MCPAuthorizationError(MCPError):
    """The product authorization provider could not authorize a request."""
