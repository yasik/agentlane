"""Shared MCP lifecycle messages and operation status labels."""

AUTHORIZATION_CHANGED_AFTER_DISCOVERY = "MCP authorization changed after discovery."
"""Returned as a tool failure when its discovered authorization is no longer valid."""

AUTHORIZATION_CHANGED_DURING_DISCOVERY = "MCP authorization changed during discovery."
"""Used in errors when authorization changes before a catalog can be accepted."""

CLIENT_CLOSED = "MCP client manager is closed."
"""Used in errors when an operation requires a client that is already closed."""

CONNECTION_UNAVAILABLE = "MCP connection is unavailable."
"""Returned as a tool failure when its MCP connection cannot accept the call."""

STARTUP_CANCELLED = "MCP connection startup was cancelled."
"""Used to fail connection readiness when startup is cancelled."""

TOOL_CANCELLED = "MCP tool call was cancelled."
"""Returned as a tool failure when an MCP call is cancelled."""

SHUTDOWN_TIMED_OUT = "MCP shutdown timed out; connection cleanup is still in progress."
"""Used in timeout errors while owned connection cleanup continues."""

STATUS_OK = "ok"
"""Logged when MCP discovery or tool execution completes successfully."""

STATUS_ERROR = "error"
"""Logged when an MCP tool result reports a failure."""

STATUS_STALE = "stale"
"""Logged when cached discovery data is used after a recoverable refresh failure."""

STATUS_UNAVAILABLE = "unavailable"
"""Logged when an unexpected error ends the MCP tool-change subscription."""
