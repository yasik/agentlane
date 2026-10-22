"""Native tool aliases, schema conversion, and one-shot tool execution."""

import asyncio
import hashlib
import re
import time
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict

from agentlane.models import (
    Tool,
    ToolError,
    ToolErrorKind,
    ToolExecutionContext,
    ToolFailure,
)
from agentlane.runtime import CancellationToken

from ..filesystem import FileWriter
from ._connection import MCPConnection
from ._constants import CONNECTION_UNAVAILABLE, STATUS_ERROR, STATUS_OK, TOOL_CANCELLED
from ._errors import (
    MCP_AUTHORIZATION_TOOL_ERROR_KIND,
    MCP_TRANSPORT_TOOL_ERROR_KIND,
    MCPDiscoveryError,
    MCPFailureKind,
)
from ._operation import mcp_operation
from ._output import render_mcp_result_with_storage
from ._sdk import exception_kind
from ._types import MCPRemoteTool, MCPServer

logger = structlog.get_logger(__name__)


class _MCPArguments(BaseModel):
    model_config = ConfigDict(extra="allow")


type _ToolCall = Callable[
    [str, dict[str, Any], CancellationToken], Awaitable[str | ToolFailure]
]


def native_tool(
    server: MCPServer, remote_tool: MCPRemoteTool, dispatch: _ToolCall
) -> Tool[Any, Any]:
    """Bind metadata to a lease dispatch, not to a transport generation."""

    async def handler(
        args: _MCPArguments,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> str | ToolFailure:
        del context
        return await dispatch(
            remote_tool.name, args.model_dump(exclude_none=False), cancellation_token
        )

    description = (
        remote_tool.description or f"Call {remote_tool.name} on {server.name}."
    )
    return Tool(
        name=_visible_tool_name(server.name, remote_tool.name),
        description=f"[{server.name}] {description}",
        args_model=_MCPArguments,
        # Native tools belong to a run; the cached catalog stays private.
        parameters_schema=deepcopy(remote_tool.input_schema),
        handler=handler,
        strict=False,
        retry_on_timeout=False,
    )


async def call_tool(
    connection: MCPConnection,
    name: str,
    arguments: dict[str, Any],
    cancellation_token: CancellationToken,
    is_closed: Callable[[], bool],
    timeout_seconds: float,
    *,
    result_writer: FileWriter | None = None,
    result_directory: str | None = None,
) -> str | ToolFailure:
    if cancellation_token.is_cancelled:
        return tool_failure(TOOL_CANCELLED, ToolErrorKind.CANCELLED.value)
    if is_closed() or connection.failed or connection.closing:
        return tool_failure(CONNECTION_UNAVAILABLE, MCP_TRANSPORT_TOOL_ERROR_KIND)
    started_at = time.monotonic()
    with mcp_operation() as operation:
        operation.expected_authorization_generation = (
            connection.authorization.generation
        )
        call = asyncio.create_task(
            connection.get_client().call_tool(
                name,
                arguments,
                read_timeout_seconds=timeout_seconds,
            )
        )
        connection.active_calls.add(call)
        cancellation_token.link_future(call)
        try:
            result = await asyncio.wait_for(call, timeout=timeout_seconds)
        except asyncio.CancelledError:
            if cancellation_token.is_cancelled:
                return tool_failure(TOOL_CANCELLED, ToolErrorKind.CANCELLED.value)
            if operation.failure_kind == MCPFailureKind.AUTHORIZATION.value:
                connection.authorization.reject()
                return tool_failure(
                    "MCP authorization failure.", MCP_AUTHORIZATION_TOOL_ERROR_KIND
                )
            if is_closed() or connection.closing:
                return tool_failure(
                    CONNECTION_UNAVAILABLE, MCP_TRANSPORT_TOOL_ERROR_KIND
                )
            raise
        except Exception as exc:
            failure_kind = operation.failure_kind or exception_kind(exc)
            if failure_kind == MCPFailureKind.TRANSPORT.value:
                connection.failed = True
            if failure_kind == MCPFailureKind.AUTHORIZATION.value:
                connection.authorization.reject()
            logger.warning(
                "mcp_tool_failed",
                server=connection.server.name,
                tool=name,
                duration=time.monotonic() - started_at,
                status=failure_kind,
                protocol_version=connection.protocol_version,
                retry_reason=operation.retry_reason,
            )
            kind = (
                MCPFailureKind.TIMEOUT.value
                if failure_kind == MCPFailureKind.TIMEOUT.value
                else f"mcp_{failure_kind}"
            )
            return tool_failure(f"MCP {failure_kind} failure.", kind)
        finally:
            connection.active_calls.discard(call)
            connection.secrets.update(operation.secrets)

        if result_writer is not None:
            if cancellation_token.is_cancelled:
                return tool_failure(TOOL_CANCELLED, ToolErrorKind.CANCELLED.value)
            if is_closed() or connection.closing:
                return tool_failure(
                    CONNECTION_UNAVAILABLE, MCP_TRANSPORT_TOOL_ERROR_KIND
                )

        preparation = render_mcp_result_with_storage(
            result,
            policy=connection.server.result_policy,
            secrets=connection.secrets,
            writer=result_writer,
            directory=result_directory,
        )
        if result_writer is None:
            rendered = await preparation
        else:
            # Shutdown must cancel and drain storage after the provider call
            # ends, before the connection owner releases its resources.
            preparing = asyncio.create_task(preparation)
            connection.active_calls.add(preparing)
            cancellation_token.link_future(preparing)
            try:
                rendered = await preparing
            except asyncio.CancelledError:
                if cancellation_token.is_cancelled:
                    return tool_failure(TOOL_CANCELLED, ToolErrorKind.CANCELLED.value)
                if is_closed() or connection.closing:
                    return tool_failure(
                        CONNECTION_UNAVAILABLE, MCP_TRANSPORT_TOOL_ERROR_KIND
                    )
                raise
            finally:
                connection.active_calls.discard(preparing)

            # A writer can finish its write while handling cancellation.
            # Do not publish its reference after the run or connection closes.
            if cancellation_token.is_cancelled:
                return tool_failure(TOOL_CANCELLED, ToolErrorKind.CANCELLED.value)
            if is_closed() or connection.closing:
                return tool_failure(
                    CONNECTION_UNAVAILABLE, MCP_TRANSPORT_TOOL_ERROR_KIND
                )

        logger.info(
            "mcp_tool_completed",
            server=connection.server.name,
            tool=name,
            duration=time.monotonic() - started_at,
            status=STATUS_ERROR if isinstance(rendered, ToolFailure) else STATUS_OK,
            protocol_version=connection.protocol_version,
            retry_reason=operation.retry_reason,
        )
        return rendered


def _normalize_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    if not normalized:
        raise MCPDiscoveryError("MCP tool or server name cannot be normalized.")
    if normalized[0].isdigit():
        normalized = f"mcp_{normalized}"
    return normalized.lower()


def _visible_tool_name(server_name: str, tool_name: str) -> str:
    name = f"{_normalize_name(server_name)}__{_normalize_name(tool_name)}"
    if len(name) <= 64:
        return name
    # Native tool adapters share a 64-character provider name limit. Keep the
    # alias stable across discovery while dispatching with the original name.
    suffix = hashlib.sha256(name.encode()).hexdigest()[:12]
    return f"{name[:51]}_{suffix}"


def reject_normalized_collisions(
    server_name: str, tools: tuple[MCPRemoteTool, ...]
) -> None:
    seen: set[str] = set()
    for tool in tools:
        visible = _visible_tool_name(server_name, tool.name)
        if visible in seen:
            raise MCPDiscoveryError(f"Multiple MCP tools normalize to {visible!r}.")
        seen.add(visible)


def tool_failure(message: str, kind: str) -> ToolFailure:
    return ToolFailure(text=message, error=ToolError(message=message, kind=kind))
