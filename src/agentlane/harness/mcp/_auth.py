"""Keep product credentials and authorization generations inside MCP runtime."""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from ._errors import MCPAuthorizationError, MCPFailureKind
from ._operation import current_mcp_operation
from ._types import MCPAccessToken, MCPAuthorizationContext, MCPServer


@dataclass(slots=True)
class MCPAuthorizationState:
    """In-memory credential generation shared across connection replacements."""

    generation: int = 0
    secrets: set[str] = field(default_factory=set[str], repr=False)
    _identity: bytes | None = field(default=None, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def reject(self) -> None:
        """Make every previously discovered catalog ineligible for reuse."""
        self.generation += 1

    async def get_token(
        self, server: MCPServer, context: MCPAuthorizationContext
    ) -> MCPAccessToken | None:
        provider = server.authorization
        operation = current_mcp_operation()
        if provider is None:
            if operation is not None:
                operation.authorization_generation = self.generation
            return None

        # Complete provider lookups in order, so a delayed older lookup cannot
        # overwrite the credential identity observed by a newer lookup.
        async with self._lock:
            try:
                token = await provider.get_access_token(server, context)
                self.secrets.add(token.token)
                if operation is not None:
                    operation.secrets.add(token.token)

                identity = hashlib.sha256(
                    json.dumps([token.token, sorted(set(token.scopes))]).encode()
                ).digest()
            except Exception:
                self.reject()
                if operation is not None:
                    operation.failure_kind = MCPFailureKind.AUTHORIZATION

                raise MCPAuthorizationError("MCP authorization failed.") from None

            if identity != self._identity:
                self._identity = identity
                self.generation += 1

            if operation is not None:
                operation.authorization_generation = self.generation

            return token


class _HTTPResponse(Protocol):
    status_code: int


async def record_http_failure(response: Any) -> None:
    """Keep an HTTP failure category before the SDK converts its response."""
    _record_http_status(
        response.status_code, session_id=response.request.headers.get("mcp-session-id")
    )


def _record_http_status(status: int, *, session_id: str | None = None) -> None:
    operation = current_mcp_operation()
    if operation is None:
        return

    operation.http_status = status
    if status in {401, 403}:
        operation.failure_kind = MCPFailureKind.AUTHORIZATION
    elif status >= 500 or status == 429 or (status == 404 and session_id):
        # An established MCP session can expire independently of the endpoint.
        # Reconnect at the next discovery boundary, never replay this request.
        operation.failure_kind = MCPFailureKind.TRANSPORT
    else:
        operation.failure_kind = None


def product_bearer_auth(
    httpx2: Any,
    server: MCPServer,
    context: MCPAuthorizationContext,
    state: MCPAuthorizationState | None = None,
) -> Any:
    """Get current product credentials and retry exactly one rejected token."""
    authorization = state if state is not None else MCPAuthorizationState()

    class ProductBearerAuth(httpx2.Auth):  # type: ignore[misc]
        async def async_auth_flow(self, request: Any) -> Any:
            provider = server.authorization
            if provider is None:
                yield request
                return

            operation = current_mcp_operation()
            # Capture one caller for the whole exchange, including 401 recovery.
            # Requests without a lease use the transport's lifecycle identity.
            request_context = (
                operation.authorization_context
                if operation is not None and operation.authorization_context is not None
                else context
            )
            token = await authorization.get_token(server, request_context)
            assert token is not None
            # The provider can rotate credentials after the lease checked its
            # tool binding. Reject that change before sending the first call.
            # The explicit 401 refresh below remains a separate bounded retry.
            if (
                operation is not None
                and operation.expected_authorization_generation is not None
                and operation.expected_authorization_generation
                != authorization.generation
            ):
                operation.failure_kind = MCPFailureKind.AUTHORIZATION
                raise MCPAuthorizationError(
                    "MCP authorization changed before dispatch."
                )
            request.headers["Authorization"] = f"Bearer {token.token}"
            response = cast(_HTTPResponse, (yield request))
            if response.status_code != 401:
                if response.status_code == 403:
                    authorization.reject()

                _record_http_status(
                    response.status_code,
                    session_id=request.headers.get("mcp-session-id"),
                )
                return

            authorization.reject()
            if operation is not None:
                operation.retry_reason = "unauthorized"

            try:
                await provider.invalidate_access_token(server, request_context, token)
                replacement = await authorization.get_token(server, request_context)
                assert replacement is not None
            except Exception:
                authorization.reject()
                if operation is not None:
                    operation.failure_kind = MCPFailureKind.AUTHORIZATION

                raise MCPAuthorizationError(
                    "MCP authorization refresh failed."
                ) from None

            request.headers["Authorization"] = f"Bearer {replacement.token}"
            if operation is not None:
                # A retry may fail before receiving another HTTP response. The
                # rejected token must not mask that request's transport error.
                operation.failure_kind = None
                operation.http_status = None

            retry_response = cast(_HTTPResponse, (yield request))
            _record_http_status(
                retry_response.status_code,
                session_id=request.headers.get("mcp-session-id"),
            )
            if retry_response.status_code in {401, 403}:
                authorization.reject()

            if retry_response.status_code == 401:
                try:
                    await provider.invalidate_access_token(
                        server, request_context, replacement
                    )
                except Exception:
                    raise MCPAuthorizationError(
                        "MCP rejected-token invalidation failed."
                    ) from None

    return ProductBearerAuth()
