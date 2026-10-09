"""Credentials stay outside model history, snapshots, events, and SDK logs."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, cast

import httpx2
import pytest
from mcp import types
from mcp.server import MCPServer as SDKServer
from mcp.server import Server, ServerRequestContext
from starlette.requests import Request
from starlette.types import Message, Receive, Scope, Send

from agentlane.harness import AgentDescriptor, AgentSnapshot
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.mcp import (
    MCPAccessToken,
    MCPAuthorizationContext,
    MCPClientManager,
    MCPResultPolicy,
    MCPServer,
    MCPStreamableHTTPTransport,
    MCPToolsShim,
)
from agentlane.harness.mcp._auth import product_bearer_auth, record_http_failure
from agentlane.harness.mcp._operation import mcp_operation
from agentlane.harness.mcp._redaction import contains_secret_key, redact_known_secrets
from agentlane.harness.mcp._result import render_mcp_result
from agentlane.harness.mcp._sdk import exception_kind
from agentlane.harness.mcp._validation import validate_redirect
from agentlane.messaging import AgentId
from agentlane.models import ModelTracing, ToolExecutor, ToolFailure, Tools
from agentlane.runtime import CancellationToken

from ..tools_test_utils import (
    SequenceModel,
    make_assistant_response,
    make_tool_call,
)
from .helpers import acquire_lease, http_server


def _canary(label: str) -> str:
    return f"test-only-{label}-7c4eb2"


def test_catalog_redaction_preserves_schema_keys_and_detects_secret_keys() -> None:
    schema = {
        "type": "object",
        "properties": {
            "api_key": {"type": "string", "description": _canary("catalog")}
        },
    }
    safe = redact_known_secrets(schema, {_canary("catalog")})
    assert safe == {
        "type": "object",
        "properties": {"api_key": {"type": "string", "description": "[redacted]"}},
    }
    assert not contains_secret_key(schema, {_canary("catalog")})
    assert contains_secret_key(
        {"properties": {_canary("catalog"): {"type": "string"}}},
        {_canary("catalog")},
    )


def test_known_token_cannot_escape_in_result_mapping_keys() -> None:
    secret = _canary("mapping-key")
    rendered = render_mcp_result(
        {"content": [], "structuredContent": {secret: "server-controlled key"}},
        MCPResultPolicy(),
        secrets={secret},
    )
    assert secret not in rendered
    assert (
        json.loads(rendered)["structuredContent"]["[redacted]"]
        == "server-controlled key"
    )


@pytest.mark.parametrize("block_type", ["text", "image", "audio", "resource"])
def test_result_all_content_types_redact_credentials(block_type: str) -> None:
    secret = _canary("metadata")
    block: dict[str, Any] = {
        "type": block_type,
        "_meta": {"Authorization": secret, "nested": {"api_key": secret}},
    }
    if block_type == "text":
        block["text"] = f"The server echoed {_canary('provider-token')}"
    elif block_type == "resource":
        block["resource"] = {
            "uri": f"https://example.test/callback?code={secret}&state={secret}",
            "blob": _canary("binary"),
        }
    else:
        block.update(data=_canary("binary"), mimeType=f"{block_type}/test")
    rendered = render_mcp_result(
        {"content": [block]},
        MCPResultPolicy(),
        secrets={_canary("provider-token")},
    )
    assert secret not in rendered
    assert _canary("provider-token") not in rendered
    assert _canary("binary") not in rendered
    assert "redacted" in rendered
    json.loads(rendered)


@pytest.mark.parametrize(
    "credential_key",
    [
        "api_key",
        "AWS_ACCESS_KEY_ID",
        "aws_secret_access_key",
        "X-Goog-Api-Key",
        "vertex_credentials",
        "private_key",
        "Authorization",
    ],
)
def test_result_structured_and_json_text_redact_credentials(
    credential_key: str,
) -> None:
    data = {credential_key: _canary("credential"), "key": "domain-key", "private": True}
    rendered = render_mcp_result(
        types.CallToolResult(
            content=[types.TextContent(text=json.dumps(data))],
            structured_content=data,
        ),
        MCPResultPolicy(),
    )
    assert _canary("credential") not in rendered
    payload = json.loads(rendered)
    assert payload["structuredContent"]["key"] == "domain-key"
    assert payload["structuredContent"]["private"] is True
    assert json.loads(payload["content"][0]["text"])[credential_key] == "[redacted]"


@pytest.mark.parametrize("limit", [128, 129, 180, 256, 1024])
@pytest.mark.parametrize(
    "text",
    ['A useful answer "with quotes"\n' * 1000, "答案🙂" * 1000],
    ids=["escaped-text", "unicode-text"],
)
def test_result_truncation_keeps_valid_json_and_previews(limit: int, text: str) -> None:
    rendered = render_mcp_result(
        types.CallToolResult(
            content=[types.TextContent(text=text)],
            structured_content={"summary": "useful", "details": "x" * 3000},
        ),
        MCPResultPolicy(max_text_chars=limit),
    )
    assert len(rendered) <= limit
    payload = json.loads(rendered)
    assert payload["truncated"] is True
    if limit >= 180:
        assert payload["content"][0]["text"]
    else:
        assert payload["omittedBlocks"] >= 0
    assert "omittedChars" not in payload
    assert "structuredContent" in payload
    assert payload["structuredContent"]["summary"] == "useful"


def test_resource_blocks_return_only_metadata() -> None:
    rendered = render_mcp_result(
        {
            "content": [
                {
                    "type": "resource",
                    "resource": {
                        "uri": "resource://notes/1",
                        "mimeType": "text/plain",
                        "text": _canary("resource-body"),
                    },
                }
            ]
        },
        MCPResultPolicy(),
    )
    assert _canary("resource-body") not in rendered
    resource = json.loads(rendered)["content"][0]["resource"]
    assert resource["textCharsOmitted"] == len(_canary("resource-body"))
    assert resource["uri"] == "resource://notes/1"


def test_result_omission_counts_only_omitted_blocks() -> None:
    original = {
        "content": [
            {"type": "text", "text": "one"},
            {"type": "text", "text": "two"},
            {"type": "text", "text": "three"},
        ]
    }
    rendered = render_mcp_result(original, MCPResultPolicy(max_content_blocks=1))
    payload = json.loads(rendered)
    assert payload["content"] == [{"type": "text", "text": "one"}]
    assert "omittedChars" not in payload
    assert payload["omittedBlocks"] == 2


@pytest.mark.parametrize("query", ["access_token", "code", "state", "api-key"])
def test_endpoint_rejects_sensitive_query_without_echoing_it(query: str) -> None:
    with pytest.raises(ValueError) as caught:
        MCPStreamableHTTPTransport(
            url=f"https://example.test/mcp?{query}={_canary('url')}"
        )
    assert _canary("url") not in str(caught.value)
    transport = MCPStreamableHTTPTransport(
        url="https://example.test/mcp?tenant=tenant-1"
    )
    assert transport.url.endswith("tenant=tenant-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    [
        "https://other.test/mcp",
        "http://example.test/mcp",
        f"https://example.test/mcp?access_token={_canary('redirect')}",
    ],
)
async def test_unsafe_redirect_stops_before_credentials_leave_origin(
    target: str,
) -> None:
    requests: list[httpx2.Request] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(307, request=request, headers={"location": target})

    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),
        follow_redirects=True,
        event_hooks={"response": [validate_redirect]},
        headers={"Authorization": f"Bearer {_canary('header')}"},
    ) as client:
        with pytest.raises(ValueError) as caught:
            await client.get("https://example.test/mcp")
    assert len(requests) == 1
    assert _canary("redirect") not in str(caught.value)
    assert _canary("header") not in str(caught.value)


@dataclass
class _Provider:
    fail: bool = False
    invalidated: list[str] = field(default_factory=list[str])

    async def get_access_token(
        self, server: MCPServer, context: MCPAuthorizationContext
    ) -> MCPAccessToken:
        del server, context
        if self.fail:
            raise RuntimeError(_canary("provider-exception"))
        return MCPAccessToken(token=_canary("provider-token"))

    async def invalidate_access_token(
        self,
        server: MCPServer,
        context: MCPAuthorizationContext,
        token: MCPAccessToken,
    ) -> None:
        del server, context
        self.invalidated.append(token.token)


@pytest.mark.asyncio
async def test_auth_failure_context_is_isolated_between_parallel_requests() -> None:
    provider = _Provider()
    server = MCPServer(
        name="security",
        transport=MCPStreamableHTTPTransport(url="https://example.test/mcp"),
        authorization=provider,
    )

    async def handler(request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(0)
        return httpx2.Response(
            401 if request.url.path == "/denied" else 200, request=request
        )

    async with httpx2.AsyncClient(
        auth=product_bearer_auth(httpx2, server, MCPAuthorizationContext(key="user")),
        transport=httpx2.MockTransport(handler),
        event_hooks={"response": [record_http_failure]},
    ) as client:

        async def call(path: str) -> tuple[str | None, str | None]:
            with mcp_operation() as operation:
                await client.get(f"https://example.test/{path}")
                return operation.failure_kind, operation.retry_reason

        denied, allowed = await asyncio.gather(call("denied"), call("allowed"))
    assert denied == ("authorization", "unauthorized")
    assert allowed == (None, None)
    assert len(provider.invalidated) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("final_status", "expected_kind"),
    [(200, None), (401, "authorization"), (403, "authorization"), (500, "transport")],
)
async def test_auth_retry_keeps_the_final_http_failure_category(
    final_status: int, expected_kind: str | None
) -> None:
    provider = _Provider()
    server = MCPServer(
        name="security",
        transport=MCPStreamableHTTPTransport(url="https://example.test/mcp"),
        authorization=provider,
    )
    calls = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(401 if calls == 1 else final_status, request=request)

    async with httpx2.AsyncClient(
        auth=product_bearer_auth(httpx2, server, MCPAuthorizationContext(key="user")),
        transport=httpx2.MockTransport(handler),
        event_hooks={"response": [record_http_failure]},
    ) as client:
        with mcp_operation() as operation:
            await client.get("https://example.test/mcp")
            assert operation.failure_kind == expected_kind
            assert operation.http_status == final_status
    assert calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_kind"),
    [(httpx2.ReadTimeout, "timeout"), (httpx2.ConnectError, "transport")],
)
async def test_auth_retry_without_response_keeps_transport_failure_category(
    failure: type[httpx2.TransportError], expected_kind: str
) -> None:
    provider = _Provider()
    server = MCPServer(
        name="security",
        transport=MCPStreamableHTTPTransport(url="https://example.test/mcp"),
        authorization=provider,
    )
    calls = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx2.Response(401, request=request)
        raise failure("request failed", request=request)

    async with httpx2.AsyncClient(
        auth=product_bearer_auth(httpx2, server, MCPAuthorizationContext(key="user")),
        transport=httpx2.MockTransport(handler),
        event_hooks={"response": [record_http_failure]},
    ) as client:
        with mcp_operation() as operation:
            with pytest.raises(failure) as caught:
                await client.get("https://example.test/mcp")
            assert (
                operation.failure_kind or exception_kind(caught.value)
            ) == expected_kind
            assert operation.http_status is None
            assert operation.retry_reason == "unauthorized"
    assert calls == 2
    assert len(provider.invalidated) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("has_session", [False, True])
@pytest.mark.parametrize("authorized", [False, True])
async def test_http_404_retires_only_an_established_session(
    has_session: bool, authorized: bool
) -> None:
    server = MCPServer(
        name="security",
        transport=MCPStreamableHTTPTransport(url="https://example.test/mcp"),
        authorization=_Provider() if authorized else None,
    )

    async def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(404, request=request)

    async with httpx2.AsyncClient(
        auth=product_bearer_auth(httpx2, server, MCPAuthorizationContext(key="user")),
        transport=httpx2.MockTransport(handler),
        event_hooks={"response": [record_http_failure]},
    ) as client:
        with mcp_operation() as operation:
            await client.post(
                "https://example.test/mcp",
                headers={"MCP-Session-Id": "expired"} if has_session else {},
            )
            assert operation.failure_kind == ("transport" if has_session else None)


@pytest.mark.asyncio
async def test_expired_http_session_recovers_at_next_discovery_without_replay(
    unused_tcp_port: int,
) -> None:
    sdk_server = SDKServer("expired-session-fixture")

    @sdk_server.tool()
    def echo(value: str) -> str:
        return value

    _ = echo
    inner = sdk_server.streamable_http_app(stateless_http=True, host="127.0.0.1")
    expired = False
    initializations = 0
    call_requests = 0

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal initializations, call_requests
        if scope["type"] != "http":
            await inner(scope, receive, send)
            return
        buffered: list[Message] = []
        method = None
        payload: dict[str, Any] = {}
        if scope["method"] == "POST":
            while True:
                message = await receive()
                buffered.append(message)
                if not message.get("more_body", False):
                    break
            payload = json.loads(
                b"".join(message.get("body", b"") for message in buffered)
            )
            method = payload.get("method")
        if method == "server/discover":
            # Force the SDK's supported session-based negotiation path.
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": payload["id"],
                            "error": {"code": -32601, "message": "Method not found"},
                        }
                    ).encode(),
                }
            )
            return
        if method == "initialize":
            initializations += 1
        if method == "tools/call":
            call_requests += 1
        if (
            expired
            and dict(scope.get("headers", [])).get(b"mcp-session-id") == b"session-1"
        ):
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b"Session terminated"})
            return

        async def replay() -> Message:
            return buffered.pop(0) if buffered else await receive()

        async def respond(message: Message) -> None:
            if method == "initialize" and message["type"] == "http.response.start":
                message = {
                    **message,
                    "headers": [
                        *message.get("headers", []),
                        (b"mcp-session-id", f"session-{initializations}".encode()),
                    ],
                }
            await send(message)

        await inner(scope, replay, respond)

    async with http_server(app, unused_tcp_port), MCPClientManager() as manager:
        server = MCPServer(
            name="session",
            transport=MCPStreamableHTTPTransport(
                url=f"http://127.0.0.1:{unused_tcp_port}/mcp",
                allow_insecure_http=True,
            ),
        )
        lease = await acquire_lease(
            manager, server, MCPAuthorizationContext(key="user")
        )
        tool = (await lease.tools())[0]
        expired = True
        failed = await tool.run(tool.args_type()(value="first"), CancellationToken())
        assert isinstance(failed, ToolFailure)
        assert failed.error.kind == "mcp_transport"
        assert initializations == 1
        assert call_requests == 1
        replacement = (await lease.tools())[0]
        assert initializations == 2
        result = await replacement.run(
            replacement.args_type()(value="second"), CancellationToken()
        )
        assert not isinstance(result, ToolFailure)
        assert json.loads(result)["content"][0]["text"] == "second"
        assert call_requests == 2
        await lease.release()


@dataclass
class _HTTPFixture:
    server: MCPServer
    provider: _Provider
    denied_calls: int = 0


@asynccontextmanager
async def _http_server(port: int) -> AsyncIterator[_HTTPFixture]:
    sdk_server = SDKServer("security-fixture")

    @sdk_server.tool()
    def echo(value: str) -> dict[str, str]:
        return {
            "echo": value,
            "api_key": _canary("server-secret"),
            "authorization_echo": _canary("provider-token"),
        }

    _ = echo
    inner = sdk_server.streamable_http_app(stateless_http=True, host="127.0.0.1")
    provider = _Provider()
    fixture = _HTTPFixture(
        server=MCPServer(
            name="security",
            transport=MCPStreamableHTTPTransport(
                url=f"http://127.0.0.1:{port}/mcp", allow_insecure_http=True
            ),
            authorization=provider,
            tool_timeout_seconds=2,
        ),
        provider=provider,
    )

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await inner(scope, receive, send)
            return
        buffered: list[Message] = []
        while True:
            message = await receive()
            buffered.append(message)
            if not message.get("more_body", False):
                break
        payload = json.loads(b"".join(message.get("body", b"") for message in buffered))
        if (
            payload.get("method") == "tools/call"
            and payload.get("params", {}).get("arguments", {}).get("value") == "denied"
        ):
            fixture.denied_calls += 1
            await send({"type": "http.response.start", "status": 401, "headers": []})
            await send({"type": "http.response.body", "body": b"Unauthorized"})
            return

        async def replay() -> Message:
            return buffered.pop(0) if buffered else await receive()

        await inner(scope, replay, send)

    async with http_server(app, port):
        yield fixture


@pytest.mark.asyncio
async def test_http_sdk_preserves_auth_failure_kind(unused_tcp_port: int) -> None:
    async with _http_server(unused_tcp_port) as fixture, MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, fixture.server, MCPAuthorizationContext(key="user")
        )
        tool = (await lease.tools())[0]
        result = await tool.run(
            tool.args_type().model_validate({"value": "denied"}), CancellationToken()
        )
        assert isinstance(result, ToolFailure)
        assert result.error.kind == "mcp_authorization"
        assert fixture.denied_calls == 2
        assert len(fixture.provider.invalidated) == 2
        fixture.provider.fail = True
        result = await tool.run(
            tool.args_type().model_validate({"value": "provider failure"}),
            CancellationToken(),
        )
        assert isinstance(result, ToolFailure)
        assert result.error.kind == "mcp_authorization"
        assert _canary("provider-exception") not in result
        await lease.release()


@pytest.mark.asyncio
async def test_sdk_logs_obey_boundary_without_affecting_unrelated_logs(
    unused_tcp_port: int, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="mcp.client.streamable_http")
    async with _http_server(unused_tcp_port) as fixture, MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, fixture.server, MCPAuthorizationContext(key="user")
        )
        tools = await lease.tools()
        await ToolExecutor().execute(
            tool_calls=[
                make_tool_call(
                    tool_id="safe-call",
                    name="security__echo",
                    arguments=json.dumps({"value": _canary("tool-argument")}),
                )
            ],
            tools=Tools(tools=tools),
            tracing=ModelTracing.DISABLED,
        )
        await lease.release()
    logging.getLogger("mcp.client.streamable_http").debug("unrelated-client-message")
    sdk_messages = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name.startswith(("mcp.client", "httpx2", "httpcore2"))
    )
    assert _canary("tool-argument") not in sdk_messages
    assert _canary("provider-token") not in sdk_messages
    assert _canary("server-secret") not in sdk_messages
    assert "unrelated-client-message" in sdk_messages


@pytest.mark.asyncio
async def test_real_agent_events_and_snapshot_do_not_persist_credentials(
    unused_tcp_port: int,
) -> None:
    async with _http_server(unused_tcp_port) as fixture, MCPClientManager() as manager:
        model = SequenceModel(
            [
                make_assistant_response(
                    None,
                    tool_calls=[
                        make_tool_call(
                            tool_id="safe-call",
                            name="security__echo",
                            arguments='{"value":"hello"}',
                        )
                    ],
                ),
                make_assistant_response("Complete"),
            ]
        )
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Security fixture",
                model=model,
                shims=(
                    MCPToolsShim(
                        servers=(fixture.server,),
                        client_manager=manager,
                        authorization_context=MCPAuthorizationContext(
                            key="user", value={"credential": _canary("opaque-context")}
                        ),
                    ),
                ),
            )
        )
        stream = await agent.run_events("Call echo")
        try:
            events = [event.to_dict() async for event in stream]
            result = await stream.result()
        finally:
            await stream.aclose()
        assert result.run_state is not None
        snapshot = AgentSnapshot.capture(
            agent_id=AgentId.from_values("security", "main"), run_state=result.run_state
        )
        encoded = json.dumps({"events": events, "snapshot": snapshot.to_json()})
        for label in ("provider-token", "server-secret", "opaque-context"):
            assert _canary(label) not in encoded
        assert "[redacted]" in encoded
        assert "HTTPClient" not in encoded
        assert "_Provider" not in encoded
        assert "ClientSession" not in encoded
        assert "hello" in encoded
        restored = AgentSnapshot.from_json(snapshot.to_json()).to_run_state()
        assert restored.responses == result.run_state.responses
        assert snapshot.schema_version == 1


@pytest.mark.asyncio
async def test_concurrent_practitioners_isolate_auth_sessions_catalogs_and_sse_calls(
    unused_tcp_port: int,
) -> None:
    practitioners = ("alice", "bob")
    listed: list[str] = []
    called: list[str] = []
    sessions = {name: set[str]() for name in practitioners}
    response_types: list[bytes] = []
    both_calls_started = asyncio.Event()

    def practitioner(context: ServerRequestContext[Any]) -> str:
        request = cast(Request, context.request)
        bearer = request.headers.get("authorization")
        identity = next(
            name
            for name in practitioners
            if bearer == f"Bearer {_canary(f'practitioner-{name}')}"
        )
        session_id = request.headers.get("mcp-session-id")
        assert session_id is not None
        sessions[identity].add(session_id)
        return identity

    async def list_tools(
        context: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        del params
        identity = practitioner(context)
        listed.append(identity)
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=f"read_{identity}",
                    input_schema={"type": "object", "properties": {}},
                )
            ]
        )

    async def call_tool(
        context: ServerRequestContext[Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        identity = practitioner(context)
        assert params.name == f"read_{identity}"
        called.append(identity)
        if len(called) == len(practitioners):
            both_calls_started.set()
        await asyncio.wait_for(both_calls_started.wait(), timeout=2)
        return types.CallToolResult(
            content=[types.TextContent(text=f"{identity} notes")]
        )

    sdk_server = Server(
        "practitioner-isolation", on_list_tools=list_tools, on_call_tool=call_tool
    )
    inner = sdk_server.streamable_http_app(
        stateless_http=False, json_response=False, host="127.0.0.1"
    )

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await inner(scope, receive, send)
            return
        buffered: list[Message] = []
        while True:
            message = await receive()
            buffered.append(message)
            if not message.get("more_body", False):
                break
        payload = json.loads(b"".join(message.get("body", b"") for message in buffered))
        if payload.get("method") == "server/discover":
            # Use the SDK's legacy handshake so this test observes actual
            # server-issued HTTP session IDs, as well as SSE tool responses.
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b"Not Found"})
            return

        async def replay() -> Message:
            return buffered.pop(0) if buffered else await receive()

        async def capture_response(message: Message) -> None:
            if (
                message["type"] == "http.response.start"
                and payload.get("method") == "tools/call"
            ):
                response_types.append(dict(message["headers"])[b"content-type"])
            await send(message)

        await inner(scope, replay, capture_response)

    class PractitionerProvider:
        async def get_access_token(
            self, server: MCPServer, context: MCPAuthorizationContext
        ) -> MCPAccessToken:
            del server
            assert context.key in practitioners
            return MCPAccessToken(token=_canary(f"practitioner-{context.key}"))

        async def invalidate_access_token(
            self,
            server: MCPServer,
            context: MCPAuthorizationContext,
            token: MCPAccessToken,
        ) -> None:
            raise AssertionError("The server accepts both test practitioner tokens.")

    server = MCPServer(
        name="notes",
        transport=MCPStreamableHTTPTransport(
            url=f"http://127.0.0.1:{unused_tcp_port}/mcp", allow_insecure_http=True
        ),
        authorization=PractitionerProvider(),
    )
    async with http_server(app, unused_tcp_port), MCPClientManager() as manager:

        async def run(identity: str) -> None:
            visible_name = f"notes__read_{identity}"
            model = SequenceModel(
                [
                    make_assistant_response(
                        None,
                        tool_calls=[
                            make_tool_call(
                                tool_id=identity, name=visible_name, arguments="{}"
                            )
                        ],
                    ),
                    make_assistant_response("Complete"),
                ]
            )
            agent = DefaultAgent(
                descriptor=AgentDescriptor(
                    name=f"{identity} assistant",
                    model=model,
                    shims=(
                        MCPToolsShim(
                            servers=(server,),
                            client_manager=manager,
                            authorization_context=MCPAuthorizationContext(key=identity),
                        ),
                    ),
                )
            )
            result = await agent.run("Read my notes")
            assert result.final_output == "Complete"
            assert model.call_tools[0] is not None
            assert [tool.name for tool in model.call_tools[0].normalized_tools] == [
                visible_name
            ]
            messages = json.dumps(model.calls[-1])
            assert f"{identity} notes" in messages
            assert _canary(f"practitioner-{identity}") not in messages

        await asyncio.gather(*(run(identity) for identity in practitioners))
        for identity in practitioners:
            lease = await acquire_lease(
                manager, server, MCPAuthorizationContext(key=identity)
            )
            try:
                assert [tool.name for tool in await lease.tools()] == [
                    f"notes__read_{identity}"
                ]
            finally:
                await lease.release()
        assert sorted(listed) == sorted(called) == list(practitioners)
        assert all(len(ids) == 1 for ids in sessions.values())
        assert sessions["alice"].isdisjoint(sessions["bob"])
        assert len(response_types) == 2
        assert all(value.startswith(b"text/event-stream") for value in response_types)
