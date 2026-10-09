"""Native tool schemas must not expose mutable shared catalog state."""

import asyncio
from copy import deepcopy
from typing import Any, cast

import pytest
from mcp import types
from mcp.server import Server, ServerRequestContext

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.mcp import (
    MCPClientManager,
    MCPServer,
    MCPStreamableHTTPTransport,
    MCPToolsShim,
)

from ..tools_test_utils import SequenceModel, make_assistant_response
from .helpers import http_server


def _parameters(model: SequenceModel) -> dict[str, Any]:
    tools = model.call_tools[0]
    assert tools is not None
    tool = tools.normalized_tools[0]
    assert tool.strict is False
    return cast(dict[str, Any], tool.schema["parameters"])


def _agent(
    model: SequenceModel, server: MCPServer, manager: MCPClientManager
) -> DefaultAgent:
    return DefaultAgent(
        descriptor=AgentDescriptor(
            name="Schema reader",
            model=model,
            shims=(MCPToolsShim(servers=(server,), client_manager=manager),),
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("overlap", [False, True])
async def test_cached_schema_mutation_stays_in_its_run(
    unused_tcp_port: int, overlap: bool
) -> None:
    original: dict[str, Any] = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Original text."},
            "options": {
                "type": "object",
                "properties": {
                    "labels": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["labels"],
            },
        },
        "required": ["message"],
    }
    list_calls = 0

    async def list_tools(
        context: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        del context, params
        nonlocal list_calls
        list_calls += 1
        return types.ListToolsResult(
            tools=[types.Tool(name="echo", input_schema=deepcopy(original))],
            ttl_ms=300_000,
        )

    sdk_server = Server("schema-isolation", on_list_tools=list_tools)
    app = sdk_server.streamable_http_app(stateless_http=True, host="127.0.0.1")
    server = MCPServer(
        name="notes",
        transport=MCPStreamableHTTPTransport(
            url=f"http://127.0.0.1:{unused_tcp_port}/mcp", allow_insecure_http=True
        ),
    )
    first, second, later = (
        SequenceModel([make_assistant_response("done")]) for _ in range(3)
    )

    async with http_server(app, unused_tcp_port), MCPClientManager() as manager:
        first_agent = _agent(first, server, manager)
        second_agent = _agent(second, server, manager)
        if overlap:
            await asyncio.gather(
                first_agent.run("Inspect"), second_agent.run("Inspect")
            )
        else:
            await first_agent.run("Inspect")

        exposed = _parameters(first)
        exposed["properties"]["message"]["description"] = "First run annotation."
        exposed["properties"]["options"]["required"].append("private")
        exposed["required"].append("options")

        if not overlap:
            await second_agent.run("Inspect")
        await _agent(later, server, manager).run("Inspect again")

        assert _parameters(second) == original
        assert _parameters(later) == original
        assert _parameters(second) is not exposed
        assert list_calls == 1
