# MCP

`MCPToolsShim` adds tools from Model Context Protocol (MCP) servers to an
AgentLane agent. It supports Streamable HTTP and local stdio servers.
Install the optional dependency:

```bash
uv add 'agentlane[mcp]'
```

## Connect an agent

Attach the shim through `AgentDescriptor.shims`. Replace the server URL and
use your configured AgentLane `model`:

```python
from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.mcp import (
    MCPServer,
    MCPStreamableHTTPTransport,
    MCPToolsShim,
)

notes = MCPServer(
    name="notes",
    transport=MCPStreamableHTTPTransport(url="https://notes.example.com/mcp"),
)

agent = DefaultAgent(
    descriptor=AgentDescriptor(
        name="Notes assistant",
        model=model,
        shims=(MCPToolsShim(servers=(notes,)),),
    ),
)

result = await agent.run("Find the latest launch decision in the notes.")
print(result.final_output)
```

The shim opens and closes connections for each run. For a runnable local
example, see the
[MCP meeting assistant](../../../examples/harness/mcp_meeting_assistant/README.md).

HTTP servers require HTTPS. For development HTTP, set
`MCPStreamableHTTPTransport(..., allow_insecure_http=True)`. Supply credentials
through an authorization provider, not the URL. Legacy SSE, MCP resources,
and MCP prompts are not supported as model capabilities.

### Use a local server

Replace the HTTP transport with `MCPStdioTransport`. Set the command, working
directory, and environment for your server:

```python
from agentlane.harness.mcp import MCPStdioTransport

notes = MCPServer(
    name="notes",
    transport=MCPStdioTransport(
        command="python",
        args=("-m", "my_notes_mcp"),
        cwd="/srv/my-product",
        env={"NOTES_DATABASE": "/data/notes.db"},
    ),
)
```

Stdio inherits only the MCP SDK's allowed environment variables plus explicit
`env` values. Pass the variables your server needs.

## Select tools

The model sees names such as `notes__list_meetings`. Use these names in
`RESTRICT_TOOLS`, `OVERRIDE_TOOLS`, and `ExcludeToolsShim` policies.
Names are normalized; use simple, unique server names to avoid collisions.

To expose only selected tools, set `MCPServer.tools`. Filter patterns match
the original names reported by the server:

```python
from agentlane.harness.mcp import MCPToolFilter

notes = MCPServer(
    name="notes",
    transport=MCPStreamableHTTPTransport(url="https://notes.example.com/mcp"),
    tools=MCPToolFilter(include=("list_*", "get_*"), exclude=("*_transcript",)),
)
```

Place `MCPToolsShim` before `SkillsShim` when skill tool rules must cover MCP
tools. For subagent and handoff policies, see [shims](../shims.md).

## Authorization

Your app handles OAuth consent, credential storage, refresh, and revocation.
Implement `MCPAuthorizationProvider` with two async methods:

- `get_access_token(server, context)` returns a valid `MCPAccessToken` with
  `token` and optional `expires_at` and `scopes`.
- `invalidate_access_token(server, context, token)` invalidates a token that
  the server rejected.

Attach your `provider` to the HTTP server and pass the user's identity:

```python
from agentlane.harness.mcp import MCPAuthorizationContext

authorized_notes = MCPServer(
    name="notes",
    transport=MCPStreamableHTTPTransport(url="https://notes.example.com/mcp"),
    authorization=provider,
)

shim = MCPToolsShim(
    servers=(authorized_notes,),
    authorization_context=MCPAuthorizationContext(key=user_id, value=user_id),
)
```

Use a stable, non-secret context `key` for each authorized user or connection.
Different keys keep connections and tool catalogs separate. The optional
`value` passes application data to your provider.

The provider is called repeatedly; reuse valid tokens until refresh is needed.
On HTTP `401`, AgentLane invalidates the token and retries once with a token
from the provider. AgentLane does not store refresh tokens.

## Reuse connections

For multiple agents or runs, share one application-scoped `MCPClientManager`:

```python
from agentlane.harness.mcp import MCPClientManager

async with MCPClientManager() as manager:
    shim = MCPToolsShim(servers=(notes,), client_manager=manager)
    # Attach this shim to agents and run them inside this context.
```

If your app manages shutdown explicitly, call `await manager.aclose()`.
Finish active runs before closing the manager.
For authorized servers, also pass the user's `authorization_context` to each shim.

Set `MCPClientManager(limits=MCPClientLimits(...))` to change the defaults:
64 connections, 300 seconds of idle time, and a 10-second shutdown timeout.
Both classes are exported from `agentlane.harness.mcp`.

## Configure limits

Set these options on `MCPServer`:

| Option | Default | Use |
| --- | --- | --- |
| `required` | `True` | Stop the run if discovery fails. Set `False` to continue without that server's tools. |
| `connect_timeout_seconds` | `30` | Limit connection setup and the MCP handshake. |
| `discovery_timeout_seconds` | `30` | Limit tool discovery. |
| `tool_timeout_seconds` | `120` | Limit each tool call. |
| `catalog_ttl_seconds` | `300` | Limit how long the tool list can stay cached. |

Timeouts and catalog TTL must be finite and greater than zero. Optional
servers retry transient discovery failures on later turns. AgentLane does
not automatically repeat a tool call after a transport failure or timeout.

To change output limits, set `MCPServer.result_policy=MCPResultPolicy(...)`.
Without result storage, the defaults are 32 content blocks and 51,200
characters. Servers repeat structured content in text blocks for clients
that cannot read it. If a successful result has structured content, the model
gets the structured content and the other content blocks, but not the text
blocks. A failed result
keeps its text blocks, because they contain the failure details. Set
`include_structured_content=False` to omit structured results and keep all
content blocks.
Image, audio, and resource bodies are returned as metadata.

## Store full results

Pass an existing `FileWriter` and a directory to `MCPToolsShim`. Give the
native [`read`](./read.md) tool the same filesystem:

```python
shim = MCPToolsShim(
    servers=(notes,),
    result_writer=filesystem,
    result_directory="/workspace/tool-results",
)
reader = read_tool(reader=filesystem)
shims = (shim, HarnessToolsShim((reader,)))
```

Import `HarnessToolsShim` and `read_tool` from `agentlane.harness.tools`, and
pass `shims` to `AgentDescriptor`. Use a directory that the filesystem can
resolve. See [File I/O interfaces](../filesystem.md) for mounts and backends.

With storage enabled, AgentLane uses its existing serialization and redaction
for all text blocks, structured content, and embedded resource text. If the
text fits `max_text_chars`, it returns inline. Otherwise, the same text is
saved to one UUID-named `.txt` file.

A saved result returns a preview and `[Full MCP result: "<path>"]`. The model
calls `read(path="<path>", offset=1, limit=200)` and follows the reader's next
offset. Text lines wrap at 4,096 characters so large strings remain readable.
The file contains serialized MCP text with JSON escaping; line wrapping means
it is not a stable JSON format. Image, audio, and blob bodies remain metadata.

The response, including its path, stays within `max_text_chars`. Storage uses
the existing write and cancellation handling. If a write fails or the path
cannot fit, the preview reports `[Full MCP result unavailable.]`. The provider
call is not repeated. The host owns write deadlines and file retention.

Configure both storage options together. With neither option set, the existing
response and processing limits are unchanged. With storage enabled, all text
is processed before budgeting; `max_content_blocks` and
`include_structured_content` do not apply.

## Troubleshooting

Use the symptom or error to choose the next step:

| Symptom or error | Action |
| --- | --- |
| `MCPDependencyError` | Install `agentlane[mcp]` in the environment that runs the agent. |
| Connection or discovery fails | Check the endpoint or local command, server availability, and credentials. Use `MCPDiscoveryError.failure_kind` to distinguish authorization, transport, timeout, capacity, and protocol failures. |
| A tool is missing | Check the server's tool list, `MCPToolFilter`, and agent tool policies. Filters use original names; agent policies use names such as `notes__list_meetings`. |
| `ToolNameCollisionError` | Give servers distinct names and check for duplicate tool names after normalization. |
| HTTP `401` or `403` | Check token validity and required scopes. Ensure invalidation lets the provider return a fresh token. |
| A tool call times out | Check whether the server completed the operation before retrying. Increase `tool_timeout_seconds` only if the operation needs more time. |
| `MCPPoolCapacityError` | Reduce concurrent runs or increase `MCPClientLimits.max_connections`. |
| `MCPShutdownTimeoutError` | Check for a server that does not stop. Increase `shutdown_timeout_seconds` if needed; call `aclose()` again to wait for ongoing cleanup. |
| Results contain `truncated` or `omittedBlocks` | Configure `result_writer` and `result_directory`, request less data, or increase the relevant `MCPResultPolicy` limit. `max_text_chars` must be at least 128. |

Tool execution failures return `ToolFailure`. Treat server results as
untrusted input, and keep secrets out of tool arguments and results.
