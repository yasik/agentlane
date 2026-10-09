# Harness MCP Tools

`MCPToolsShim` discovers tools from Model Context Protocol (MCP) servers and
adds them to an agent as native AgentLane tools. It supports Streamable HTTP
and stdio. Install the optional dependency:

```bash
uv add 'agentlane[mcp]'
```

The extra requires MCP SDK 2.2 or later within major version 2. Output-schema
validation supports local references and does not fetch external schemas.

## Connect an agent

Attach the shim through `AgentDescriptor.shims`. This example uses an HTTPS
server and a configured AgentLane `model`:

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

mcp = MCPToolsShim(servers=(notes,))

agent = DefaultAgent(
    descriptor=AgentDescriptor(
        name="Notes assistant",
        model=model,
        shims=(mcp,),
    ),
)

result = await agent.run("Find the latest launch decision in the notes.")
print(result.final_output)
```

The shim opens and closes its connections for each run.

For a runnable example with a local server, see the
[MCP Meeting Assistant](../../examples/harness/mcp_meeting_assistant/README.md).

## Tool names and context

The model receives a flat list of tool definitions named `<server>__<tool>`.
For example, `list_meetings` on the `notes` server becomes
`notes__list_meetings`. The catalog is not added to the system prompt.

Names are converted to lowercase, unsupported character runs become `_`, and
leading or trailing underscores are removed. A name that starts with a digit
gets an `mcp_` prefix. Long combined names receive a stable hash suffix to fit
the 64-character limit. Calls to the MCP server use the original names.

Duplicate names in MCP catalogs fail discovery. A collision between an MCP
tool and another built-in tool contribution raises `ToolNameCollisionError`
from `agentlane.harness.shims` before the model call, in either shim order.
Custom shims must contribute tools through `PreparedTurn.add_tools(...)` to
participate in this check.

Use `MCPToolFilter` to limit which tools enter the model context. Its include
and exclude patterns match original MCP tool names:

```python
from agentlane.harness.mcp import MCPToolFilter

notes = MCPServer(
    name="notes",
    transport=MCPStreamableHTTPTransport(url="https://notes.example.com/mcp"),
    tools=MCPToolFilter(include=("list_*", "get_*"), exclude=("*_transcript",)),
)
```

MCP tools preserve the server's input schema with `strict=False`. They set
`retry_on_timeout=False` to prevent automatic repeats after a tool timeout.
See [model tool policy](../models/overview.md#schema-and-timeout-policy).

## Authorization

The application owns OAuth consent, credential storage, token refresh, and
revocation. AgentLane receives only access tokens through an
`MCPAuthorizationProvider` with two async methods:

- `get_access_token(server, context)` returns an `MCPAccessToken` containing
  `token`, optional `expires_at`, and optional `scopes`.
- `invalidate_access_token(server, context, token)` invalidates a rejected
  token in the application's credential service.

Attach your provider to the HTTP server and pass the user's identity to the
shim:

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

The context `key` must be a stable, non-secret string that identifies the
authorized user or connection. Different keys isolate connections and cached
catalogs. The optional `value` is opaque application data passed to the
provider. Both the provider and context value stay in memory.

Each run lease retains its full authorization context. Discovery, tool calls,
and their `401` invalidation and retry use that lease's context, including
when another lease shares the same key with a different value. Connection
startup, the background tool-change listener, and HTTP session termination
use the full context of the lease that opened the current connection. A new
connection records the context of its opener again.

Before each HTTP request and catalog check, AgentLane asks for a current token.
The provider should reuse a valid token until refresh is needed. A changed token
or set of scopes invalidates the cached catalog. On `401`, AgentLane invalidates
that token, requests another, and retries once. Provider errors, a final `401`,
and `403` are authorization failures. A rejected token, `403`, or provider failure
also invalidates existing catalogs, even if a later lookup returns the same
token. AgentLane does not request or store refresh tokens.

## Connection lifecycle and discovery

Use one application-scoped `MCPClientManager` for reuse across agents and runs.
Pass it as `client_manager` to each shim and close it with `aclose()` at
application shutdown, or use `async with MCPClientManager()`.
Connections are shared only for matching server settings, provider identity,
and authorization context key. Without a shared manager, the shim creates and
closes its own manager for each run.

Configure pool limits with `MCPClientLimits`. These are the defaults:

```python
from agentlane.harness.mcp import MCPClientLimits, MCPClientManager

limits = MCPClientLimits(
    max_connections=64,
    idle_timeout_seconds=300,
    shutdown_timeout_seconds=10,
)

async with MCPClientManager(limits=limits) as manager:
    shim = MCPToolsShim(servers=(notes,), client_manager=manager)
    # Create and run agents inside this context.
```

Opening and closing connections count toward the pool limit. The manager
closes idle connections after the idle timeout. When the pool is full, it
first closes the least recently used idle connection. It does not evict a
connection with a run lease or work in progress. If no connection can be
evicted, acquisition raises `MCPPoolCapacityError` from
`agentlane.harness.mcp`. All eviction waits in one acquisition share one
`shutdown_timeout_seconds` deadline. If cleanup cannot free capacity before
that deadline, acquisition raises `MCPPoolCapacityError`; cleanup continues
and keeps its capacity reserved until it finishes. Connection acquisition
and leases are internal; applications configure the manager and pass it to
a shim.

The shim checks catalogs at startup and before each model turn. A fresh catalog
needs no `tools/list` request. The `max_concurrent_discoveries` parameter on
`MCPToolsShim` limits concurrent server checks to 8 by default. This limit
covers connection setup and catalog discovery, and does not limit tool execution.

With MCP 2026-07-28, each page's `ttlMs` bounds its
lifetime from receipt; the earliest page expiry applies to the complete catalog.
Explicit `ttlMs=0` requires refresh on the next check. `catalog_ttl_seconds`
caps that lifetime and supplies the fallback when hints are absent or the server
uses an older protocol. Server TTL hints are also capped at 24 hours.
Catalogs stay private to the authorization context, including those marked
`cacheScope="public"` by the server.

Each run receives separate copies of the tool input schemas, including nested
objects and lists. Changes to a run's schemas do not change the cached catalog
or another run's schemas.

A tool-list change notification, expiry, or authorization change refreshes the
tools on the next check, including additions and removals. Confirmation of a
notification subscription also invalidates catalogs whose discovery started
before that confirmation. If returned pages use different credentials,
discovery restarts once within the same timeout.
Unstable authorization fails discovery without publishing a mixed catalog.
Authorization is checked again before the first HTTP tool request. Credentials
must still match discovery before that request is sent. The single refresh and
retry after an explicit `401` remains supported.

An active run can use its own last successful catalog after a transient
transport or timeout failure. New runs cannot use a stale catalog from another
run. Calls from a retained catalog use the current connection. Authorization
changes or failures, schema errors, and protocol errors do not allow this
fallback.

Servers are required by default: a connection or discovery failure stops the
run before the next model call. Set `required=False` to let the run continue
without tools from an unavailable server. Optional servers retry transient
connection, timeout, and capacity failures on a later turn preparation, even
if the first connection attempt failed. The retry delays are 1, 2, 4, 8, 16,
then 30 seconds between attempts. A successful check resets the delay.
There are no background retry attempts. Authorization, configuration, and
protocol failures disable retries for that server for the current run.
Failures during tool execution return `ToolFailure`.

A lost connection is reopened at the next catalog check. If a notification
subscription ends normally or loses its connection, the next catalog check
reopens the connection, fetches the catalog, and restores the subscription.
AgentLane does not replay a tool call after a transport failure or timeout.
Manager shutdown cancels active work and closes connections concurrently.
The configured shutdown timeout bounds each `aclose()` wait. If cleanup is
still in progress, `aclose()` raises `MCPShutdownTimeoutError` from
`agentlane.harness.mcp`.
Cleanup continues in owned background tasks; another `aclose()` call waits
for the same cleanup with a new timeout. `manager.closed` means the manager
rejects new work. It does not mean all transports have finished closing.

At run cleanup, a shim with its own manager calls `aclose()` directly. A shim
with a shared manager releases its run leases concurrently and leaves the
manager open. If the last lease closes a failed or incomplete connection,
`shutdown_timeout_seconds` also bounds that release wait. A timeout raises
`MCPShutdownTimeoutError` while the connection cleanup continues.

The harness reports cleanup failures in an exception group. If connection
setup or the run has already failed or was cancelled, cleanup failures do not
replace that original error or cancellation.

The connection cleanup deadline also covers credential lookup during HTTP
session termination. Stdio cleanup can continue beyond that deadline while
the SDK completes its bounded graceful-exit and forced-kill stages. Closing
connections keep their capacity reserved until cleanup finishes.

## Timeouts

Set separate connection, discovery, and execution limits on `MCPServer`:

```python
server = MCPServer(
    name="notes",
    transport=MCPStreamableHTTPTransport(url="https://notes.example.com/mcp"),
    connect_timeout_seconds=30,
    discovery_timeout_seconds=30,
    tool_timeout_seconds=120,
    catalog_ttl_seconds=300,
)
```

These are the defaults, in seconds. The connection limit includes the MCP
handshake for both transports. `MCPStreamableHTTPTransport` also has
`connect_timeout_seconds=30` and `read_timeout_seconds=300` for individual HTTP
operations. These configured timeout and TTL values must be finite and greater
than zero.

`discovery_timeout_seconds` covers waiting for another catalog discovery,
fetching all pages, and any discovery restart. The initial credential check has
its own limit of `discovery_timeout_seconds`. Connection startup uses the
separate `connect_timeout_seconds` limit.

## Inheritance and tool policies

`INHERIT_TOOLS`, `RESTRICT_TOOLS`, `OVERRIDE_TOOLS`, and `ExcludeToolsShim` use
the model-visible names, such as `notes__list_meetings`. Tool-call and
round-trip limits apply after all shims contribute tools.

Subagents and handoffs bind inherited MCP tools independently, limited to the
names allowed by the parent's policy. Child-local tools remain available.
Only servers that supply inherited tool names are connected in the child.
Name collisions between inherited and child-local tools raise an error; use
`OVERRIDE_TOOLS` for intentional replacement. Each child has its own connection
lease, so ending one run does not close another run's connection.

When an MCP source is wrapped, the child binds the original wrapper chain
again and gets fresh wrapper state. The restricted source binding selects
the permitted servers and tool names before it creates the child session.
The original MCP configuration cannot broaden that inherited set. Custom
wrappers must pass the complete `ShimBindingContext` to the inner `bind(...)`
method. See [dynamic source inheritance](./shims.md#advanced-bound-sessions)
for the `ToolSourceBinding` contract.

Place `MCPToolsShim` before `SkillsShim` when skill `tools` or
`disallowed-tools` rules must cover MCP tools. `ExcludeToolsShim` works in
either order. See [shims](./shims.md) for preparation and inheritance callbacks.

## Results and data handling

Rendered server results contain text and structured content as JSON. Each
rendered server result includes the server's `isError` flag, including truncated
results. `MCPResultPolicy` defaults to 32 content blocks and 51,200 output
characters. Set `MCPServer.result_policy` to change these limits or disable
structured content with
`include_structured_content=False`. `max_text_chars` must be at least 128.
Truncated results remain valid JSON and include `truncated` and `omittedBlocks`.
They do not report an omitted-character count. Image, audio, and resource bodies
are replaced with metadata.

Rendering inspects a bounded portion of the decoded result. Strings that exceed
the inspection budget are replaced with an omission marker. An oversized key
causes its entire entry to be omitted. Retained strings are redacted before
output truncation. Text that starts with `{` or `[` and cannot be safely parsed
or rendered as JSON is replaced with an omission marker. These rendering limits
do not bound SDK response decoding or schema validation.

AgentLane redacts known provider token values and credential-shaped fields
from results and framework errors. Managed SDK and transport logs are
suppressed; AgentLane's own MCP logs contain operation metadata. Tool data is
included in tracing only when the tracing policy permits it. Treat server text
as untrusted input.

Tokens, providers, HTTP clients, and MCP sessions are kept out of prompts,
`RunState`, snapshots, and run events.

## Transport configuration

Remote servers require HTTPS. Set `allow_insecure_http=True` for development
HTTP. Redirects may stay on the same origin or upgrade from HTTP to HTTPS on
the same host using default ports. URLs must not contain credentials in
userinfo, known token or OAuth callback query parameters, or fragments.

For a local server process, use `MCPStdioTransport`:

```python
from agentlane.harness.mcp import MCPStdioTransport

local_server = MCPServer(
    name="local-notes",
    transport=MCPStdioTransport(
        command="python",
        args=("-m", "my_notes_mcp"),
        cwd="/srv/my-product",
        env={"NOTES_DATABASE": "/data/notes.db"},
    ),
)
```

Stdio inherits the MCP SDK's safe environment allow-list plus explicit `env`
values. Pass only the environment variables the server needs.

Legacy SSE and MCP resources or prompts as model capabilities are not supported.
