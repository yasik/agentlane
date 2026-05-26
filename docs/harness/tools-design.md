# Harness Tool Design

`agentlane.harness.tools` provides first-party harness tool definitions for
common local workspace actions. Each helper returns a `HarnessToolDefinition`,
which wraps an `agentlane.models.ToolSpec` plus optional prompt metadata for
`HarnessToolsShim`. Most definitions are executable `agentlane.models.Tool`
values. The `agent` definition is declarative and is executed by the harness
runner.

These tools are opinionated defaults for agent loops. They use stable argument
names, deterministic text results, visible truncation messages, `.gitignore`
handling where appropriate, and clear model-facing errors. That consistency
lets higher-level agents spend fewer turns rediscovering local shell behavior
and gives application code a predictable contract to test.

Use these tools with
[`DefaultAgent`](./default-agents.md) when you want the smallest local starting
point for a high-level agent. `DefaultAgent` owns the local runtime, runner, run
state, tool loop, and shim binding. `HarnessToolsShim` adds the tools and the
model guidance that tells the agent how to use them. As the application grows,
the same `AgentDescriptor`, `Tools`, shims, and native `Tool` values can move
down to the lower-level harness agent or runtime APIs.

## Import Path

```python
from agentlane.harness import INHERIT_TOOLS, OVERRIDE_TOOLS, RESTRICT_TOOLS
from agentlane.harness.tools import (
    AllOfToolPermissionPolicy,
    HarnessToolsShim,
    PathScopeToolPermissionPolicy,
    SideEffectApprovalToolPermissionPolicy,
    ToolPermissionGrant,
    ToolPermissionGrantPolicy,
    ToolPermissionPolicy,
    ToolOperation,
    ToolPermissionDecision,
    ToolPermissionRequest,
    WorkspaceToolPermissionPolicy,
    agent_tool,
    base_harness_tools,
    bash_tool,
    evaluate_tool_permission,
    find_tool,
    format_tool_permission_result,
    grep_tool,
    patch_tool,
    plan_tool,
    parse_tool_permission_grants,
    read_tool,
    workspace_tool_policy,
    write_tool,
)
```

## Tool Definitions

Tool helpers return definitions, not raw model tools:

```python
definition = write_tool(cwd=WORKSPACE)
tool = definition.tool
```

Use the definition when you want prompt snippets and guidelines to be rendered
by `HarnessToolsShim`. Use `definition.tool` when you need the underlying model
tool schema.

The current standard set is `read`, `find`, `grep`, `patch`, `write`,
`write_plan`, `bash`, and `agent`. The public base-tools set currently does not
include `ls`.

`base_harness_tools()` returns the standard set. By default each local tool
captures `Path.cwd()` at construction time and remains permissive. Pass `cwd=`,
`permissions=`, and optionally `approval_callback=` when an agent should
operate inside a specific workspace boundary:

```python
workspace_tools = base_harness_tools(
    cwd=WORKSPACE,
    permissions=WorkspaceToolPermissionPolicy(root=WORKSPACE),
)
```

You can also construct tools individually when an agent needs a custom tool
set:

```python
workspace_tools = (
    read_tool(cwd=WORKSPACE),
    find_tool(cwd=WORKSPACE),
    grep_tool(cwd=WORKSPACE),
    patch_tool(cwd=WORKSPACE),
    write_tool(cwd=WORKSPACE),
    plan_tool(),
    bash_tool(cwd=WORKSPACE),
    agent_tool(),
)
```

## HarnessToolsShim

`HarnessToolsShim` merges tool schemas into each prepared turn and appends the
definitions' prompt metadata to the first turn's system instructions:

```python
from agentlane.harness import AgentDescriptor
from agentlane.harness.tools import (
    HarnessToolsShim,
    agent_tool,
    find_tool,
    grep_tool,
    patch_tool,
    bash_tool,
    read_tool,
    write_tool,
)
from agentlane.models import Tools

descriptor = AgentDescriptor(
    name="Workspace Agent",
    model=model,
    instructions="Use workspace tools before answering workspace questions.",
    tools=Tools(
        tools=[],
        tool_call_limits={
            "find": 1,
            "grep": 1,
            "patch": 1,
            "read": 1,
            "write": 1,
            "bash": 1,
        },
    ),
    shims=(
        HarnessToolsShim(
            (
                read_tool(cwd=WORKSPACE),
                find_tool(cwd=WORKSPACE),
                grep_tool(cwd=WORKSPACE),
                patch_tool(cwd=WORKSPACE),
                write_tool(cwd=WORKSPACE),
                bash_tool(cwd=WORKSPACE),
                agent_tool(),
            )
        ),
    ),
)
```

For quick prototypes that should use the process working directory, pass the
standard set directly:

```python
descriptor = AgentDescriptor(
    name="Workspace Agent",
    model=model,
    instructions="Use workspace tools before answering workspace questions.",
    shims=(HarnessToolsShim(base_harness_tools()),),
)
```

## Output Limits

Text output is capped at shared deterministic limits. `read` output is capped
at 2000 lines or 51200 bytes, whichever limit is reached first. `find` output
is capped at 1000 matching paths or 51200 bytes, whichever limit is reached
first. `grep` output is capped at 100 matching entries or 51200 bytes,
whichever limit is reached first. `patch` success output is intentionally
minimal and does not need truncation. `bash` output is tail-truncated to the
most recent 2000 combined stdout/stderr lines or 51200 bytes.

Caller-provided limits are applied before the global caps. For large files, call
`read` repeatedly with `offset` and `limit`. For large search results, narrow
the `find` pattern or search path.
