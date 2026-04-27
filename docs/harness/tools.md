# Harness Tools

`agentlane.harness.tools` provides first-party harness tool definitions for
common local workspace actions. Each helper returns a
`HarnessToolDefinition`, which wraps an executable `agentlane.models.Tool` plus
optional prompt metadata for `HarnessToolsShim`.

## Import Path

```python
from agentlane.harness.tools import HarnessToolsShim, read_tool
```

## Tool Definitions

Tool helpers return definitions, not raw model tools:

```python
definition = read_tool(cwd=WORKSPACE)
tool = definition.tool
```

Use the definition when you want prompt snippets and guidelines to be rendered
by `HarnessToolsShim`. Use `definition.tool` when you only need the executable
`Tool` value.

## HarnessToolsShim

`HarnessToolsShim` merges executable tools into each prepared turn and appends
the definitions' prompt metadata to the first turn's system instructions:

```python
from agentlane.harness import AgentDescriptor
from agentlane.harness.tools import HarnessToolsShim, read_tool
from agentlane.models import Tools

descriptor = AgentDescriptor(
    name="Workspace Reader",
    model=model,
    instructions="Read files before answering workspace questions.",
    tools=Tools(tools=[], tool_choice="required", tool_call_limits={"read": 1}),
    shims=(HarnessToolsShim((read_tool(cwd=WORKSPACE),)),),
)
```

`base_harness_tools()` returns the current standard tool set. It contains
`read`.

## Path Policy

Filesystem tools use `ToolPathResolver`. Relative paths resolve against the
`cwd` captured when the tool is constructed. Absolute paths are allowed in
the current implementation. Paths are normalized with
`Path.resolve(strict=False)`.

The first-party path resolver does not enforce a sandbox boundary, permission
allowlist, or approval workflow.

## Output Limits

Text output is capped at 2000 lines or 51200 bytes, whichever limit is reached first.
Tool results include the resolved absolute path followed by 1-indexed,
line-numbered rows.

Caller-provided limits are applied before the global caps. For large files, call
`read` repeatedly with `offset` and `limit`.

## read

`read_tool()` exposes a `read` tool for UTF-8 text files.

Parameters:

1. `path: str`
2. `offset: int | None = None`
3. `limit: int | None = None`

Example tool result:

```text
Absolute path: /workspace/notes.txt
L1: alpha
L2: bravo
L3: charlie
```

When more lines remain after a caller limit or global line cap, the result adds
a continuation note:

```text
More than 2000 lines found
```

When the byte cap is reached, the result reports:

```text
Output truncated after 51200 bytes
```

The tool returns clear text errors for directories, missing files, likely binary
files, invalid offsets, invalid limits, and unreadable paths. Invalid UTF-8 byte
sequences are decoded with replacement characters so the model can still use
the surrounding text.
