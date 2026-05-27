# agent Tool

`agent_tool()` exposes an `agent` tool for generic spawned helpers.

Parameters:

1. `name: str`
2. `task: str`

## Permissions

`agent` does not issue a `ToolPermissionRequest` for the spawn itself. Spawned
helpers inherit the configured base tools and their `permissions=` /
`approval_callback=` through normal shim inheritance, so filesystem and bash
checks still happen inside the child when it uses those tools.

`name` must be one word. It can be task-relevant or random, and is used only
for logging and tracing. `task` is the full instruction for the helper,
including the context it needs and the expected output.

Example tool call:

```json
{
  "name": "Researcher",
  "task": "Review the refund exception policy and return the two most relevant constraints."
}
```

`agent` is agent-as-tool, not handoff. The caller waits for the helper result
and then continues its own loop. The spawned helper treats the explicit `task`
as its assigned work, not the generated `name`. Generic spawned helpers do not
inherit the parent's system prompt or conversation history.

Tool visibility has two inheritance paths that are easy to conflate:

1. Direct descriptor tools are resolved through `AgentDescriptor.tools` and
   `ToolConfig`. These are the tools inherited, replaced, or filtered by
   `INHERIT_TOOLS`, `OVERRIDE_TOOLS`, and `RESTRICT_TOOLS.only(...)`.
2. Shim-contributed tools are added later by inherited descriptor shims during
   `prepare_turn(...)`. When the parent exposes
   `base_harness_tools(cwd=..., permissions=..., approval_callback=...)`
   through `HarnessToolsShim`, spawned helpers get those same configured base
   tools and prompt guidance through this shim path.

Inherited direct tools and shim-contributed tools are merged by tool name so
duplicate definitions are exposed only once. Predefined handoffs are separate:
the model sees them as schemas, but the runner handles them as control
transfers rather than normal agent-as-tool calls.

Tool inheritance is controlled by the same `ToolConfig` policy used by
`AgentDescriptor.tools`:

1. `INHERIT_TOOLS` inherits parent tools and merges child-local additions.
2. `OVERRIDE_TOOLS` ignores parent tools; with no explicit tools it exposes no
   direct tools.
3. `RESTRICT_TOOLS.only(...)` filters inherited parent tools by name and then
   merges child-local additions.
4. Bare `Tools(...)` and `None` are compatibility shorthands for override
   behavior.

`agent` supports parallel calls when the parent `Tools` configuration enables
`parallel_tool_calls`. Recursive spawning is bounded by process-local
`Runner` safety limits: `Runner(agent_max_depth=4, agent_max_threads=16)`.
`agent_max_depth` is inclusive: a direct child is depth 1, and the default
allows spawned agents through depth 4. These limits are execution policy, not
part of the tool schema. When the depth limit is reached, the tool result is:

```text
Agent depth limit reached. Solve the task yourself.
```

When the live-agent thread limit is reached, the tool result is:

```text
Agent thread limit reached. Solve the task yourself.
```
