# Harness Tools

`agentlane.harness.tools` provides first-party harness tool definitions for
common local workspace actions. Each helper returns a `HarnessToolDefinition`,
which wraps an executable `agentlane.models.Tool` or runner-owned `ToolSpec`
plus optional prompt metadata for `HarnessToolsShim`.

## Index

Core docs:

1. [Tool design](./tools-design.md): construction model, imports,
   `HarnessToolsShim`, base tool sets, and shared output limits.
2. [Tool permissions](./tools-permissions.md): permissive defaults, path
   policy, bundled permission policies, approval callbacks and brokers, and
   `bash` boundaries.

Tool reference:

1. [agent](./tools-agent.md): delegate focused work to a fresh helper agent.
2. [read](./tools-read.md): read UTF-8 text files with deterministic truncation.
3. [find](./tools-find.md): walk directories with gitignore-aware pattern
   matching.
4. [grep](./tools-grep.md): search file contents with ripgrep-backed results.
5. [patch](./tools-patch.md): apply minimal SEARCH/REPLACE edits.
6. [write](./tools-write.md): create or overwrite UTF-8 files.
7. [write_plan](./tools-write-plan.md): update the visible task plan.
8. [bash](./tools-bash.md): run bounded non-interactive shell commands.

## Quick Start

For common local-agent apps, use `WorkspaceToolsShim` so the standard tools
share one workspace root and the model sees workspace-relative path guidance.

```python
from pathlib import Path

from agentlane.harness.tools import WorkspaceToolsShim

WORKSPACE = Path.cwd()

shims = (WorkspaceToolsShim(WORKSPACE),)
```

For a trusted local prototype that needs direct control over the tool set, use
the standard tools directly. A specific `cwd` controls relative path
resolution; it is not a sandbox.

```python
from pathlib import Path

from agentlane.harness.tools import HarnessToolsShim, base_harness_tools

WORKSPACE = Path.cwd()

shims = (
    HarnessToolsShim(
        base_harness_tools(cwd=WORKSPACE),
    ),
)
```

For a workspace-bounded coding assistant, pass an explicit permission policy:

```python
from agentlane.harness.tools import (
    WorkspaceToolsShim,
    WorkspaceToolPermissionPolicy,
)

shims = (
    WorkspaceToolsShim(
        WORKSPACE,
        permissions=WorkspaceToolPermissionPolicy(WORKSPACE),
    ),
)
```

For approval workflows, broader path scopes, operation grants, and `bash`
behavior, see [Tool permissions](./tools-permissions.md).

## Standard Set

`base_harness_tools()` returns `read`, `find`, `grep`, `patch`, `write`,
`write_plan`, `bash`, and `agent`. The public base-tools set currently does
not include `ls`.

Pass `include=` or `exclude=` when an application needs the standard set with
only selected tools. Selectors use the model-visible tool names above, and
returned tools keep the standard order:

```python
search_tools = base_harness_tools(
    cwd=WORKSPACE,
    include=("read", "find", "grep"),
)

no_shell_tools = base_harness_tools(
    cwd=WORKSPACE,
    exclude=("bash",),
)
```

Unknown selector names raise `ValueError`. Passing the same name in both
`include` and `exclude` also raises `ValueError`.

You can also construct tools individually when an agent needs a custom tool set:

```python
from agentlane.harness.tools import (
    agent_tool,
    bash_tool,
    find_tool,
    grep_tool,
    patch_tool,
    plan_tool,
    read_tool,
    write_tool,
)

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
