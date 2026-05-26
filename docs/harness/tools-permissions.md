# Harness Tool Permissions

## Path Policy

Filesystem tools use `ToolPathResolver`. Relative paths resolve against the
`cwd` captured when the tool is constructed. Absolute paths are allowed in
the current implementation. Paths are normalized with
`Path.resolve(strict=False)`.

AgentLane is a framework, so first-party helpers stay permissive unless an
application passes an explicit policy. With no `permissions=` argument,
`read`, `find`, `grep`, `patch`, `write`, and `bash` keep their trusted local
behavior. A specific `cwd` only changes path resolution; it is not a sandbox:

```python
tools = base_harness_tools(cwd=WORKSPACE)
```

Applications opt into permissioning at tool construction time. The bundled
policies are small and composable:

1. `AllowAllToolPermissionPolicy` allows every request. This is the implicit
   default when no `permissions=` value is passed.
2. `WorkspaceToolPermissionPolicy` is a single-root path boundary. It allows
   path operations only when the resolved target stays inside `root`.
3. `PathScopeToolPermissionPolicy` is an explicit set of approved files or
   directories. Use it when a coding assistant works from one `cwd` but the
   host has approved files or directories outside that workspace.
4. `ToolPermissionGrantPolicy` is a tool and operation allowlist. It does not
   sandbox paths and does not ask for approval.
5. `SideEffectApprovalToolPermissionPolicy` returns `require_approval` for
   writes, patches, directory creation, and command execution. The host
   application still owns the approval callback and UI.
6. `AllOfToolPermissionPolicy` composes policies conservatively: deny wins,
   then approval, then allow.

Policy defaults are intentionally explicit:

1. `allowed_operations=None` on path policies allows all path operations and
   denies command execution.
2. `ToolPermissionGrantPolicy(())` has an empty allowlist and denies every
   request it checks.
3. `AllOfToolPermissionPolicy(())` allows because no nested policy denies; use
   it only when an empty composition is intentional.

For a path-only workspace boundary, pass `WorkspaceToolPermissionPolicy`:

```python
tools = base_harness_tools(
    cwd=WORKSPACE,
    permissions=WorkspaceToolPermissionPolicy(root=WORKSPACE),
)
```

This policy only answers "does the resolved path stay inside this root?" It is
not an allowlist, approval workflow, or process sandbox.

For workspace plus approved outside files or directories, use
`PathScopeToolPermissionPolicy` and include every approved scope explicitly:

```python
tools = base_harness_tools(
    cwd=WORKSPACE,
    permissions=PathScopeToolPermissionPolicy(
        paths=(
            WORKSPACE,
            EXTERNAL_REVIEW_FILE,
            EXTERNAL_REFERENCE_DIR,
        ),
    ),
)
```

Existing directory scopes allow descendants. Existing file scopes allow only
that exact file. Non-existing scopes allow only that exact future path, which
keeps a file grant from silently becoming a directory grant. Empty `paths=()`
denies all path operations. Prefer absolute paths for policy scopes; relative
scope entries resolve when the policy is constructed, the same as
`WorkspaceToolPermissionPolicy(root=...)`.

For the common application policy "stay inside this workspace, apply a tool
grant allowlist, and require app approval before side effects", use
`workspace_tool_policy(...)`. It is a convenience constructor over the same
public primitives, not a separate policy system:

```python
grants, invalid_entries = parse_tool_permission_grants(
    "read, find, grep, write:create_file, patch:modify_file, "
    "bash:execute_command"
)
if invalid_entries:
    raise ValueError(f"Unsupported tool grants: {invalid_entries}")

tools = base_harness_tools(
    cwd=WORKSPACE,
    permissions=workspace_tool_policy(
        root=WORKSPACE,
        grants=grants,
        require_approval_for_side_effects=True,
        require_bash_approval=True,
    ),
    approval_callback=approve,
)
```

Use `grants=None` when you do not want a grant allowlist. Pass an empty
iterable, `grants=()`, only when the grant layer should deny every tool.
`require_approval_for_side_effects=True` makes file creation, overwrites,
patches, and directory creation require approval. `bash` is separate:
`require_bash_approval=True` is the only way this helper admits
`bash:execute_command`, and it always makes command execution require approval
before the process starts. If grants are configured, `bash:execute_command`
must still be granted. This does not sandbox the command after startup.

The helper composes the public low-level policies in this order:

1. `WorkspaceToolPermissionPolicy` denies path operations outside the root and
   denies `bash` unless `require_bash_approval=True`.
2. `ToolPermissionGrantPolicy`, when provided, denies requests that do not
   match a whole-tool or operation-level grant.
3. `SideEffectApprovalToolPermissionPolicy` returns `require_approval` for
   file side effects when `require_approval_for_side_effects=True`, and for
   command execution when `require_bash_approval=True`.
4. `AllOfToolPermissionPolicy` combines those decisions conservatively: deny
   wins, then approval, then allow.

For the same grant and approval behavior with approved outside paths, compose
the public policies directly:

```python
tools = base_harness_tools(
    cwd=WORKSPACE,
    permissions=AllOfToolPermissionPolicy(
        (
            PathScopeToolPermissionPolicy(
                paths=(WORKSPACE, EXTERNAL_REVIEW_FILE),
            ),
            ToolPermissionGrantPolicy(grants),
            SideEffectApprovalToolPermissionPolicy(),
        )
    ),
    approval_callback=approve,
)
```

That composition still denies `bash` by default because path scopes are not
process sandboxes. To admit command execution into the policy pipeline, pass an
explicit `allowed_operations` set that includes `ToolOperation.EXECUTE_COMMAND`
and any path operations the app also wants to allow.

The exported permission primitives are framework extension points:

1. `ToolPermissionRequest` describes one operation before a side effect.
2. `ToolPermissionDecision` returns `allow`, `deny`, or `require_approval`.
3. `WorkspaceToolPermissionPolicy` is the built-in single-root path policy.
4. `PathScopeToolPermissionPolicy` is the built-in multi-scope path policy.
5. `AllOfToolPermissionPolicy` composes policies conservatively: deny wins,
   then approval, then allow.
6. `ToolPermissionGrantPolicy` and `parse_tool_permission_grants()` are small
   helpers for operation-level allowlists.
7. `SideEffectApprovalToolPermissionPolicy` is the generic approval policy for
   side-effecting operations.
8. `evaluate_tool_permission()` and `format_tool_permission_result()` are
   reusable helpers for custom tools that want the same deny and
   approval-required result contract.

Custom policies only need a `check(request)` method:

```python
class DenyBash:
    def check(self, request: ToolPermissionRequest) -> ToolPermissionDecision:
        if request.operation == ToolOperation.EXECUTE_COMMAND:
            return ToolPermissionDecision.deny()
        return ToolPermissionDecision.allow()
```

`WorkspaceToolPermissionPolicy(root=...)` resolves existing paths through
symlinks. New paths are checked through their nearest existing parent, so
writing `nested/file.txt` still requires the target parent chain to stay inside
the workspace. Absolute paths are still accepted, but a workspace policy denies
absolute targets outside the root.

If an app needs a different scope shape, use one of these explicit choices:

1. Use a broader `WorkspaceToolPermissionPolicy(root=...)` while keeping
   `cwd=WORKSPACE` narrow for relative path resolution.
2. Use `PathScopeToolPermissionPolicy(paths=...)` for workspace plus approved
   outside files or directories.
3. Pass no policy for a fully trusted local tool set.
4. Implement `ToolPermissionPolicy.check(...)` for app-specific rules such as
   per-user grants, read-only subtrees, or remote policy decisions.

Denied calls return stable tool-result text before side effects:

```text
permission denied: read is not allowed for `/workspace/private.txt`
```

Policies may also return `require_approval`. The core harness does not provide
interactive approval UI. Instead, tools accept an optional `approval_callback`
that an application, CLI, desktop app, or service can use to decide whether
the pending `ToolPermissionRequest` should proceed:

```python
async def approve(
    request: ToolPermissionRequest,
    decision: ToolPermissionDecision,
) -> ToolPermissionDecision:
    return ToolPermissionDecision.allow()
```

If no callback is configured, the tool returns a stable approval-required
result and does not execute.

When a first-party tool runs through the default runner, permission requests
also receive framework correlation from an explicit
`agentlane.models.ToolExecutionContext`. The runner builds that context for
each model tool call, `ToolExecutor` passes it to the tool handler, and the
first-party permission helper copies these fields onto `ToolPermissionRequest`
before policy and approval evaluation:

1. `run_id`
2. `agent_name`
3. `tool_call_id`
4. `metadata` for application-defined correlation

The framework does not render correlation metadata back to the model by
default. Applications can use it in policies, approval callbacks, audit logs,
or UI prompts. There is no hidden ambient permission context; custom tools that
need framework correlation should accept the `ToolExecutionContext` passed by
`Tool.run(...)`.

Operations are intentionally small and tool-oriented:

| Tool | Operations |
| --- | --- |
| `read` | `read_file` |
| `find` | `search_files` |
| `grep` | `read_file` for file paths, `search_files` for directories |
| `write` | `create_file`, `overwrite_file`, `create_directory` |
| `patch` | `modify_file` |
| `bash` | `execute_command` |

The shared policy can require approval for local `bash` before the process
starts, but the default local executor is not filesystem-confined after
startup. Because `WorkspaceToolPermissionPolicy` is a path sandbox, it denies
`execute_command` unless `ToolOperation.EXECUTE_COMMAND` is explicitly
included in `allowed_operations` or another host policy allows it.
Applications that need real process isolation should provide a sandboxed
`BashExecutor`, container, remote worker, or equivalent host boundary.
