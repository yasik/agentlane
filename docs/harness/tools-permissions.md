# Harness Tool Permissions

## Path Policy

Filesystem tools use `ToolPathResolver`. Relative paths resolve against the
`cwd` captured when the tool is constructed. Absolute path strings are accepted
as tool inputs, but workspace and path-scope policies still enforce boundaries
on the resolved target. Paths are normalized with `Path.resolve(strict=False)`.

AgentLane is a framework, so first-party helpers stay permissive unless an
application passes an explicit policy. With no `permissions=` argument,
`read`, `find`, `grep`, `patch`, `write`, and `bash` keep their trusted local
behavior. A specific `cwd` only changes path resolution; it is not a sandbox:

```python
tools = base_harness_tools(cwd=WORKSPACE)
```

Applications opt into permissioning at tool construction time. The bundled
policies are small and composable:

1. `permissions=None` allows every request. This is the canonical allow-all
   path for trusted app code and prototypes; there is no separate public
   allow-all policy to compose.
2. `WorkspaceToolPermissionPolicy` is a single-root path boundary. It allows
   path operations only when the resolved target stays inside `root`. It
   denies `ToolOperation.EXECUTE_COMMAND` unless that operation is explicitly
   included in `allowed_operations`, because a path boundary cannot prove shell
   side effects.
3. `PathScopeToolPermissionPolicy` is an explicit set of approved files or
   directories. Use it when a coding assistant works from one `cwd` but the
   host has approved files or directories outside that workspace.
4. `ToolPermissionGrantPolicy` is a tool and operation allowlist. It does not
   sandbox paths and does not ask for approval.
5. `SideEffectApprovalToolPermissionPolicy` returns `require_approval` for
   writes, patches, directory creation, command execution, and network access.
   The host application still owns the approval callback and UI.
6. `AllOfToolPermissionPolicy` composes policies conservatively: deny wins,
   then approval, then allow.

Policy defaults are intentionally explicit:

1. `allowed_operations=None` on path policies allows all path operations and
   denies command execution.
2. `ToolPermissionGrantPolicy(())` has an empty allowlist and denies every
   request it checks.
3. `AllOfToolPermissionPolicy(())` allows because no nested policy denies; use
   it only when an empty composition is intentional.

## Composition semantics (strictest wins)

`AllOfToolPermissionPolicy` resolves nested decisions on the ordering
`allow < require_approval < deny`. It scans every nested policy: a `deny` is
terminal and returned immediately; a `require_approval` is remembered; an
`allow` never widens a decision a stricter policy already made. The final
outcome is the strictest decision any nested policy returned.

This has one trap worth stating plainly. `SideEffectApprovalToolPermissionPolicy`
returns `require_approval` for every side-effecting operation it covers. If you
also compose a `ToolPermissionGrantPolicy` (or `workspace_tool_policy(grants=...)`)
to pre-grant `bash:execute_command`, that grant is **outcome-inert** by default:
the grant policy returns `allow`, but the side-effect policy still returns
`require_approval`, and strictest-wins keeps the approval. The configured grant
looks like it permits the operation but never actually skips approval.

Two supported ways to make a side-effect grant meaningful:

1. Pass the grants to the side-effect policy so a matching grant downgrades
   `require_approval` to `allow` for the operations it covers:

   ```python
   AllOfToolPermissionPolicy(
       (
           WorkspaceToolPermissionPolicy(WORKSPACE, allowed_operations=...),
           ToolPermissionGrantPolicy(grants),
           SideEffectApprovalToolPermissionPolicy(grants=grants),
       )
   )
   ```

2. With the `workspace_tool_policy(...)` helper, opt in with
   `grants_downgrade_side_effect_approval=True`:

   ```python
   workspace_tool_policy(
       WORKSPACE,
       grants=grants,
       require_bash_approval=True,
       grants_downgrade_side_effect_approval=True,
   )
   ```

   With the flag set, a request matching a grant returns `allow` from the
   side-effect policy, so `bash:execute_command` actually skips approval.
   Without it (the default), the grant stays inert and `bash` still requires
   approval — the conservative behavior is the default so opting into trust is
   always explicit. `grants=None` on `SideEffectApprovalToolPermissionPolicy`
   keeps the always-require-approval behavior; only an explicit grant list
   downgrades.

For a path-only workspace boundary, pass `WorkspaceToolPermissionPolicy`:

```python
tools = base_harness_tools(
    cwd=WORKSPACE,
    permissions=WorkspaceToolPermissionPolicy(WORKSPACE),
)
```

This policy only answers "does the resolved path stay inside this root?" It is
not an allowlist, approval workflow, or process sandbox. Used by itself, it
denies `ToolOperation.EXECUTE_COMMAND`; include that operation explicitly only
when a separate policy or executor will handle command execution.

For workspace plus approved outside files or directories, use
`PathScopeToolPermissionPolicy` and include every approved scope explicitly:

```python
tools = base_harness_tools(
    cwd=WORKSPACE,
    permissions=PathScopeToolPermissionPolicy(
        (
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
`WorkspaceToolPermissionPolicy(...)`. Scope matching is a simple linear scan,
which keeps the bundled policy easy to understand for short approved lists.
Applications with hundreds of approved scopes or hot-loop checks should
provide an indexed custom policy.

Both path policies resolve symlinks and check the nearest existing parent for
new paths. This is a framework permission boundary, not a TOCTOU-safe operating
system sandbox; hosts that need hard filesystem isolation should provide a
sandboxed executor, container, or remote worker. The nearest-parent check uses
filesystem stats at permission time, which is appropriate for interactive
agent tool calls but not a substitute for kernel-enforced isolation.

For the common application policy "stay inside this workspace, apply a tool
grant allowlist, and require app approval before side effects", use
`workspace_tool_policy(...)`. It is an opinionated convenience constructor for
that workspace-app shape over the same public primitives, not a separate
policy system:

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
        WORKSPACE,
        grants=grants,
        require_approval_for_side_effects=True,
        require_bash_approval=True,
    ),
    approval_callback=approve,
)
```

Grant defaults are intentionally distinct. Omit `grants` or pass
`grants=None` when you do not want a grant allowlist; the helper will compose
only workspace and approval policies. Pass an empty iterable, `grants=()`,
only when the grant layer should exist and deny every tool.
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
                (WORKSPACE, EXTERNAL_REVIEW_FILE),
            ),
            ToolPermissionGrantPolicy(grants),
            SideEffectApprovalToolPermissionPolicy(),
        )
    ),
    approval_callback=approve,
)
```

That composition still denies `bash` by default because path scopes are not
process sandboxes. To include command execution in this composition, pass an
explicit `allowed_operations` set that includes `ToolOperation.EXECUTE_COMMAND`
and any path operations the app also wants to allow.

The exported permission primitives are framework extension points:

1. `ToolPermissionRequest` describes one operation before a side effect.
2. `ToolPermissionDecision` returns `allow`, `deny`, or `require_approval`. Its
   `outcome` field is a `ToolPermissionOutcome` enum (`ALLOW`, `DENY`,
   `REQUIRE_APPROVAL`), and `ToolApprovalCallback` is the callback type that
   resolves `require_approval` decisions.
3. `WorkspaceToolPermissionPolicy` is the built-in single-root path policy.
4. `PathScopeToolPermissionPolicy` is the built-in multi-scope path policy.
5. `AllOfToolPermissionPolicy` composes policies conservatively: deny wins,
   then approval, then allow.
6. `ToolPermissionGrantPolicy` and `parse_tool_permission_grants()` are small
   helpers for operation-level allowlists.
7. `SideEffectApprovalToolPermissionPolicy` is the generic approval policy for
   side-effecting operations.
8. `ToolApprovalBroker` is the optional host-facing approval lifecycle helper.
   It tracks pending approval records and exposes a callback compatible with
   first-party tools without owning any UI. Related event, record, and status
   types are exported alongside it.
9. `evaluate_tool_permission()` and `format_tool_permission_result()` are
   reusable helpers for custom tools that want the same deny and
   approval-required result contract. `evaluate_tool_permission()` returns
   `None` when execution may proceed, or a model-facing block result string
   when execution must stop.

`parse_tool_permission_grants()` accepts comma-separated whole-tool entries
such as `read` and operation entries such as `write:create_file`. It returns
`(grants, invalid_entries)`, preserves duplicates, and does not let later
entries override earlier ones. Callers should reject or report
`invalid_entries` before constructing a policy. Partial success is intentional:
CLI and environment-variable callers can collect every unsupported entry and
report them together instead of failing on the first one.

For programmatic grants, construct `ToolPermissionGrant` values directly.
`ToolPermissionGrant.all_operations("read")` makes whole-tool intent explicit;
`ToolPermissionGrant("write", ToolOperation.CREATE_FILE)` grants one
operation.

Custom policies only need a `check(request)` method:

```python
class DenyBash:
    def check(self, request: ToolPermissionRequest) -> ToolPermissionDecision:
        if request.operation == ToolOperation.EXECUTE_COMMAND:
            return ToolPermissionDecision.deny()
        return ToolPermissionDecision.allow()
```

`WorkspaceToolPermissionPolicy(...)` resolves existing paths through
symlinks. New paths are checked through their nearest existing parent, so
writing `nested/file.txt` still requires the target parent chain to stay inside
the workspace. Absolute path strings are accepted as inputs, but the policy
denies resolved absolute targets outside the root.

If an app needs a different scope shape, use one of these explicit choices:

1. Use a broader `WorkspaceToolPermissionPolicy(...)` while keeping
   `cwd=WORKSPACE` narrow for relative path resolution.
2. Use `PathScopeToolPermissionPolicy(...)` for workspace plus approved
   outside files or directories.
3. Pass no policy for a fully trusted local tool set.
4. Implement `ToolPermissionPolicy.check(...)` for app-specific rules such as
   per-user grants, read-only subtrees, or remote policy decisions.

`ToolPermissionRequest` has three groups of fields. Tool-action fields
(`tool_name`, `operation`, `cwd`, `path`, `command`) are built by the tool
after argument validation and path resolution. Framework correlation
(`run_id`, `agent_name`, `tool_call_id`) comes from `ToolExecutionContext`.
`metadata` is for app-defined correlation and is merged with context metadata,
with request metadata winning on key conflicts. `skill_name` and `reason` are
optional caller hints.

At the execution boundary, a `Tool` handler receives one
`ToolExecutionContext`. `ToolExecutor.execute(...)` accepts a mapping keyed by
tool-call id because one executor invocation may run several tool calls in
parallel and each call needs its own correlation. Custom executors should
preserve that shape: accept per-call context at the batch boundary, then pass
one `ToolExecutionContext` into each individual `Tool.run(...)` call.

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

Hosts that need pending-state tracking can use `ToolApprovalBroker` instead
of writing their own request id, event, and future glue:

```python
broker = ToolApprovalBroker()

tools = base_harness_tools(
    cwd=WORKSPACE,
    permissions=workspace_tool_policy(
        WORKSPACE,
        require_approval_for_side_effects=True,
    ),
    approval_callback=broker.callback,
)
```

Each brokered request receives a stable `request_id` and a
`ToolApprovalRecord` containing the original `ToolPermissionRequest`, the
approval-required `ToolPermissionDecision`, current status, and final decision
when complete. Hosts can use `broker.pending()` for a tuple snapshot and
`broker.events()` for async lifecycle events emitted after the subscription
starts:

```python
async for event in broker.events():
    if event.status == ToolApprovalStatus.PENDING:
        record = event.record
        await render_host_approval_prompt(record.request_id, record.request)
```

Use `broker.pending()` to seed UI state before consuming the live event stream.
Call `broker.close()` when the host is shutting down broker event observation;
this unblocks active `broker.events()` subscribers and makes future event
subscriptions end immediately. Closing event observation does not resolve
pending approvals or synthesize a permission decision.

The host resolves a pending request with a normal permission decision:

```python
await broker.resolve(request_id, ToolPermissionDecision.allow())
await broker.resolve(request_id, ToolPermissionDecision.deny())
await broker.resolve(request_id, original_approval_required_decision)
```

The broker does not own timeout, deadline, or cancellation policy. A host that
needs those behaviors should implement them at the application boundary and
resolve the request with an existing `ToolPermissionDecision`. For example, a
host-side timeout can preserve the normal approval-required tool output by
resolving with the original approval-required decision.

Custom tools use the same callback by delegating permission evaluation through
`evaluate_tool_permission(...)`:

```python
async def run_custom_tool(
    request: CustomArgs,
    *,
    permissions: ToolPermissionPolicy | None,
    approval_callback: ToolApprovalCallback | None,
) -> str:
    blocked = await evaluate_tool_permission(
        ToolPermissionRequest(
            tool_name="custom_tool",
            operation=ToolOperation.EXECUTE_COMMAND,
            cwd=WORKSPACE,
        ),
        policy=permissions,
        approval_callback=approval_callback,
    )
    if blocked is not None:
        return blocked

    return await perform_custom_side_effect(request)
```

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
| `grep` | `read_file` for existing file paths, `search_files` for directories and missing paths |
| `write` | `create_file`, `overwrite_file`, `create_directory` |
| `patch` | `modify_file` |
| `bash` | `execute_command` |

Egress tools (web search, API callers) use `network_access`; it is not bound to
a first-party local tool because the framework ships no egress tool, but the
operation is a stable part of `ToolOperation` for application egress tools.

`write` may issue two permission requests for one tool call. If the target's
parent directory does not exist, it checks `create_directory` for that parent
first. It then checks `create_file` for a missing target or `overwrite_file`
for an existing target. Policies that audit or log requests should expect this
one-or-two request shape.

The shared policy can require approval for local `bash` before the process
starts, but the default local executor is not filesystem-confined after
startup. Because `WorkspaceToolPermissionPolicy` is a path sandbox, it denies
`execute_command` unless `ToolOperation.EXECUTE_COMMAND` is explicitly
included in `allowed_operations` or another host policy allows it.
Applications that need real process isolation should provide a sandboxed
`BashExecutor`, container, remote worker, or equivalent host boundary.

## Network access (egress)

`ToolOperation.NETWORK_ACCESS` is the operation for tools that send data off
the machine, such as a web search or an API caller. Network egress has no
filesystem path for a path policy to bound, so an egress tool builds a
`ToolPermissionRequest` with `operation=ToolOperation.NETWORK_ACCESS`, an
optional `command`/payload describing the outbound data, and a human-readable
`reason`. Use this operation instead of borrowing `EXECUTE_COMMAND` for network
calls: a path or command sandbox would either deny the request or mislabel the
payload as a shell command.

`NETWORK_ACCESS` is classified as a side effect, so
`SideEffectApprovalToolPermissionPolicy()` (with default `operations`) returns
`require_approval` for it. The `reason` on the request flows through
`format_tool_permission_result(...)`: when a decision carries no `reason`, the
default approval-required text names the tool's network access rather than a
"command", so an approval UI can render an egress prompt without special-casing
a synthetic command shape.

```python
blocked = await evaluate_tool_permission(
    ToolPermissionRequest(
        tool_name="web_search",
        operation=ToolOperation.NETWORK_ACCESS,
        cwd=WORKSPACE,
        command=query,
        reason="approval required: outbound web search awaiting approval",
    ),
    policy=permissions,
    approval_callback=approval_callback,
)
if blocked is not None:
    return blocked
```

## Recording immediate decisions through the broker

By default `ToolApprovalBroker.callback(...)` only handles `require_approval`
decisions and raises on an already-decided `allow`/`deny`, because those need
no host resolution round-trip. A host that runs auto-allow or auto-deny modes
would then make those decisions outside the broker and keep its own counters,
so they never appear in `broker.events()`.

Construct the broker with `record_immediate_decisions=True` to route every
decision through the same observable place:

```python
broker = ToolApprovalBroker(record_immediate_decisions=True)
```

With the flag set, passing an `allow`/`deny` decision to `callback(...)`
records the request as resolved in the same step: the broker emits a `pending`
then a `resolved` event for it and returns the decision unchanged. The request
never lingers in `broker.pending()`. `require_approval` decisions still suspend
for host resolution exactly as before. The default stays `False` so existing
hosts that only broker approvals keep raising on non-approval decisions.
