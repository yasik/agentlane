# Harness Tools Sandbox and Permissions Phase 12

## Status

Implemented for review after approval on 2026-05-15. Phase 12 should still be
reviewed before the broader milestone is marked complete.

Linear issue: DIA-5302.

## Objective

Add an explicit sandbox and permissions layer for the first-party local
harness tools:

1. `bash`
2. `read`
3. `write`
4. `patch`
5. `grep`
6. `find`

The layer should make trust boundaries concrete without changing the existing
model-facing tool contracts more than necessary. Denied and approval-required
operations should return stable tool-result text so the agent loop can recover.

## Sources Reviewed

1. `docs/code-style/tool-design.md`
2. `docs/harness/tools.md`
3. Phase 12 in `docs/plans/agentic-harness-implementation-v1.md`
4. Current tool implementations under `src/agentlane/harness/tools/`
5. Current focused tests under `tests/harness/test_tools_*.py`
6. Current skills metadata support under `src/agentlane/harness/skills/`

## Current Contract Summary

All first-party tools return `HarnessToolDefinition` values. That means each
tool's public contract is the executable `Tool` schema plus prompt snippet and
prompt guidelines rendered by `HarnessToolsShim`.

`ToolPathResolver` is currently a resolver only. It captures construction-time
`cwd`, expands user paths, accepts absolute paths, resolves with
`Path.resolve(strict=False)`, and does not enforce a sandbox boundary,
permission allowlist, or approval workflow.

Tool-specific behavior today:

1. `read` reads one UTF-8-ish text file, rejects directories and binary-looking
   content, returns raw file contents plus continuation notes, and is read-only.
2. `find` walks one directory, respects `.gitignore`, skips `.git/`, does not
   follow symlinked directories, and returns matching file paths.
3. `grep` delegates content search to ripgrep, supports one explicit file or
   directory search, ignores per-file binary or warning noise in directory
   searches, and returns `Search path:` output.
4. `patch` edits one existing file through `llm-patch-tool`, with all-or-nothing
   SEARCH/REPLACE semantics and minimal success output.
5. `write` creates parent directories, creates or overwrites one UTF-8 file,
   and replaces existing files through a sibling temporary file where
   practical.
6. `bash` executes arbitrary non-interactive `bash -lc` commands in the
   captured `cwd` and returns combined stdout/stderr. The default executor
   allows all commands and does not sandbox the process; command permission
   checks are delegated to the shared permission policy.

The important gap is that path tools have structured operations but no policy
check, while `bash` has a policy check but cannot infer all filesystem effects
from an arbitrary shell string.

## Design Principles

1. Keep execution anchored on the existing `Tool` and `HarnessToolDefinition`
   primitives.
2. Keep construction-time configuration on tool helpers such as
   `read_tool(cwd=...)`.
3. Add one shared permission vocabulary instead of one custom policy type per
   tool.
4. Preserve existing success output shapes.
5. Return stable model-facing text for denied and approval-required decisions.
6. Check permissions before side effects.
7. Be honest about local `bash`: command approval is not the same as an OS
   sandbox. True process confinement requires a sandboxed executor supplied by
   the host application.
8. Keep skills on the same policy layer. Skill metadata should narrow or label
   requests; it should not create a parallel skill-only permission system.

## Non-Goals

1. No container runtime, chroot, seccomp, macOS Seatbelt profile, or remote
   worker implementation in Phase 12.
2. No shell parser that tries to prove a `bash` command's filesystem effects.
3. No interactive approval UI in the core harness.
4. No broad redesign of `ToolExecutor`, runner hooks, or model tool-call
   plumbing.
5. No new model-visible tool arguments for permissions.
6. No permission logic for `write_plan` or `agent` in this phase.

## Proposed Architecture

Add a small shared permissions module under `agentlane.harness.tools`.

Proposed internal module:

```text
src/agentlane/harness/tools/_permissions.py
```

Public exports should include stable developer-facing pieces that host
applications need to configure policies:

1. `ToolOperation`
2. `ToolPermissionRequest`
3. `ToolPermissionDecision`
4. `ToolPermissionPolicy`
5. `WorkspaceToolPermissionPolicy`
6. `PathScopeToolPermissionPolicy`
7. `ToolPermissionGrant`
8. `ToolPermissionGrantPolicy`
9. `SideEffectApprovalToolPermissionPolicy`
10. `AllOfToolPermissionPolicy`
11. `workspace_tool_policy`
12. `parse_tool_permission_grants`
13. `evaluate_tool_permission`
14. `format_tool_permission_result`

The policy boundary should be evaluated inside each tool helper before the
tool performs filesystem or process side effects. The runner and
`ToolExecutor` should not need to know about first-party tool permission
semantics.

### Permission Request

Use one request object for every first-party tool operation:

```python
@dataclass(frozen=True, slots=True)
class ToolPermissionRequest:
    tool_name: str
    operation: ToolOperation
    cwd: Path
    path: Path | None = None
    command: str | None = None
    skill_name: str | None = None
    reason: str | None = None
    run_id: str | None = None
    agent_name: str | None = None
    tool_call_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
```

`path` is the resolved filesystem target or search root when the operation is
path-based. `command` is set only for `bash`. `skill_name` is optional context
for future skill-driven calls. The explicit correlation fields are framework
owned; `metadata` is reserved for host-application correlation.

### Operations

Use operation names that describe real risk, not implementation details:

1. `READ_FILE`
2. `SEARCH_FILES`
3. `CREATE_FILE`
4. `OVERWRITE_FILE`
5. `MODIFY_FILE`
6. `CREATE_DIRECTORY`
7. `EXECUTE_COMMAND`

Tool mapping:

| Tool | Operations |
| --- | --- |
| `read` | `READ_FILE` |
| `find` | `SEARCH_FILES` |
| `grep` explicit file | `READ_FILE` |
| `grep` directory | `SEARCH_FILES` |
| `write` new file | `CREATE_FILE`, plus `CREATE_DIRECTORY` when parent directories would be created |
| `write` existing file | `OVERWRITE_FILE` |
| `patch` | `MODIFY_FILE` |
| `bash` | `EXECUTE_COMMAND` |

### Decisions

`ToolPermissionDecision` should have exactly three outcomes:

1. `allow`
2. `deny`
3. `require_approval`

Denied and approval-required decisions should render as tool results, not
exceptions. Draft stable result text:

```text
permission denied: read is not allowed for `/workspace/private.txt`
```

```text
approval required: bash command requires application approval before execution
```

The policy can carry an optional human-readable reason, but first-party tools
should render it only when it is already sanitized and intentionally
model-facing. The default denied message should be enough for the model to pick
a safer next step.

## Sandbox Semantics

Phase 12 should introduce a path sandbox for Python-owned filesystem tools.

Recommended first implementation:

1. `permissions=None` preserves current behavior when no policy is supplied.
   This is intentional framework behavior: AgentLane should give developers an
   explicit policy surface without forcing sandbox defaults on every
   application.
2. `WorkspaceToolPermissionPolicy(...)` allows path operations only when
   the resolved target is inside the configured root.
3. The policy resolves existing paths with symlink targets considered, so a
   symlink that points outside the root is denied.
4. For new paths, the nearest existing parent is resolved and must be inside
   the root before parent directories or files are created.
5. Absolute paths are not globally forbidden, but they must resolve inside the
   sandbox root when a workspace policy is active.
6. `find` and directory `grep` are allowed when the search root is inside the
   sandbox root. Phase 12 should not implement fine-grained per-entry deny
   lists inside an allowed search root.
7. `bash` is permission-checked but not path-sandboxed by the default local
   executor. The workspace policy denies `EXECUTE_COMMAND` unless that
   operation is explicitly granted, because a path sandbox cannot prove shell
   side effects. A host that needs real process confinement must provide a
   sandboxed `BashExecutor`.

This keeps the first layer simple and enforceable. Granular deny patterns,
read-only subtrees, and command parsers can be future extensions after the
shared contract is proven.

## Tool API Shape

Each tool helper should accept an optional shared policy:

```python
read_tool(cwd=workspace, permissions=WorkspaceToolPermissionPolicy(workspace))
```

Apply the same keyword to:

1. `read_tool`
2. `find_tool`
3. `grep_tool`
4. `patch_tool`
5. `write_tool`
6. `bash_tool`

Existing calls without `permissions=` keep the current trusted local behavior.
This preserves compatibility while giving applications an explicit safe path.
Public docs must state this clearly: the framework is permissive by default
until a developer passes a policy.

`base_harness_tools()` should gain optional `cwd` and `permissions` arguments so
applications can construct a coherent sandboxed tool set without manually
threading policy through every tool:

```python
workspace_tools = base_harness_tools(
    cwd=workspace,
    permissions=WorkspaceToolPermissionPolicy(workspace),
)
```

The legacy `bash_tool(policy=...)` hook has been removed during follow-up
cleanup. Command-level permission checks now flow through the shared
`permissions=` policy.

`bash` should remain part of `base_harness_tools()`. AgentLane is the framework,
not the final harness application, so it should expose the standard tool set
and let applications decide whether to pass a policy that denies command
execution or requires approval before it starts.

## Approval Callback

Phase 12 should define a framework-level approval seam, but it should not
implement an interactive approval UI.

Add an optional callback contract that host applications can provide:

```python
type ToolApprovalCallback = Callable[
    [ToolPermissionRequest, ToolPermissionDecision],
    ToolPermissionDecision | Awaitable[ToolPermissionDecision],
]
```

When a policy decides an operation requires approval:

1. if no approval callback is configured, the tool returns the stable
   `approval required: ...` result and does not execute,
2. if a callback is configured, the framework calls it with the same
   `ToolPermissionRequest` and the approval-required decision,
3. the callback can return allow, deny, or require-approval,
4. the core framework provides only the callback boundary and result handling;
   CLI, desktop, web, or service-specific approval UX belongs in the host
   application.

This gives future harness apps a clean integration point without baking an
application workflow into the library.

## Tool-Specific Plan

### `read`

Check `READ_FILE` after argument validation and path resolution, before
`is_dir()` and before opening the file.

Denied output should not be confused with OS permission errors. It should use
the shared policy-result wording, while existing `PermissionError` handling
continues to report OS-level permission failures.

### `find`

Check `SEARCH_FILES` on the resolved search root before `is_dir()` validation
and before `os.walk`.

Keep the existing `.gitignore`, `.git/`, symlinked-directory, ordering, and
truncation contracts unchanged.

### `grep`

Check `READ_FILE` for explicit file paths and `SEARCH_FILES` for directory
paths before invoking ripgrep.

Keep current best-effort handling for ripgrep per-file warnings, but avoid
using that mechanism for policy denials. A denied explicit path or denied
search root should return a policy result immediately.

### `write`

Validate path and content first. Then resolve the target and decide which
operations are needed:

1. missing parent directories require `CREATE_DIRECTORY`
2. missing target requires `CREATE_FILE`
3. existing file target requires `OVERWRITE_FILE`

All required operations must be allowed before any directory creation, temp
file creation, or replacement occurs.

The sibling temporary file for atomic replacement should be considered part of
the same approved target operation and should stay in the approved parent
directory.

### `patch`

Validate path and patch text first. Resolve the target, reject missing or
directory targets as today, then check `MODIFY_FILE` before parsing or applying
edits.

Do not build a diff-based approval workflow in the first pass. The existing
all-or-nothing patch contract remains the safety boundary for edit mechanics.

### `bash`

Check `EXECUTE_COMMAND` after command and timeout validation, before executor
startup.

Keep the existing `BashExecutor` seam. The shared policy decides whether the
command may start; the executor decides where and how it runs.

The docs must state this plainly:

1. local `bash` can be denied or require approval before start,
2. local `bash` is not filesystem-confined after start,
3. real process sandboxing requires a host-provided executor, container, or
   remote worker.

Full-output temp logs should remain outside the model-facing sandbox contract
for now, but the docs should note that hosts with strict data boundaries should
provide an executor that controls log storage.

## Skills Mapping

Skill tool metadata controls model-visible tool exposure, not host permission
grants. The allowlist field is `tools`; the deny-list field is
`disallowedTools`.

Phase 12 should enforce these fields without adding script execution.

Required behavior:

1. Missing `tools` inherits the current/session tool pool, subject to deny
   filters.
2. Present `tools` replaces the current/session tool pool with exactly those
   named tools.
3. `disallowedTools` is deny-first and subtractive. It removes tools before
   the model sees the active skill context.
4. If a name appears in both `tools` and `disallowedTools`, the deny rule wins.
5. Both fields accept a comma-separated string or YAML list.
6. Tool-selection metadata cannot override the developer's outer permission,
   sandbox, and approval policy.

This keeps Phase 12 focused on composable framework contracts: tool exposure is
handled before model invocation, while execution permission stays with the
host-provided policy layer.

## Implementation Checklist

- [x] Review and approve this design before runtime code changes.
- [x] Add shared permission primitives in `src/agentlane/harness/tools/_permissions.py`.
- [x] Export the approved public permission primitives from
      `agentlane.harness.tools`.
- [x] Add focused unit tests for allowed, denied, approval-required, and
      workspace-root decisions.
- [x] Add policy checks to `read`, `find`, and `grep` without changing
      successful output contracts.
- [x] Add policy checks to `write` and `patch`, including create, overwrite,
      modify, and parent-directory cases.
- [x] Adapt `bash` to the shared policy. The interim `BashPolicy`
      compatibility hook was removed during follow-up cleanup.
- [x] Add `cwd` and `permissions` arguments to `base_harness_tools()`.
- [x] Add the approval callback seam and approval-required result handling
      without adding an app-specific approval UI.
- [x] Enforce skill `tools` and `disallowedTools` metadata before model
      exposure.
- [x] Add tests for skill tool replacement, inherited tool filtering, and
      deny-first behavior.
- [x] Update `docs/code-style/tool-design.md` with the approved permission
      guidance.
- [x] Update `docs/harness/tools.md` with the new sandbox and permission
      behavior.
- [x] Update `docs/harness/skills.md` with `tools` and `disallowedTools`
      behavior.
- [x] Update `docs/plans/agentic-harness-implementation-v1.md` Phase 12 review
      notes after implementation.
- [x] Run targeted tests for all affected tools and skills.
- [x] Run `/usr/bin/make format`.
- [x] Run `/usr/bin/make lint`.
- [x] Run `/usr/bin/make tests`, noting the known duplicate-`conftest.py`
      typecheck blocker only if it appears in the selected validation path.
- [x] Stop for user review before marking Phase 12 complete.

## Targeted Validation Plan

Run these before repository-wide validation:

1. `uv run pytest -q tests/harness/test_tools_foundation.py`
2. `uv run pytest -q tests/harness/test_tools_read.py`
3. `uv run pytest -q tests/harness/test_tools_find.py`
4. `uv run pytest -q tests/harness/test_tools_grep.py`
5. `uv run pytest -q tests/harness/test_tools_write.py`
6. `uv run pytest -q tests/harness/test_tools_patch.py`
7. `uv run pytest -q tests/harness/test_tools_bash.py`
8. `uv run pytest -q tests/harness/test_skills.py`

Add one focused test module only if the shared permissions primitives become
large enough to justify it:

```text
tests/harness/test_tools_permissions.py
```

## Review Decisions

Reviewed on 2026-05-15:

1. Existing helpers remain permissive until a policy is passed explicitly. This
   is framework behavior and must be documented clearly.
2. `require_approval` should have a framework callback seam for future CLI,
   desktop, web, or service implementations. AgentLane should not implement the
   actual approval UX in this phase.
3. Skill tool selection uses `tools` for allowlist/replacement semantics and
   `disallowedTools` for deny-first subtraction.
4. `bash` should not be excluded from `base_harness_tools(...)`; applications
   decide whether to restrict it through policy.

## Review Notes

Implemented for review on 2026-05-15.

1. Added shared permission primitives, workspace-root policy enforcement,
   approval-required handling, and operation-level grant parsing.
2. Threaded `permissions=` and `approval_callback=` through `read`, `find`,
   `grep`, `write`, `patch`, `bash`, and `base_harness_tools()`.
3. Preserved permissive defaults when no policy is supplied.
4. Kept `bash` in the base tools set and documented that command permission is
   not process confinement. The workspace policy denies `execute_command`
   unless it is explicitly granted.
5. Updated skills parsing and activation so `tools` replaces the visible tool
   pool, `disallowedTools` subtracts before model exposure, and deny wins when
   both fields mention the same tool.
6. Added explicit framework correlation through `ToolExecutionContext`:
   the runner builds it, `ToolExecutor` passes it to the tool handler, and
   first-party permission checks copy `run_id`, `agent_name`, `tool_call_id`,
   and application metadata onto `ToolPermissionRequest`.
7. Generic spawned helpers now inherit parent descriptor shims directly, so
   configured base-tools `cwd`, permissions, and approval callback flow through
   the same `HarnessToolsShim` the parent already uses.
8. Validation passed:
   - `uv run pytest -q tests/models/test_tooling.py tests/harness/test_tools_permissions.py tests/harness/test_tools_read.py tests/harness/test_tools_bash.py tests/harness/test_tools_patch.py tests/harness/test_tools_agent.py tests/harness/test_skills.py tests/harness/test_runner.py tests/harness/test_shims.py` (172 passed)
   - `/usr/bin/make format`
   - `/usr/bin/make lint`
   - `/usr/bin/make tests` (524 passed)

## Follow-Up: Common Workspace App Policy

Review feedback after the first implementation identified that application
harnesses need a concise way to express the common policy shape:

1. path operations stay inside a workspace root,
2. tool or operation grants define the exposed capability set,
3. side-effecting operations require application approval, and
4. `bash` command execution is explicitly admitted to the permission layer
   before approval can decide whether it should run.

Tracked follow-up:

- [x] Add `workspace_tool_policy(...)` as a typed convenience constructor.
- [x] Keep `WorkspaceToolPermissionPolicy`, `ToolPermissionGrantPolicy`, and
      `AllOfToolPermissionPolicy` public for custom compositions.
- [x] Require `require_bash_approval=True` before `bash:execute_command` can
      pass the workspace policy, and make that path require approval before the
      process starts.
- [x] Add tests covering read allow, write approval, bash approval, outside-path
      denial, and missing bash approval denial.
- [x] Document the helper in `docs/harness/tools.md`.

## Follow-Up: Permission Clarity Pass

Review feedback after the convenience helper asked that the permission system
be unambiguous, straightforward to use, and documented with high-quality
inline comments.

Tracked follow-up:

- [x] Clarify permission code comments around approval, grant allowlists,
      workspace path checks, explicit correlation, and `bash` admission.
- [x] Document that `grants=None` means no grant allowlist and `grants=()`
      means an empty allowlist.
- [x] Add regression tests for omitted grants versus an empty grant list.
- [x] Update public docs so common workspace policy setup is a direct recipe
      and low-level primitives remain clear extension points.
- [x] Remove the public allow-all policy sentinel and document
      `permissions=None` as the canonical permissive default.

## Follow-Up: Approved External Path Scopes

Review feedback after the clarity pass identified a common coding-assistant
case: the app may run from a workspace `cwd`, but the user may approve
specific files or directories outside that workspace for review.

Tracked follow-up:

- [x] Keep `WorkspaceToolPermissionPolicy` as a hard single-root boundary.
- [x] Add `PathScopeToolPermissionPolicy` for explicit files or directories,
      including approved paths outside the workspace.
- [x] Make the generic side-effect approval policy public so apps can compose
      path scopes, grants, and approval directly.
- [x] Document alternative paths: broader root, path scopes, no policy for
      trusted tools, or custom policy implementation.
