# patch Tool

`patch_tool()` exposes a `patch` tool for precise edits to existing UTF-8 text
files. It is backed by
[`llm-patch-tool`](https://github.com/yasik/patch-tool), which handles parsing
SEARCH/REPLACE blocks, exact-then-fuzzy matching, all-or-nothing application,
and atomic writes.

Parameters:

1. `path: str`
2. `edits: str`

## Permissions

`patch` resolves `path` through `ToolPathResolver` and checks
`ToolOperation.MODIFY_FILE` before parsing or applying edits. A denied request
returns:

```text
permission denied: patch is not allowed for `/workspace/private.txt`
```

`SideEffectApprovalToolPermissionPolicy` and
`workspace_tool_policy(require_approval_for_side_effects=True)` request
approval for patch operations before the file is modified.

An approval-required request returns:

```text
approval required: patch requires application approval for `/workspace/notes.txt` before execution
```

`path` is structured tool input and resolves through `ToolPathResolver`.
`edits` should contain one or more bare SEARCH/REPLACE blocks without path
lines:

```text
<<<<<<< SEARCH
old text already present in the file
=======
replacement text
>>>>>>> REPLACE
```

Example tool result:

```text
Applied 1 edit to /workspace/notes.txt.
```

Use `patch` after reading the file when you need targeted changes. Each SEARCH
block must match exactly one location. If the text is missing, appears more
than once, overlaps another edit, has an empty SEARCH block, or would not
change the file, the tool returns a stable recoverable message and leaves the
file unchanged. Use `write` instead for new files or full-file rewrites.

The tool returns clear text errors for empty paths, paths containing null bytes,
missing files, directory targets, malformed SEARCH/REPLACE blocks, invalid
UTF-8 edit text, invalid UTF-8 files, permission failures, and failed writes.
Unexpected implementation errors return a stable generic failure message so the
agent loop can continue.
