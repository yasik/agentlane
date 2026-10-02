# write Tool

`write_tool()` exposes a `write` tool for creating or overwriting UTF-8 text
files.

Parameters:

1. `path: str`
2. `content: str`

At construction, pass `writer=` to write to application storage. The native
schema and result format stay the same. The writer receives relative POSIX
paths and opens a byte writer; `cwd` defaults to its root (`.`). Omit `writer`
for local files. See [File I/O interfaces](filesystem.md) for metadata,
replacement, and cancellation requirements.

## Permissions

For local files, `write` resolves `path` through `ToolPathResolver`. An injected
writer uses relative storage paths. Both modes may issue two checks:
`ToolOperation.CREATE_DIRECTORY` for a missing parent directory, then
`ToolOperation.CREATE_FILE` or `ToolOperation.OVERWRITE_FILE` for the target.
A denied request returns:

```text
permission denied: write is not allowed for `/workspace/private.txt`
```

`SideEffectApprovalToolPermissionPolicy` and
`workspace_tool_policy(require_approval_for_side_effects=True)` request
approval for each required write operation before any directory or file is
created.

An approval-required request returns:

```text
approval required: write requires application approval for `/workspace/notes.txt` before execution
```

Example tool result:

```text
Wrote 128 bytes to /workspace/notes.txt.
```

The tool asks the writer to create parent directories automatically. The local
writer replaces existing files through a sibling temporary file. Injected
writer contexts must complete writes on successful exit and preserve an
existing file on failure. The tool uses `write_all` to handle short writes.

Use `write` for new files or complete rewrites. It does not provide append mode
or precise patch operations.

The tool returns clear text errors for empty paths, paths containing null bytes,
directory targets, parent paths that are files, invalid UTF-8 content,
permission failures, and other failed writes. Unexpected implementation errors
return a stable generic failure message so the agent loop can continue.
