# write Tool

`write_tool()` exposes a `write` tool for creating or overwriting UTF-8 text
files.

Parameters:

1. `path: str`
2. `content: str`

## Permissions

`write` resolves `path` through `ToolPathResolver` and may issue two checks:
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

```text
approval required: write requires application approval for `/workspace/notes.txt` before execution
```

Example tool result:

```text
Wrote 128 bytes to /workspace/notes.txt.
```

The tool creates parent directories automatically. Existing files are replaced
through a sibling temporary file where practical.

Use `write` for new files or complete rewrites. It does not provide append mode
or precise patch operations.

The tool returns clear text errors for empty paths, paths containing null bytes,
directory targets, parent paths that are files, invalid UTF-8 content,
permission failures, and other failed writes. Unexpected implementation errors
return a stable generic failure message so the agent loop can continue.
