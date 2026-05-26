# write Tool

`write_tool()` exposes a `write` tool for creating or overwriting UTF-8 text
files.

Parameters:

1. `path: str`
2. `content: str`

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
