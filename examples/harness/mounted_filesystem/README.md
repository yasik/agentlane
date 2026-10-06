# Mounted filesystem tools

This example writes, finds, patches, and reads a mounted file with a captured
working directory of `/tenant/sessions/session-123`. It also reads another
mount with an explicit rooted path. Grep and bash operate separately in the
process workspace. Two temporary local directories provide the storage; no
model credentials are required.

The `rg` executable from ripgrep must be available on `PATH` for the grep call.
Run from the repository root:

```bash
uv run python examples/harness/mounted_filesystem/main.py
```

The file tools resolve `notes.txt` to
`/tenant/sessions/session-123/notes.txt`. The tenant backend receives
`sessions/session-123/notes.txt`. A search under `/workspace` returns
`process.txt`; the example reads it with `/workspace/process.txt`. This read
does not change the session cwd for later patch and read calls.

Replace a child `LocalFileSystem` with an adapter that implements the filesystem
capabilities to use client storage. A child with no `FileWriter` is read-only.
The example has no permission policy; applications must supply one to restrict
writes to a session. A cwd selects a default location and does not limit access.
Grep and bash access remote data only if the host exposes it to their process
filesystem. See [File I/O interfaces](../../../docs/harness/filesystem.md).
