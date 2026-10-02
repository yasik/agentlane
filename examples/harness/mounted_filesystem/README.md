# Mounted Filesystem Tools

This example writes, finds, patches, and reads a mounted file. Grep and bash
operate separately in the process workspace. It uses two temporary local
directories and needs no model credentials.

The `rg` executable from ripgrep must be available on `PATH` for the grep call.
Run from the repository root:

```bash
uv run python examples/harness/mounted_filesystem/main.py
```

Replace a child `LocalFileSystem` with an adapter that implements the filesystem
capabilities to use client storage. A child with no `FileWriter` is read-only.
Grep and bash access remote data only if the host exposes it to their process
filesystem. See [File I/O interfaces](../../../docs/harness/filesystem.md).
