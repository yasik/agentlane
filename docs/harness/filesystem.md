# File I/O Interfaces

File tools depend on interfaces. AgentLane supplies a local filesystem by
default. Applications can supply their own implementations for object stores,
remote services, or other storage. The framework does not include cloud clients.

## Byte Streams

`agentlane.io` defines two independent protocols:

| Protocol | Method | Contract |
| --- | --- | --- |
| `Reader` | `read(size=-1) -> bytes` | Short reads are valid; empty bytes mean end of stream |
| `Writer` | `write(data: bytes) -> int` | Return the number of bytes accepted; short writes are valid |

These protocols have no paths, metadata, seeking, line operations, or harness
dependencies. A client implements only the methods it needs. Python binary
file streams and `BytesIO` satisfy them.

Buffering and complete transfers are separate operations:

- `buffered_reader(reader)` adds Python's standard buffering and `readline`.
  Closing this wrapper does not close the supplied reader.
- `read_all(reader)` reads to the end of the stream, including short reads.
- `write_all(writer, data)` handles short writes and rejects zero progress or
  invalid byte counts. Completion of this operation does not close the writer.

The filesystem's context manager owns stream cleanup and write completion.
`BinaryReader` remains available from `agentlane.harness.filesystem` as an alias
for `Reader`; implementations no longer need a `readline` method.

## Filesystem Capabilities

Path operations live in `agentlane.harness.filesystem`, independently of the
stream protocols and concrete implementations.

| Protocol | Methods | Consumers |
| --- | --- | --- |
| `FileReader` | `open_read(path)` | Read and patch tools |
| `FileWriter` | `open_write(path)` | Patch writes |
| `FileStat` | `stat(path)` | File inspection and permission decisions |
| `DirectoryLister` | `list_directory(path)` | Directory traversal |
| `SkillReader` | Read and list | Skill discovery and activation |
| `ReadableFileSystem` | Read, list, and stat | Find tool |
| `WritableFileSystem` | Write and stat | Write tool |

`open_read` returns a context manager for a `Reader`. `open_write` returns one
for a `Writer`. Each context owns its stream. Use standard `OSError` subclasses
for storage failures.

A writer creates parent directories as needed. Successful context exit must
complete the write. A failed context must preserve an existing target instead
of committing partial content. The backend owns concurrent-write control.
`LocalFileSystem` uses a temporary file and atomic replacement for this contract.
New files use normal creation permissions under the process umask. Replacement
preserves existing mode bits.
A remote adapter must implement the contract using its provider's facilities.

`stat` returns `FileInfo(is_directory=...)`, or `None` for a missing path.
The root `.` must exist. Metadata is independent of write access.

`list_directory` returns direct children as `DirectoryEntry` values. Names must
be single path components. Directory links must set `is_symlink=True` so
traversal can avoid cycles. Raise `FileNotFoundError` for a missing directory
and `NotADirectoryError` for a file. `DirectoryEntry.modified_time` is an
optional Unix timestamp; find uses zero when absent and breaks ties by path.
Find uses listing metadata without a separate stat call for each file.
Unexpected listing failures return a tool error, not an empty result.

## Tool Construction

Pass your storage implementation to each tool that needs it:

```python
from agentlane.harness.tools import read_tool, write_tool, find_tool, patch_tool

# storage is the application's filesystem implementation.
tools = (
    read_tool(reader=storage),
    write_tool(writer=storage),
    find_tool(reader=storage),
    patch_tool(reader=storage, writer=storage),
)
```

Tools depend on the required interfaces without checking provider types.
Find uses one traversal for local and supplied filesystems. Patch uses the
`llm-patch-tool` edit algorithm; its content operation applies edits while the
supplied filesystem performs the reads and writes.

`FilesystemSkillLoader(reader=storage, roots=("skills",))` uses the same reader
and directory-listing contracts. Activation returns a `read_path` that includes
the skill root. Supply that path to a read tool at storage `cwd="."`.

## Paths and Workspace

Injected file tools use relative POSIX paths. `cwd` defaults to `.` within the
supplied storage. The adapter owns its physical root and access restrictions.
`normalize_relative_path(path, root=".")` normalizes paths without local file
access. It rejects absolute paths, drive paths, backslashes, null bytes, and
`..` paths that leave the storage root. It also rejects path components that
start with `~`, including `~user`. Mount routing can expose these components
to home-directory expansion in a local backend. Names such as `report~backup.txt`
remain valid. The same rules apply to the storage `cwd`.

The default local tools retain absolute paths and capture the working directory
at construction. A `LocalFileSystem(root=...)` supplied to a file tool uses the
tool's injected path rules. Its physical root is a working directory, not a
security boundary.

The base factory keeps the process workspace separate from injected storage:

```python
from agentlane.harness.tools import base_harness_tools

tools = base_harness_tools(
    cwd="/workspace",       # Files visible to grep and bash in their environment.
    reader=storage,
    writer=storage,
    storage_cwd="skills",   # Relative directory for read, write, find, and patch.
)
```

If the reader implements `WritableFileSystem`, the factory uses it as the
writer unless an explicit writer is supplied. Missing capabilities for selected
file tools raise `ValueError`; use selectors for a read-only tool set.

Grep and bash remain process tools. They use `cwd` in the environment that runs
them, such as the user's computer or a sandbox. Grep invokes ripgrep in the
harness process environment. Bash uses its executor; the base factory accepts
`bash_executor=` for an application-supplied executor. To keep both tools in
one sandbox workspace, run the tool handlers in that environment. An explicit
bash cwd passes to custom executors without host symlink resolution or tilde
expansion; relative paths are resolved by that executor. The base tools add
prompt guidance about the separate storage and process path namespaces.

Neither tool consumes the reader or writer. A logical mount does not create an
operating-system mount. If commands need remote data, the host must expose it
to their process filesystem. AgentLane does not copy remote files for search.

## Mixed Local and Remote Filesystems

`MountedReader` routes reads, listings, and metadata through named children.
`MountedFileSystem` adds writes for children that implement `FileWriter`.
Read-only children reject writes with `PermissionError`.

To combine local files with your `remote_storage` adapter, assign each backend
a mount name:

```python
from agentlane.harness.filesystem import LocalFileSystem, MountedFileSystem
from agentlane.harness.tools import base_harness_tools

storage = MountedFileSystem({
    "workspace": LocalFileSystem(root="/workspace"),
    "tenant": remote_storage,
})
tools = base_harness_tools(cwd="/workspace", reader=storage)
```

Mount names are canonical single POSIX path components. The mapping is copied
at construction. A path such as `tenant/reports/result.txt` selects `tenant`
and passes `reports/result.txt` to that child. The same path works through
read, write, find, and patch. Mounted implementations do not contain search,
copying, buffering, or cloud-provider logic.

Listing `.` returns sorted mount names as directories. A named mount root
lists the child at `.`. Reads and writes to roots raise `IsADirectoryError`.
Writes cannot create mounts. Unknown mounts raise `FileNotFoundError` for
reads, listings, and writes; `stat` returns `None`. Child errors propagate.
Readers without `FileStat` can still be mounted: metadata is obtained from
their directory listings. Find uses optional timestamps from these same entries.

Path normalization can cross mounts: `workspace/../tenant/file.txt` becomes
`tenant/file.txt`. Tool `cwd` is not an access boundary. Child adapters must
enforce physical access restrictions, including symlink rules.

Use `MountedReader` with `FilesystemSkillLoader` to combine skill roots.
Earlier roots retain precedence when skill names repeat. Skill activation does
not change the tool working directory.

See the [runnable example](../../examples/harness/mounted_filesystem/main.py).

## Permissions and Execution

Injected file-tool permission requests carry logical `PurePosixPath` values.
The local `WorkspaceToolPermissionPolicy` and `PathScopeToolPermissionPolicy`
deny these because their checks apply to local paths. Supply an appropriate
namespace policy. Operation grants and approval callbacks still apply.

Read checks `READ_FILE`, find checks `SEARCH_FILES`, and patch checks
`MODIFY_FILE` before accessing data. Write inspects metadata to distinguish
creation from overwrite, then checks permissions before opening a writer.
Grep and bash retain their process-tool permission behavior. See
[Tool permissions](tools-permissions.md).

Skill loading calls the adapter directly. Tool policies do not govern skill
loader access; restrict it through the adapter and configured roots.

Adapter methods are synchronous and file tools run them in worker threads.
Each call must own its stream. Shared clients must support concurrent calls or
provide their own synchronization. The adapter owns timeouts and retries.
Cancellation cannot stop a blocking adapter call. Write and patch wait for a
started write context to settle before they propagate task cancellation.

## Writer Migration

A previous writer adapter implemented `stat(path)` and `write(path, bytes)`.
Implement `open_write(path)` as a context manager that yields a `Writer`, and
keep `stat` for the write tool. Whole-file writing is now an operation over the
stream. `LocalFileSystem.write` remains a convenience method; tools depend on
the stream interface. Readers need only `read`; `buffered_reader` supplies
buffering and line reads.

Injected patch calls through one tool instance serialize the complete read,
edit, and write operation. This lock does not cover other tool instances or
external writers. Applications must coordinate those operations as a whole;
serializing write commits alone does not prevent stale read-modify-write updates.
