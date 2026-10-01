# File I/O Adapters

Use `agentlane.harness.filesystem` to supply storage for the native read tool,
write tool, and skill loader. AgentLane keeps the tool schemas, text limits,
permission checks, skill discovery, private frontmatter parser, and activation
behavior. The application supplies file I/O.

## Interfaces

Implement only the protocols that the application needs. No inheritance is
required.

| Protocol | Methods | Used by |
| --- | --- | --- |
| `FileReader` | `open_read(path)` | `read_tool(reader=...)` |
| `FileWriter` | `stat(path)`, `write(path, content)` | `write_tool(writer=...)` |
| `DirectoryLister` | `list_directory(path)` | Skill discovery and resource listing |
| `SkillReader` | `FileReader` and `DirectoryLister` methods | `FilesystemSkillLoader(reader=...)` |

`open_read` returns a context manager for a binary stream. The stream implements
`read(size=-1)` and `readline(size=-1)`; seeking is not required. AgentLane opens,
reads, and closes the stream in the same worker operation.

`stat` returns `FileInfo(is_directory=...)`, or `None` for a missing path. The
root `.` must exist. Metadata must describe the same storage that `write`
changes. `write` receives bytes and creates parent directories as needed. The
adapter owns replacement guarantees.

`list_directory` returns direct children as `DirectoryEntry` values. Each name
must be one path component. Set `is_directory` and `is_symlink` correctly so
resource discovery can skip directory links. Raise `FileNotFoundError` for a
missing directory and `NotADirectoryError` for a file. Use standard `OSError`
subclasses for storage errors.

## Paths and Composition

The adapter owns its physical root, such as a bucket prefix or database tenant.
Injected adapters receive normalized relative POSIX path strings. No host path
resolution occurs. Absolute paths, Windows drive paths, backslashes, null bytes,
and `..` paths that leave the storage root are rejected.

The public `normalize_relative_path(path, root=".")` helper applies these rules
and returns a `PurePosixPath`. It removes redundant separators and `.` segments,
resolves `..` within the storage root, and raises `ValueError` for invalid paths.
It does not expand `~` or inspect local files or symlinks.

With an injected adapter, tool `cwd` is a relative directory in that storage
and defaults to `.`. It is not an additional access boundary: `../file.txt`
can leave `cwd` if it stays inside the adapter's root. Use adapter access checks
and a permission policy when the application needs a narrower boundary.

Use the same reader namespace for skill loading and resource reading:

```python
from agentlane.harness.skills import FilesystemSkillLoader, SkillsShim
from agentlane.harness.tools import HarnessToolsShim, read_tool, write_tool

loader = FilesystemSkillLoader(reader=storage, roots=("skills",))
shims = (
    SkillsShim(loader=loader),
    HarnessToolsShim((read_tool(reader=storage), write_tool(writer=storage))),
)
```

Here `storage` is an application adapter that implements `SkillReader` and
`FileWriter`. The loader defaults to `roots=(".",)` for an injected reader.
It ignores `include_default_roots` and does not inspect local home or working
directories. Activated resources keep their skill-relative `path`; `read_path`
adds the skill root and is suitable for the read tool at `cwd="."`.

For a skill under `skills/refund-policy`, the paths are:

| Value | Path |
| --- | --- |
| Loader root | `skills` |
| Manifest `root` | `skills/refund-policy` |
| Manifest `skill_file` | `skills/refund-policy/SKILL.md` |
| Resource `path` | `references/policy.md` |
| Activation `read_path` | `skills/refund-policy/references/policy.md` |

The read tool passes that `read_path` unchanged when `cwd="."`. With
`cwd="skills/refund-policy"`, pass `references/policy.md` instead. The tool
resolves it to the same storage path. Skill activation does not change `cwd`.
The adapter maps the resulting path to its physical root. AgentLane does not
create a local copy or choose an organization, agent, or session prefix.

`base_harness_tools(reader=storage, writer=storage, include=("read", "write"))`
is another way to construct those two tools. The factory injects only `read`
and `write`. `find`, `grep`, `patch`, and `bash` still use local files. Construct
tools separately when they need different working directories or policies.

Omit `reader` and `writer` to retain local defaults through `LocalFileSystem`.
Local tools still accept absolute paths, and the local skill loader still uses
its configured roots and standard local roots. An explicitly supplied
`LocalFileSystem(root=...)` follows injected relative-path rules; its physical
root remains a working directory, not a security boundary.
The local reader accepts regular files and rejects special files such as named
pipes. Local directory listings omit special files and identify symbolic links.

### Mixed Local and Remote Readers

Use `MountedReader` to combine local and remote storage in one reader. Share
it between `FilesystemSkillLoader` and one `read` tool:

```python
from agentlane.harness.filesystem import LocalFileSystem, MountedReader
from agentlane.harness.skills import FilesystemSkillLoader, SkillsShim
from agentlane.harness.tools import HarnessToolsShim, read_tool

storage = MountedReader(
    {
        "workspace": LocalFileSystem(root="/app"),
        "tenant": remote_reader,
    }
)
loader = FilesystemSkillLoader(
    reader=storage,
    roots=("tenant/skills", "workspace/.agents/skills"),
)
shims = (
    SkillsShim(loader=loader),
    HarnessToolsShim((read_tool(reader=storage),)),
)
```

Here `remote_reader` is an application adapter that implements `SkillReader`.
Each mounted reader must support file reads and directory listings. The mount
mapping is copied at construction. Mount names must be single, canonical
relative POSIX path components, such as `workspace` or `tenant`.

The first component of a normalized path selects the reader. The selected
reader receives the remaining path:

| Tool path | Reader | Path passed to reader |
| --- | --- | --- |
| `tenant/skills/refund/SKILL.md` | `remote_reader` | `skills/refund/SKILL.md` |
| `workspace/.agents/skills/review/SKILL.md` | `LocalFileSystem` | `.agents/skills/review/SKILL.md` |
| `workspace/reports/result.txt` | `LocalFileSystem` | `reports/result.txt` |

Earlier loader roots win when skill names repeat. With this example, a remote
skill takes precedence over a local skill with the same name. Discovery and
activation use the same selected skill. A resource's `path` remains relative
to its skill directory; its `read_path` includes the mount, such as
`tenant/skills/refund/references/policy.md`.

Listing `.` returns mount names as directories in sorted order. Listing a
mount root, such as `tenant`, calls that reader's `list_directory(".")`.
Reading `.` or a mount root raises `IsADirectoryError`. Unknown mounts raise
`FileNotFoundError`. Errors from a selected reader propagate; no other reader
is tried.

Tool permissions receive the full logical path, including the mount name.
Path normalization can move between mounts: `workspace/../tenant/file.txt`
resolves to `tenant/file.txt`. Tool `cwd` is not an access boundary. Each child
reader must enforce physical access restrictions, including local symlink
restrictions. `LocalFileSystem(root=...)` alone does not enforce these limits.

`MountedReader` supports reads and directory listings only. Local tools such
as `find`, `grep`, `patch`, and `bash` keep their local path rules. A mounted
path such as `workspace/report.txt` refers to `/app/report.txt` in this example;
pass a local path to those other tools.

## Permissions and Execution

Injected tool requests carry `PurePosixPath` values in `cwd` and `path`.
`WorkspaceToolPermissionPolicy` and `PathScopeToolPermissionPolicy` deny these
paths because their checks apply to local `Path` objects. Supply a policy that
checks the storage namespace. Operation grants and approval callbacks still
apply. Write tools check directory creation and file creation or overwrite
before calling the adapter's `write`; metadata reads occur before these checks.
See [Tool permissions](tools-permissions.md).

`FilesystemSkillLoader` calls its reader directly for discovery and activation.
Policies passed to read or write tools do not govern these loader calls. Limit
skill access through the reader and its configured storage root.

All adapter methods are synchronous and run on worker threads. Each read must
own its stream. Shared clients must support concurrent calls, or the adapter
must protect them. Configure storage request timeouts and retry limits in the
adapter; AgentLane does not set a storage deadline.

Cancellation cannot stop a blocking thread. A read or listing can continue
after its caller is cancelled. The write tool waits for a started write to
finish, including after repeated cancellation, before it propagates
cancellation. This does not roll back the write. A slow write can therefore
delay cancellation until the adapter returns or raises.

## Minimal Reader Example

Save this example as a Python file and run it with `uv run python <file>`.
It uses the native read tool without local files or credentials. The dictionary
is copied at construction and is not changed after that, so concurrent reads
use separate streams over fixed bytes.

```python
import asyncio
from io import BytesIO

from agentlane.harness.tools import read_tool
from agentlane.models import Tool
from agentlane.runtime import CancellationToken


class MemoryReader:
    def __init__(self, files: dict[str, bytes]) -> None:
        self._files = dict(files)

    def open_read(self, path: str) -> BytesIO:
        try:
            content = self._files[path]
        except KeyError as error:
            raise FileNotFoundError(path) from error
        return BytesIO(content)


async def main() -> None:
    storage = MemoryReader({"notes/example.txt": b"alpha\nbravo\n"})
    tool = read_tool(reader=storage).tool
    assert isinstance(tool, Tool)
    args = tool.args_type().model_validate({"path": "notes/example.txt", "limit": 1})
    result = await tool.run(args, CancellationToken())
    print(result)


asyncio.run(main())
```

The result contains `alpha` and the native continuation note. To use an adapter
with the skill loader, also implement `list_directory`; do not copy or expose
the private parser. See [Skills](skills.md) for discovery and activation.
