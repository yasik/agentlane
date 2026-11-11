"""Small file I/O contracts shared by tools and skill discovery."""

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable

from agentlane.io import Reader, Writer

# Retain the existing import name for reader implementations.
BinaryReader = Reader


@dataclass(frozen=True, slots=True)
class FileInfo:
    """File metadata for permission decisions."""

    is_directory: bool
    """Whether the path identifies a directory instead of a file."""


@dataclass(frozen=True, slots=True)
class DirectoryEntry:
    """One direct child of a directory."""

    name: str
    """Child name without path separators; never `.` or `..`."""

    is_directory: bool
    """Whether the child is a directory that can be traversed."""

    is_symlink: bool = False
    """Whether the child is a symbolic link; recursive listings skip directory links."""

    modified_time: float | None = None
    """Unix modification time when available; find uses zero when absent."""


class FileReader(Protocol):
    """Open files in a storage namespace.

    Injected readers receive relative POSIX paths unless they implement
    `FilePathResolver`, which defines their logical namespace. Mounted child
    readers receive relative POSIX paths. The implementation owns the
    physical root, credentials, timeouts, and error translation. Use standard
    `OSError` subclasses for file errors. Calls run in worker threads, so each
    call must own its stream and be safe to run alongside other calls.
    """

    def open_read(self, path: str) -> AbstractContextManager[BinaryReader]:
        """Open a binary stream and close it when the context exits."""
        ...


@runtime_checkable
class FilePathResolver(Protocol):
    """Optional lexical path resolution for a filesystem's logical namespace.

    File tools use this capability to resolve cwd and tool arguments before
    permission checks and I/O. Resolution must not depend on mutable cwd state
    or require that the target exists.
    """

    def resolve_path(self, path: str, *, cwd: str = "/") -> PurePosixPath:
        """Return a canonical path in this filesystem's logical namespace."""
        ...


@runtime_checkable
class FileStat(Protocol):
    """Inspect paths independently of read or write access."""

    def stat(self, path: str) -> FileInfo | None:
        """Return metadata or None for a missing path. The root must exist."""
        ...


@runtime_checkable
class FileWriter(Protocol):
    """Open a writer in the same logical namespace as its reader.

    Paths are relative POSIX paths unless `FilePathResolver` defines a different
    namespace. Mounted child writers receive relative POSIX paths.
    """

    def open_write(self, path: str) -> AbstractContextManager[Writer]:
        """Create or replace a file and create its parents as needed.

        Successful context exit completes the write. An exception must abort
        replacement and preserve any existing file. Each context owns its
        stream. The backend owns concurrent-write control.
        """
        ...


class FileRevisionConflictError(OSError):
    """A revision-checked write found that the file changed after it was read."""


@runtime_checkable
class FileRevisions(Protocol):
    """Optional revision checks for read-modify-write tools.

    A revision is an opaque token for the stored content of one file. Only the
    backend creates and compares revisions. A backend that cannot report one for
    a file returns None, and the caller then writes without a check.

    The revision check and the replacement must be one atomic operation when the
    write commits. A check only when the writer opens leaves a race: another
    writer can commit between that check and the replacement.
    """

    def open_read_revision(
        self, path: str
    ) -> AbstractContextManager[tuple[BinaryReader, str | None]]:
        """Open a binary stream and the revision of exactly the content it reads.

        The revision must identify the bytes this stream returns, not a later or
        earlier version of the file.
        """
        ...

    def open_write_revision(
        self, path: str, revision: str
    ) -> AbstractContextManager[Writer]:
        """Replace a file only while it still has `revision`.

        Compare the revision and replace the file in one atomic operation when the
        writer commits. Raise `FileRevisionConflictError` and preserve the stored
        file when the revision no longer matches. Otherwise behave like
        `FileWriter.open_write`.
        """
        ...


@runtime_checkable
class WritableFileSystem(FileWriter, FileStat, Protocol):
    """Write tools need both write access and permission-relevant metadata."""


class DirectoryLister(Protocol):
    """List direct children in the same namespace as a reader."""

    def list_directory(self, path: str) -> Sequence[DirectoryEntry]:
        """Return children and identify directory links to prevent cycles.

        Names must be single path components. Raise `FileNotFoundError` for
        missing directories and `NotADirectoryError` for files.
        """
        ...


@runtime_checkable
class SkillReader(FileReader, DirectoryLister, Protocol):
    """Read and list files for the SDK's native skill loader."""


@runtime_checkable
class ReadableFileSystem(FileReader, DirectoryLister, FileStat, Protocol):
    """Read, list, and inspect paths for file search."""
