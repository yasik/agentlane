"""Small file I/O contracts shared by tools and skill discovery."""

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
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

    Injected readers receive relative POSIX paths. The implementation owns the
    physical root, credentials, timeouts, and error translation. Use standard
    `OSError` subclasses for file errors. Calls run in worker threads, so each
    call must own its stream and be safe to run alongside other calls.
    """

    def open_read(self, path: str) -> AbstractContextManager[BinaryReader]:
        """Open a binary stream and close it when the context exits."""
        ...


@runtime_checkable
class FileStat(Protocol):
    """Inspect paths independently of read or write access."""

    def stat(self, path: str) -> FileInfo | None:
        """Return metadata or None for a missing path. The root must exist."""
        ...


@runtime_checkable
class FileWriter(Protocol):
    """Open a writer in a storage namespace. Paths are relative POSIX paths."""

    def open_write(self, path: str) -> AbstractContextManager[Writer]:
        """Create or replace a file and create its parents as needed.

        Successful context exit completes the write. An exception must abort
        replacement and preserve any existing file. Each context owns its
        stream. The backend owns concurrent-write control.
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
