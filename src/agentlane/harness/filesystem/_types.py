"""Small file I/O contracts shared by tools and skill discovery."""

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol


class BinaryReader(Protocol):
    """Blocking byte stream. Seeking is not required."""

    def read(self, size: int = -1, /) -> bytes:
        """Return up to `size` bytes, or all remaining bytes if negative."""
        ...

    def readline(self, size: int = -1, /) -> bytes:
        """Return one line, including its ending, with an optional byte limit."""
        ...


@dataclass(frozen=True, slots=True)
class FileInfo:
    """File metadata required for write permission decisions."""

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


class FileWriter(Protocol):
    """Inspect and write files in a storage namespace.

    Calls are blocking and run in worker threads. Paths are relative POSIX
    paths for injected storage. Use standard `OSError` subclasses for failures.
    Metadata must describe the same namespace that `write` changes.
    """

    def stat(self, path: str) -> FileInfo | None:
        """Return file metadata, or `None` for a missing path.

        Object stores can report implicit directories as directories. The root
        `.` must exist. This operation must not create files or directories.
        """
        ...

    def write(self, path: str, content: bytes) -> None:
        """Create or replace a file, creating its parent directories as needed."""
        ...


class DirectoryLister(Protocol):
    """List direct children in the same namespace as a reader."""

    def list_directory(self, path: str) -> Sequence[DirectoryEntry]:
        """Return children and identify directory links to prevent cycles.

        Names must be single path components. Raise `FileNotFoundError` for
        missing directories and `NotADirectoryError` for files.
        """
        ...


class SkillReader(FileReader, DirectoryLister, Protocol):
    """Read and list files for the SDK's native skill loader."""
