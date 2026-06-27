"""Pluggable filesystem seam the skill loader and read tool operate over.

The skills subsystem only needs to list directories and read files. Capturing
that behind a small `SkillFilesystem` protocol lets the same loader and read
tool serve skills stored anywhere a path tree can be exposed — the local disk is
one implementation (`LocalSkillFilesystem`), a remote object store or a
versioned content service is another.

The contract is intentionally storage-agnostic: it knows nothing about skills,
only files and directories addressed by a `root` (a named tree, such as a local
directory or a remote repository) and a POSIX-style path relative to that root.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class SkillFilesystemEntry(Protocol):
    """One immediate child of a directory listing."""

    @property
    def name(self) -> str:
        """Entry name within its parent directory, not a full path."""
        ...

    @property
    def is_dir(self) -> bool:
        """Whether the entry is itself a directory."""
        ...


@runtime_checkable
class SkillFilesystem(Protocol):
    """A path-based store of files and directories addressed under named roots.

    A `root` namespaces a tree of paths (a local directory, a remote repository,
    ...). Paths are POSIX-style and relative to their root; the empty string
    addresses the root itself.
    """

    async def read_bytes(self, root: str, path: str, /) -> bytes:
        """Return the raw bytes of one file.

        Raises:
            FileNotFoundError: When the path does not exist.
            IsADirectoryError: When the path names a directory.
        """
        ...

    async def list_dir(self, root: str, path: str, /) -> Sequence[SkillFilesystemEntry]:
        """Return the immediate children of one directory.

        Raises:
            FileNotFoundError: When the path does not exist.
            NotADirectoryError: When the path names a file.
        """
        ...


class LocalSkillFilesystem(SkillFilesystem):
    """Default `SkillFilesystem` backed by the local disk through `pathlib`.

    Each root is a local directory path; relative paths join onto it. This is
    the implementation used when no filesystem is supplied, so the local skill
    loading path is unchanged.
    """

    async def read_bytes(self, root: str, path: str, /) -> bytes:
        """Read one file's bytes, surfacing the path-shape errors of the contract."""
        return _local_target(root, path).read_bytes()

    async def list_dir(self, root: str, path: str, /) -> Sequence[SkillFilesystemEntry]:
        """List one directory's immediate children in filesystem order."""
        target = _local_target(root, path)
        if not target.exists():
            raise FileNotFoundError(str(target))
        if not target.is_dir():
            raise NotADirectoryError(str(target))
        return tuple(_LocalEntry(child) for child in target.iterdir())


class _LocalEntry(SkillFilesystemEntry):
    """A `SkillFilesystemEntry` over one local path."""

    __slots__ = ("_path",)

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def name(self) -> str:
        return self._path.name

    @property
    def is_dir(self) -> bool:
        return self._path.is_dir()


def _local_target(root: str, path: str) -> Path:
    """Join one root-relative path onto a local root, where an empty path is the root."""
    base = Path(root)
    return base / path if path else base
