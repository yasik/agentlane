"""Route file access through named backends in one logical namespace."""

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import PurePosixPath

from agentlane.io import Writer, write_all

from ._paths import normalize_relative_path, normalize_virtual_path
from ._types import (
    BinaryReader,
    DirectoryEntry,
    FileInfo,
    FileStat,
    FileWriter,
    SkillReader,
)


class MountedReader:
    """Combine local or remote readers in a rooted virtual POSIX namespace.

    Share this reader between the skill loader and the native read tool. Each
    mount name selects one reader; that reader receives the remaining path.
    Calls are synchronous, and each child must support concurrent worker calls.
    Child readers own physical access restrictions, timeouts, and stream cleanup.
    Absolute paths start at the virtual root, not the host filesystem root.
    Direct relative paths also start at the virtual root. Tools use `resolve_path`
    to resolve relative arguments from their own working directories.

    Args:
        mounts: Named readers that support file reads and directory listings.
            Names must be canonical single relative POSIX path components.
            The mapping is copied at construction. An empty mapping is valid.

    Raises:
        ValueError: If a mount name is invalid.
    """

    def __init__(self, mounts: Mapping[str, SkillReader]) -> None:
        self._mounts = dict(mounts)

        # Require canonical keys so normalization cannot change which mount wins.
        for name in self._mounts:
            path = normalize_relative_path(name)
            if len(path.parts) != 1 or path.as_posix() != name:
                raise ValueError(
                    "Mount names must contain one canonical path component"
                )

    def resolve_path(self, path: str, *, cwd: str = "/") -> PurePosixPath:
        """Resolve a logical path without accessing or selecting a backend."""
        return normalize_virtual_path(path, cwd=cwd)

    def open_read(self, path: str) -> AbstractContextManager[BinaryReader]:
        """Open a file through the selected reader's context manager.

        Unknown mounts raise `FileNotFoundError`. The logical root and named
        mount roots raise `IsADirectoryError`. Child errors propagate without
        trying another reader.
        """
        normalized = self.resolve_path(path)
        if normalized == PurePosixPath("/"):
            raise IsADirectoryError(path)

        reader, child_path = self._resolve(normalized)
        if child_path == ".":
            raise IsADirectoryError(path)

        return reader.open_read(child_path)

    def list_directory(self, path: str) -> Sequence[DirectoryEntry]:
        """List named mounts at `/` or delegate to one child directory.

        Mount names are sorted. A mount root delegates to its reader at `.`.
        Unknown mounts raise `FileNotFoundError`; child listing errors propagate.
        """
        normalized = self.resolve_path(path)
        if normalized == PurePosixPath("/"):
            return tuple(
                DirectoryEntry(name=name, is_directory=True)
                for name in sorted(self._mounts)
            )

        reader, child_path = self._resolve(normalized)
        return reader.list_directory(child_path)

    def stat(self, path: str) -> FileInfo | None:
        """Inspect a mounted path without consulting the host filesystem."""
        normalized = self.resolve_path(path)

        # The router owns these directories; they need no physical host paths.
        if normalized == PurePosixPath("/"):
            return FileInfo(is_directory=True)

        try:
            reader, child_path = self._resolve(normalized)
        except FileNotFoundError:
            return None

        if child_path == ".":
            return FileInfo(is_directory=True)

        if isinstance(reader, FileStat):
            return reader.stat(child_path)

        # Read/list-only adapters can still provide existence and directory type.
        # Use the child's namespace; host stat would inspect an unrelated path.
        child = PurePosixPath(child_path)
        try:
            entries = reader.list_directory(child.parent.as_posix())
        except (FileNotFoundError, NotADirectoryError):
            return None

        for entry in entries:
            if entry.name == child.name:
                return FileInfo(is_directory=entry.is_directory)

        return None

    def _resolve(self, path: PurePosixPath) -> tuple[SkillReader, str]:
        name = path.parts[1]
        reader = self._mounts.get(name)
        if reader is None:
            raise FileNotFoundError(str(path))

        child_path = normalize_relative_path(path.relative_to(PurePosixPath("/", name)))
        return reader, child_path.as_posix()


class MountedFileSystem(MountedReader):
    """Read, list, and write through named storage mounts.

    Readers that implement `FileWriter` accept writes. Other mounts remain
    read-only. Mount roots cannot be replaced, and unknown mounts are never
    created. The child backend owns atomic replacement and concurrency control.
    """

    def write(self, path: str, content: bytes) -> None:
        """Convenience operation over the selected mount's writer."""
        with self.open_write(path) as stream:
            write_all(stream, content)

    def open_write(self, path: str) -> AbstractContextManager[Writer]:
        """Open a writable mount; reject read-only mounts and mount roots."""
        normalized = self.resolve_path(path)
        if normalized == PurePosixPath("/"):
            raise IsADirectoryError(path)

        reader, child_path = self._resolve(normalized)
        if child_path == ".":
            raise IsADirectoryError(path)

        # A read-only mount must fail here rather than write through another backend.
        if not isinstance(reader, FileWriter):
            raise PermissionError(path)

        return reader.open_write(child_path)
