"""Read files from named storage backends in one logical namespace."""

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import PurePosixPath

from ._paths import normalize_relative_path
from ._types import BinaryReader, DirectoryEntry, SkillReader


class MountedReader:
    """Combine local or remote readers behind relative POSIX path prefixes.

    Share this reader between the skill loader and the native read tool. Each
    mount name selects one reader; that reader receives the remaining path.
    Calls are synchronous, and each child must support concurrent worker calls.
    Child readers own physical access restrictions, timeouts, and stream cleanup.

    Args:
        mounts: Named readers that support file reads and directory listings.
            Names must be canonical single relative POSIX path components.
            The mapping is copied at construction. An empty mapping is valid.

    Raises:
        ValueError: If a mount name is invalid.
    """

    def __init__(self, mounts: Mapping[str, SkillReader]) -> None:
        self._mounts = dict(mounts)
        for name in self._mounts:
            path = normalize_relative_path(name)
            if len(path.parts) != 1 or path.as_posix() != name:
                raise ValueError(
                    "Mount names must contain one canonical path component"
                )

    def open_read(self, path: str) -> AbstractContextManager[BinaryReader]:
        """Open a file through the selected reader's context manager.

        Unknown mounts raise `FileNotFoundError`. The logical root and named
        mount roots raise `IsADirectoryError`. Child errors propagate without
        trying another reader.
        """
        normalized = normalize_relative_path(path)
        if normalized == PurePosixPath("."):
            raise IsADirectoryError(path)

        reader, child_path = self._resolve(normalized)
        if child_path == ".":
            raise IsADirectoryError(path)

        return reader.open_read(child_path)

    def list_directory(self, path: str) -> Sequence[DirectoryEntry]:
        """List named mounts at `.` or delegate to one child directory.

        Mount names are sorted. A mount root delegates to its reader at `.`.
        Unknown mounts raise `FileNotFoundError`; child listing errors propagate.
        """
        normalized = normalize_relative_path(path)
        if normalized == PurePosixPath("."):
            return tuple(
                DirectoryEntry(name=name, is_directory=True)
                for name in sorted(self._mounts)
            )

        reader, child_path = self._resolve(normalized)
        return reader.list_directory(child_path)

    def _resolve(self, path: PurePosixPath) -> tuple[SkillReader, str]:
        name = path.parts[0]
        reader = self._mounts.get(name)
        if reader is None:
            raise FileNotFoundError(str(path))

        child_path = normalize_relative_path(path.relative_to(name))
        return reader, child_path.as_posix()
