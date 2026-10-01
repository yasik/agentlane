"""Default local implementation of file I/O contracts."""

import contextlib
import stat
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path

from ._types import BinaryReader, DirectoryEntry, FileInfo


class LocalFileSystem:
    """Read, list, and write files on the local machine.

    The root is captured at construction. Absolute paths remain supported for
    local tools. The root is a working directory, not a security boundary; use
    tool permission policies to restrict access. Directory listings identify
    symlinks so callers can prevent recursive discovery through cycles.

    Args:
        root: Working directory for relative paths. Defaults to `Path.cwd()`.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = (
            Path(root).expanduser().resolve() if root is not None else Path.cwd()
        )

    def open_read(self, path: str) -> AbstractContextManager[BinaryReader]:
        """Open one local file as a binary stream."""
        target = self._resolve(path)
        mode = target.stat().st_mode
        if stat.S_ISDIR(mode):
            raise IsADirectoryError(str(target))
        if not stat.S_ISREG(mode):
            raise OSError("path is not a regular file")

        return target.open("rb")

    def stat(self, path: str) -> FileInfo | None:
        """Return local metadata, or `None` if the path does not exist."""
        try:
            metadata = self._resolve(path).stat()
        except (FileNotFoundError, NotADirectoryError):
            return None

        return FileInfo(is_directory=stat.S_ISDIR(metadata.st_mode))

    def list_directory(self, path: str) -> tuple[DirectoryEntry, ...]:
        """Return direct children with their directory and link metadata."""
        entries: list[DirectoryEntry] = []
        for child in self._resolve(path).iterdir():
            is_directory = child.is_dir()
            if not is_directory and not child.is_file():
                continue

            entries.append(
                DirectoryEntry(
                    name=child.name,
                    is_directory=is_directory,
                    is_symlink=child.is_symlink(),
                )
            )

        return tuple(entries)

    def write(self, path: str, content: bytes) -> None:
        """Write bytes and atomically replace an existing local file.

        Parent directories are created as needed. A failed replacement leaves
        the existing file intact and removes the temporary file when possible.
        """
        target = self._resolve(path)
        if target.is_dir():
            raise IsADirectoryError(str(target))

        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(content)
            return

        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                delete=False,
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)

            temporary.replace(target)
        finally:
            if temporary is not None:
                with contextlib.suppress(OSError):
                    temporary.unlink(missing_ok=True)

    def _resolve(self, path: str) -> Path:
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = self._root / target

        return target.resolve(strict=False)
