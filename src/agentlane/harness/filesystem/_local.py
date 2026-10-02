"""Default local implementation of file I/O contracts."""

import contextlib
import os
import stat
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from uuid import uuid4

from agentlane.io import Writer, write_all

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

        # Devices and pipes can block indefinitely or have side effects on read.
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
            try:
                metadata = child.stat()
            except (FileNotFoundError, NotADirectoryError):
                # A child can disappear between the directory listing and stat.
                continue

            is_directory = stat.S_ISDIR(metadata.st_mode)
            if not is_directory and not stat.S_ISREG(metadata.st_mode):
                continue

            entries.append(
                DirectoryEntry(
                    name=child.name,
                    is_directory=is_directory,
                    is_symlink=child.is_symlink(),
                    modified_time=metadata.st_mtime,
                )
            )

        return tuple(entries)

    def write(self, path: str, content: bytes) -> None:
        """Convenience operation over the streaming writer."""
        with self.open_write(path) as stream:
            write_all(stream, content)

    @contextmanager
    def open_write(self, path: str) -> Generator[Writer, None, None]:
        """Replace a file only after the stream closes successfully.

        Failed writes remove the temporary file and preserve the target. Parent
        directories are created as needed. Existing mode bits are preserved.
        """
        target = self._resolve(path)
        try:
            metadata = target.stat()
        except FileNotFoundError:
            metadata = None

        if metadata is not None and stat.S_ISDIR(metadata.st_mode):
            raise IsADirectoryError(str(target))

        mode = stat.S_IMODE(metadata.st_mode) if metadata is not None else None
        target.parent.mkdir(parents=True, exist_ok=True)

        # A sibling permits atomic replacement on the same filesystem. Its name
        # stays short even when the target already uses the maximum name length.
        temporary = target.with_name(f".agentlane-{uuid4().hex}.tmp")

        # Exclusive creation retains normal new-file permissions under the umask.
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                if mode is not None:
                    # Replacement installs a new inode, so copy the target mode.
                    temporary.chmod(mode)

                yield stream

            # An exception from the caller skips this commit and keeps the target.
            temporary.replace(target)
        finally:
            # Cleanup must not replace the original write or commit error.
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)

    def _resolve(self, path: str) -> Path:
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = self._root / target

        return target.resolve(strict=False)
