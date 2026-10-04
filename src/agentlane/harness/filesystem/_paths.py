"""Lexical paths for storage that does not use the host filesystem."""

import posixpath
from pathlib import PurePath, PurePosixPath, PureWindowsPath


def normalize_relative_path(
    path: str | PurePath,
    *,
    root: str | PurePath = ".",
) -> PurePosixPath:
    """Resolve a POSIX path relative to a storage root without filesystem I/O.

    Args:
        path: Relative file or directory path.
        root: Relative directory from which to resolve `path`.

    Returns:
        Normalized path in the storage namespace.

    Raises:
        ValueError: If a path is empty, absolute, contains a null byte or
            backslash, has a component that starts with `~`, or leaves the
            storage namespace through `..`.
    """
    path_text = str(path)
    root_text = str(root)
    for value in (path_text, root_text):
        if not value.strip():
            raise ValueError("path must not be empty")
        if "\x00" in value or "\\" in value:
            raise ValueError("path must use POSIX separators and contain no null bytes")
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive:
            raise ValueError("path must be relative to the storage root")
        # Mount routing can expose any component to a local backend's expanduser.
        if any(part.startswith("~") for part in PurePosixPath(value).parts):
            raise ValueError("path must not contain home-directory prefixes")

    normalized_root = posixpath.normpath(root_text)
    normalized = posixpath.normpath(posixpath.join(normalized_root, path_text))
    for value in (normalized_root, normalized):
        if value == ".." or value.startswith("../"):
            raise ValueError("path must stay within the storage root")

    return PurePosixPath(normalized)
