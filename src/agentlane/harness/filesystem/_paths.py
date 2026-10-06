"""Lexical paths for storage that does not use the host filesystem."""

import posixpath
from pathlib import PurePath, PurePosixPath, PureWindowsPath


def normalize_virtual_path(path: str, *, cwd: str = "/") -> PurePosixPath:
    """Resolve a path from a working directory in a virtual POSIX namespace.

    Absolute paths start at the virtual root. Relative paths start at `cwd`.
    A relative `cwd` starts at the virtual root. This function does not access
    the host filesystem or expand home directories.

    Args:
        path: File or directory path in the virtual namespace.
        cwd: Working directory from which to resolve a relative path.

    Returns:
        Canonical absolute path in the virtual namespace.

    Raises:
        ValueError: If either path is empty, contains a null byte, backslash,
            drive or home prefix, or traverses above the virtual root.
    """
    cwd_parts = _virtual_parts(cwd)
    path_parts = _virtual_parts(path)
    resolved_cwd = _collapse_virtual_parts([], cwd_parts)
    base = [] if path.startswith("/") else resolved_cwd
    resolved = _collapse_virtual_parts(base, path_parts)
    return PurePosixPath("/", *resolved)


def _virtual_parts(path: str) -> list[str]:
    if not path.strip():
        raise ValueError("path must not be empty")
    if "\x00" in path or "\\" in path:
        raise ValueError("path must use POSIX separators and contain no null bytes")

    parts = path.split("/")
    for part in parts:
        if PureWindowsPath(part).drive:
            raise ValueError("path must not contain drive prefixes")
        if part.startswith("~"):
            raise ValueError("path must not contain home-directory prefixes")

    return parts


def _collapse_virtual_parts(base: list[str], parts: list[str]) -> list[str]:
    resolved = base.copy()
    for part in parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not resolved:
                raise ValueError("path must stay within the virtual root")
            resolved.pop()
        else:
            resolved.append(part)

    return resolved


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
