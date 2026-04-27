"""Shared output limits and truncation helpers for harness tools."""

from dataclasses import dataclass
from pathlib import Path

TEXT_MAX_LINES = 2000
"""Maximum number of text-file lines returned by default."""

TEXT_MAX_BYTES = 50 * 1024
"""Maximum number of text-file bytes returned by default."""

BASH_MAX_LINES = 2000
"""Maximum number of bash output lines returned by default."""

BASH_MAX_BYTES = 50 * 1024
"""Maximum number of bash output bytes returned by default."""

GREP_DEFAULT_LIMIT = 100
"""Default maximum number of grep matches returned."""

GREP_MAX_LINE_LENGTH = 500
"""Maximum length of one grep result line."""

FIND_DEFAULT_LIMIT = 1000
"""Default maximum number of find results returned."""

LS_DEFAULT_LIMIT = 500
"""Default maximum number of ls entries returned."""


@dataclass(frozen=True, slots=True)
class TruncatedOutput:
    """Rendered output and whether configured limits were applied."""

    text: str
    truncated: bool


def truncate_output(
    text: str,
    *,
    max_lines: int,
    max_bytes: int,
    tail: bool = False,
) -> TruncatedOutput:
    """Apply deterministic line and byte limits to a tool output string."""
    if max_lines < 1:
        raise ValueError("max_lines must be at least 1.")
    if max_bytes < 1:
        raise ValueError("max_bytes must be at least 1.")

    lines = text.splitlines(keepends=True)
    selected_lines = lines[-max_lines:] if tail else lines[:max_lines]
    line_truncated = len(lines) > max_lines

    selected_text = "".join(selected_lines)
    byte_truncated = len(selected_text.encode("utf-8")) > max_bytes
    if byte_truncated:
        selected_text = _trim_to_utf8_limit(
            selected_text,
            max_bytes=max_bytes,
            tail=tail,
        )

    truncated = line_truncated or byte_truncated
    if not truncated:
        return TruncatedOutput(text=selected_text, truncated=False)

    direction = "last" if tail else "first"
    marker = (
        f"[output truncated: showing {direction} {max_lines} lines or "
        f"{max_bytes} bytes]\n"
    )
    return TruncatedOutput(text=f"{marker}{selected_text}", truncated=True)


def _trim_to_utf8_limit(text: str, *, max_bytes: int, tail: bool) -> str:
    """Trim text to a byte limit without returning invalid UTF-8."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text

    trimmed = encoded[-max_bytes:] if tail else encoded[:max_bytes]
    return trimmed.decode("utf-8", errors="ignore")


def is_likely_binary_file(path: Path, *, sample_size: int = 8192) -> bool:
    """Return whether a file sample contains binary-only markers."""
    with path.open("rb") as file:
        sample = file.read(sample_size)

    if b"\0" in sample:
        return True

    return False
