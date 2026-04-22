"""Filesystem discovery helpers for harness skills."""

from pathlib import Path


def default_skill_roots() -> tuple[Path, ...]:
    """Return the standard local skill roots with absolute paths."""
    return (
        (Path.cwd() / ".agents" / "skills").resolve(),
        (Path.home() / ".agents" / "skills").resolve(),
    )
