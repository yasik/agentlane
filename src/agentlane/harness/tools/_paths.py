"""Shared path resolution for filesystem-oriented harness tools."""

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Self

from agentlane.harness.filesystem import normalize_relative_path


@dataclass(frozen=True, slots=True)
class ToolPathResolver:
    """Resolve tool paths relative to a construction-time working directory."""

    cwd: Path = field(default_factory=Path.cwd)
    """Local working directory captured when the resolver is constructed."""

    @classmethod
    def for_optional(cls, cwd: str | Path | None = None) -> Self:
        """Return a resolver for an optional caller-provided working directory."""
        if cwd is None:
            return cls()
        return cls(cwd=Path(cwd))

    def __post_init__(self) -> None:
        """Normalize the captured working directory."""
        object.__setattr__(
            self,
            "cwd",
            self.cwd.expanduser().resolve(strict=False),
        )

    def resolve(self, path: str | Path) -> Path:
        """Resolve a relative or absolute path under the configured cwd."""
        if isinstance(path, str) and path.strip() == "":
            raise ValueError("path must not be empty.")

        raw_path = Path(path).expanduser()
        if not raw_path.is_absolute():
            raw_path = self.cwd / raw_path

        return raw_path.resolve(strict=False)


@dataclass(frozen=True, slots=True)
class RelativeToolPathResolver:
    """Resolve paths lexically in an injected storage namespace."""

    cwd: PurePosixPath
    """Working directory relative to the storage root."""

    @classmethod
    def for_optional(cls, cwd: str | Path | None = None) -> Self:
        """Capture a relative working directory without local filesystem I/O."""
        return cls(cwd=normalize_relative_path(cwd if cwd is not None else "."))

    def resolve(self, path: str | Path) -> PurePosixPath:
        """Resolve a path relative to the captured working directory."""
        return normalize_relative_path(path, root=self.cwd)
