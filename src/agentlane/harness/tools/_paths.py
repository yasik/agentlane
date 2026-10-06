"""Shared path resolution for filesystem-oriented harness tools."""

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Self

from agentlane.harness.filesystem import (
    FilePathResolver,
    FileReader,
    FileWriter,
    normalize_relative_path,
)


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
class StorageToolPathResolver:
    """Resolve paths lexically in an injected storage namespace."""

    cwd: PurePosixPath
    """Working directory in the provider's logical namespace."""

    path_resolver: FilePathResolver | None = None
    """Optional provider path rules; absent providers use relative POSIX paths."""

    @classmethod
    def for_optional(
        cls,
        cwd: str | Path | None = None,
        *,
        filesystem: FileReader | FileWriter,
    ) -> Self:
        """Capture the provider's working directory without host filesystem I/O."""
        directory = str(cwd) if cwd is not None else "."
        if isinstance(filesystem, FilePathResolver):
            return cls(cwd=filesystem.resolve_path(directory), path_resolver=filesystem)

        return cls(cwd=normalize_relative_path(directory))

    def resolve(self, path: str | Path) -> PurePosixPath:
        """Resolve a path relative to the captured working directory."""
        if self.path_resolver is not None:
            return self.path_resolver.resolve_path(str(path), cwd=self.cwd.as_posix())

        return normalize_relative_path(path, root=self.cwd)


def tool_path_guideline(resolver: ToolPathResolver | StorageToolPathResolver) -> str:
    """Describe the same path rules used by this tool instance."""
    prefix = f"Relative paths resolve from the working directory `{resolver.cwd}`."
    if isinstance(resolver, ToolPathResolver):
        return f"{prefix} Absolute paths refer to the local filesystem."
    if resolver.path_resolver is not None:
        return f"{prefix} Rooted paths refer to the storage namespace, not the host filesystem."

    return f"{prefix} Paths must remain relative to the storage root."
