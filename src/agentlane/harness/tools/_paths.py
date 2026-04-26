"""Shared path resolution for filesystem-oriented harness tools."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ToolPathResolver:
    """Resolve tool paths relative to a construction-time working directory."""

    cwd: Path = field(default_factory=Path.cwd)

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
