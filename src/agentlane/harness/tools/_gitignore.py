"""Shared gitignore matching helpers for filesystem-oriented harness tools."""

from dataclasses import dataclass
from pathlib import Path

from pathspec import GitIgnoreSpec


@dataclass(frozen=True, slots=True)
class _ScopedGitignoreSpec:
    """One `.gitignore` spec scoped to the directory that declared it."""

    base_dir: Path
    spec: GitIgnoreSpec


@dataclass(frozen=True, slots=True)
class GitignoreMatcher:
    """Deterministic matcher for paths skipped by repository ignore rules."""

    root: Path
    specs: tuple[_ScopedGitignoreSpec, ...]

    @classmethod
    def from_path(cls, path: str | Path) -> "GitignoreMatcher":
        """Build a matcher from a search path and its ancestor gitignore files."""
        root = Path(path).expanduser().resolve(strict=False)
        search_root = root if root.is_dir() else root.parent

        return cls(
            root=search_root,
            specs=tuple(_discover_gitignore_specs(search_root)),
        )

    def is_ignored(self, path: str | Path, *, is_dir: bool | None = None) -> bool:
        """Return whether a path should be skipped by `.gitignore` rules."""
        raw_path = Path(path).expanduser()
        if not raw_path.is_absolute():
            raw_path = self.root / raw_path

        resolved_path = raw_path.resolve(strict=False)
        if _has_git_dir_part(resolved_path):
            return True

        path_is_dir = resolved_path.is_dir() if is_dir is None else is_dir
        for scoped_spec in self.specs:
            try:
                relative_path = resolved_path.relative_to(scoped_spec.base_dir)
            except ValueError:
                continue

            match_path = relative_path.as_posix()
            if path_is_dir and match_path:
                match_path = f"{match_path}/"
            if match_path and scoped_spec.spec.match_file(match_path):
                return True

        return False


def _discover_gitignore_specs(search_root: Path) -> list[_ScopedGitignoreSpec]:
    """Return gitignore specs from the search root up to the repo boundary."""
    specs: list[_ScopedGitignoreSpec] = []
    for directory in _directories_to_repo_boundary(search_root):
        gitignore = directory / ".gitignore"
        if not gitignore.is_file():
            continue

        lines = gitignore.read_text(encoding="utf-8").splitlines()
        specs.append(
            _ScopedGitignoreSpec(
                base_dir=directory.resolve(strict=False),
                spec=GitIgnoreSpec.from_lines(lines),
            )
        )
    return specs


def _directories_to_repo_boundary(search_root: Path) -> tuple[Path, ...]:
    """Return ancestor directories from repository root to search root."""
    directories: list[Path] = []
    current = search_root.resolve(strict=False)
    while True:
        directories.append(current)
        if (current / ".git").exists():
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    return tuple(reversed(directories))


def _has_git_dir_part(path: Path) -> bool:
    """Return whether a resolved path is inside a `.git` directory."""
    return ".git" in path.parts
