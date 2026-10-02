"""Shared gitignore matching helpers for filesystem-oriented harness tools."""

from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Self

from pathspec import GitIgnoreSpec

from agentlane.harness.filesystem import LocalFileSystem, ReadableFileSystem
from agentlane.io import read_all


@dataclass(frozen=True, slots=True)
class _ScopedGitignoreSpec:
    """One `.gitignore` spec scoped to the directory that declared it."""

    base_dir: PurePath
    spec: GitIgnoreSpec


@dataclass(frozen=True, slots=True)
class GitignoreMatcher:
    """Deterministic matcher for paths skipped by repository ignore rules."""

    root: PurePath
    specs: tuple[_ScopedGitignoreSpec, ...]
    filesystem: ReadableFileSystem

    @classmethod
    def from_path(
        cls, path: str | PurePath, *, filesystem: ReadableFileSystem | None = None
    ) -> Self:
        """Build a matcher from a search path and its ancestor gitignore files."""
        fs = filesystem if filesystem is not None else LocalFileSystem()
        root = (
            Path(path).expanduser().resolve(strict=False)
            if filesystem is None
            else path
        )
        if isinstance(root, str):
            raise TypeError("injected filesystem paths must be PurePath values")

        info = fs.stat(str(root))
        search_root = root if info is not None and info.is_directory else root.parent

        return cls(
            root=search_root,
            specs=tuple(_discover_gitignore_specs(search_root, fs)),
            filesystem=fs,
        )

    def is_ignored(self, path: str | PurePath, *, is_dir: bool | None = None) -> bool:
        """Return whether a path should be skipped by `.gitignore` rules."""
        # Preserve logical paths for injected storage; only local paths may use
        # host expansion and symlink resolution.
        raw_path = type(self.root)(path)
        if isinstance(raw_path, Path):
            raw_path = raw_path.expanduser()

        if not raw_path.is_absolute() and isinstance(self.root, Path):
            raw_path = self.root / raw_path

        resolved_path = (
            raw_path.resolve(strict=False) if isinstance(raw_path, Path) else raw_path
        )

        if _has_git_dir_part(resolved_path):
            return True

        if is_dir is None:
            info = self.filesystem.stat(str(resolved_path))
            path_is_dir = info is not None and info.is_directory
        else:
            # Traversal already knows the entry type, so it needs no extra stat.
            path_is_dir = is_dir

        for scoped_spec in self.specs:
            try:
                relative_path = resolved_path.relative_to(scoped_spec.base_dir)
            except ValueError:
                continue

            match_path = relative_path.as_posix()
            if path_is_dir and match_path:
                # The trailing slash lets pathspec apply directory-only rules.
                match_path = f"{match_path}/"
            if match_path and scoped_spec.spec.match_file(match_path):
                return True

        return False


def _discover_gitignore_specs(
    search_root: PurePath, filesystem: ReadableFileSystem
) -> list[_ScopedGitignoreSpec]:
    """Return gitignore specs from the search root up to the repo boundary."""
    specs: list[_ScopedGitignoreSpec] = []
    for directory in _directories_to_repo_boundary(search_root, filesystem):
        gitignore = directory / ".gitignore"
        info = filesystem.stat(str(gitignore))
        if info is None or info.is_directory:
            continue

        with filesystem.open_read(str(gitignore)) as stream:
            lines = read_all(stream).decode("utf-8").splitlines()

        specs.append(
            _ScopedGitignoreSpec(
                base_dir=directory,
                spec=GitIgnoreSpec.from_lines(lines),
            )
        )

    return specs


def _directories_to_repo_boundary(
    search_root: PurePath, filesystem: ReadableFileSystem
) -> tuple[PurePath, ...]:
    """Return ancestor directories from repository root to search root."""
    directories: list[PurePath] = []
    current = search_root
    while True:
        directories.append(current)

        # A .git file also marks a boundary, as in a Git worktree.
        if filesystem.stat(str(current / ".git")) is not None:
            break

        parent = current.parent
        if parent == current:
            break

        current = parent

    return tuple(reversed(directories))


def _has_git_dir_part(path: PurePath) -> bool:
    """Return whether a resolved path is inside a `.git` directory."""
    return ".git" in path.parts
