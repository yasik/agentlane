"""Skill discovery and loading through local or injected file I/O."""

import asyncio
from collections.abc import Sequence
from pathlib import Path, PurePath

from agentlane.harness.filesystem import (
    DirectoryEntry,
    LocalFileSystem,
    SkillReader,
    normalize_relative_path,
)
from agentlane.io import read_all

from ._discovery import default_skill_roots
from ._loader import SkillLoader
from ._parser import ParsedSkillFile, parse_skill_text
from ._types import LoadedSkill, SkillManifest, SkillResource


class FilesystemSkillLoader(SkillLoader):
    """Discover skills through the default filesystem or an injected reader.

    Injected readers own their storage root. Their paths remain relative to
    that root and never pass through local filesystem resolution. Readers must
    permit blocking calls from a worker thread.
    """

    def __init__(
        self,
        *,
        roots: Sequence[str | Path] | None = None,
        include_default_roots: bool = True,
        reader: SkillReader | None = None,
    ) -> None:
        """Set the directories in which to discover skill folders.

        Args:
            roots: Local directories, or relative paths within an injected
                reader. An injected reader defaults to its own root (`.`).
            include_default_roots: Add standard home and working-directory
                skill roots for local I/O only. Ignored for injected readers.
            reader: File reading and directory listing implementation. Omit
                this to use local files with the existing absolute-path rules.
        """
        self._reader = reader if reader is not None else LocalFileSystem()
        self._roots: tuple[PurePath, ...]
        if reader is None:
            self._roots = _resolve_roots(
                roots=roots,
                include_default_roots=include_default_roots,
            )
        else:
            self._roots = tuple(
                dict.fromkeys(
                    normalize_relative_path(root)
                    for root in (roots if roots is not None else (".",))
                )
            )

        self._parsed_by_name: dict[str, ParsedSkillFile] = {}

    async def discover(self) -> Sequence[SkillManifest]:
        """Discover valid skills without blocking the event loop.

        Invalid or unreadable skill files are skipped. Directory listing
        errors propagate, except for absent roots and roots that are files.
        """
        parsed_by_name = await asyncio.to_thread(self._discover)
        self._parsed_by_name = parsed_by_name

        return tuple(parsed.manifest for parsed in parsed_by_name.values())

    def _discover(self) -> dict[str, ParsedSkillFile]:
        parsed_by_name: dict[str, ParsedSkillFile] = {}
        for root in self._roots:
            for child in self._list_directory(root):
                if not child.is_directory:
                    continue

                parsed = self._parse_file(root / child.name / "SKILL.md")
                if parsed is not None:
                    # Root order defines precedence when two skills share a name.
                    parsed_by_name.setdefault(parsed.manifest.name, parsed)

        return parsed_by_name

    async def load(self, name: str) -> LoadedSkill:
        """Load a skill and list its resources without reading resource bytes.

        Cached metadata remains stable after discovery. Resource names are
        listed on activation. A missing skill raises `KeyError`.
        """
        cached = self._parsed_by_name.get(name)
        return await asyncio.to_thread(self._load, name, cached)

    def _load(self, name: str, cached: ParsedSkillFile | None) -> LoadedSkill:
        if cached is None:
            cached = self._discover().get(name)
        if cached is None:
            raise KeyError(name)

        return LoadedSkill(
            manifest=cached.manifest,
            instructions=cached.instructions,
            resources=self._list_skill_resources(cached.manifest.root),
        )

    def _parse_file(self, path: PurePath) -> ParsedSkillFile | None:
        try:
            with self._reader.open_read(str(path)) as stream:
                # A short read is not EOF; parse only after the full transfer.
                text = read_all(stream).decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        location = path.resolve() if isinstance(path, Path) else path
        return parse_skill_text(text, location)

    def _list_directory(self, path: PurePath) -> tuple[DirectoryEntry, ...]:
        try:
            entries = self._reader.list_directory(str(path))
        except (FileNotFoundError, NotADirectoryError):
            return ()

        # Injected listings must stay within their reader namespace. Local
        # listings already use native names, which may contain backslashes or
        # drive-like text on POSIX systems.
        if not isinstance(path, Path):
            for entry in entries:
                normalized = normalize_relative_path(entry.name)
                if len(normalized.parts) != 1 or str(normalized) != entry.name:
                    raise ValueError("Directory entries must contain one child name")

        return tuple(sorted(entries, key=lambda entry: entry.name))

    def _list_skill_resources(self, root: PurePath) -> tuple[SkillResource, ...]:
        files: list[PurePath] = []
        pending = [root]
        while pending:
            directory = pending.pop()
            for child in self._list_directory(directory):
                path = directory / child.name
                if child.is_directory:
                    # Resource discovery must not loop through directory links.
                    if not child.is_symlink:
                        pending.append(path)
                elif path != root / "SKILL.md":
                    files.append(path.relative_to(root))

        return tuple(
            SkillResource(path=path.as_posix())
            for path in sorted(files, key=_resource_sort_key)
        )


def _resource_sort_key(path: PurePath) -> tuple[int, str]:
    preferred_directories = {"scripts": 0, "references": 1, "assets": 2}
    return (
        preferred_directories.get(path.parts[0], len(preferred_directories)),
        path.as_posix(),
    )


def _resolve_roots(
    *,
    roots: Sequence[str | Path] | None,
    include_default_roots: bool,
) -> tuple[Path, ...]:
    """Normalize local roots to absolute paths in first-seen order."""
    configured_roots = tuple(Path(root).expanduser().resolve() for root in roots or ())
    defaults = default_skill_roots() if include_default_roots else ()
    return tuple(dict.fromkeys((*configured_roots, *defaults)))
