"""Default skill loader over a pluggable `SkillFilesystem`."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ._discovery import default_skill_roots
from ._filesystem import LocalSkillFilesystem, SkillFilesystem
from ._loader import SkillLoader
from ._parser import ParsedSkillFile, parse_skill_text
from ._types import LoadedSkill, SkillManifest, SkillResource

_SKILL_FILE = "SKILL.md"
_PREFERRED_RESOURCE_DIRECTORIES = {
    "scripts": 0,
    "references": 1,
    "assets": 2,
}


@dataclass(frozen=True, slots=True)
class _DiscoveredSkill:
    """One parsed skill plus the filesystem location it was discovered under."""

    root: str
    """Root the skill was discovered under, addressed by the filesystem."""

    skill_dir: str
    """Skill directory relative to its root."""

    parsed: ParsedSkillFile
    """Parsed manifest and instructions body."""


class FilesystemSkillLoader(SkillLoader):
    """Discover and load skills from a `SkillFilesystem`, across ordered roots.

    Storage is pluggable through `filesystem`; the discovery, parsing policy,
    first-wins de-duplication, and resource ordering are identical regardless of
    where the skills live. The default `LocalSkillFilesystem` reads the local
    disk, so omitting `filesystem` preserves the previous local-only behavior.
    """

    def __init__(
        self,
        *,
        roots: Sequence[str | Path] | None = None,
        include_default_roots: bool = True,
        filesystem: SkillFilesystem | None = None,
    ) -> None:
        """Initialize the loader.

        Args:
            roots: Roots to search, in precedence order (an earlier root wins on
                a name clash). For the default local filesystem these are local
                directories normalized to absolute paths; for a custom
                filesystem they are that store's opaque root identifiers.
            include_default_roots: Whether to also search the standard local
                skill roots (`./.agents/skills`, `~/.agents/skills`). Only the
                default local filesystem has standard roots, so this is ignored
                when a custom `filesystem` is supplied.
            filesystem: Storage the skills are read from. Defaults to
                `LocalSkillFilesystem`, which reads the local disk.
        """
        if filesystem is None:
            self._fs: SkillFilesystem = LocalSkillFilesystem()
            self._roots = tuple(
                str(root)
                for root in _resolve_local_roots(
                    roots=roots,
                    include_default_roots=include_default_roots,
                )
            )
        else:
            self._fs = filesystem
            self._roots = tuple(str(root) for root in roots or ())
        self._discovered_by_name: dict[str, _DiscoveredSkill] = {}

    async def discover(self) -> Sequence[SkillManifest]:
        """Discover valid skills across all roots, in precedence order."""
        discovered: dict[str, _DiscoveredSkill] = {}
        manifests: list[SkillManifest] = []

        for root in self._roots:
            try:
                entries = await self._fs.list_dir(root, "")
            except (FileNotFoundError, NotADirectoryError):
                continue

            for entry in sorted(entries, key=lambda item: item.name):
                if not entry.is_dir:
                    continue

                parsed = await self._parse_skill(root, entry.name)
                if parsed is None:
                    continue

                if parsed.manifest.name in discovered:
                    continue

                discovered[parsed.manifest.name] = _DiscoveredSkill(
                    root=root,
                    skill_dir=entry.name,
                    parsed=parsed,
                )
                manifests.append(parsed.manifest)

        self._discovered_by_name = discovered
        return tuple(manifests)

    async def load(self, name: str) -> LoadedSkill:
        """Load one discovered skill by name, enumerating its bundled resources."""
        if name not in self._discovered_by_name:
            await self.discover()

        discovered = self._discovered_by_name.get(name)
        if discovered is None:
            raise KeyError(name)

        return LoadedSkill(
            manifest=discovered.parsed.manifest,
            instructions=discovered.parsed.instructions,
            resources=await self._list_resources(
                discovered.root,
                discovered.skill_dir,
            ),
        )

    async def _parse_skill(self, root: str, skill_dir: str) -> ParsedSkillFile | None:
        """Read and parse one `SKILL.md`, skipping unreadable or invalid files."""
        try:
            raw = await self._fs.read_bytes(root, f"{skill_dir}/{_SKILL_FILE}")
        except (FileNotFoundError, IsADirectoryError):
            return None

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None

        skill_root = Path(root) / skill_dir
        return parse_skill_text(
            text,
            skill_file=skill_root / _SKILL_FILE,
            root=skill_root,
        )

    async def _list_resources(
        self, root: str, skill_dir: str
    ) -> tuple[SkillResource, ...]:
        """Enumerate bundled skill resources lazily on activation."""
        relative_paths: list[str] = []
        await self._collect_files(root, skill_dir, "", relative_paths)
        resources = [path for path in relative_paths if path != _SKILL_FILE]
        return tuple(
            SkillResource(path=path)
            for path in sorted(resources, key=_resource_sort_key)
        )

    async def _collect_files(
        self, root: str, skill_dir: str, sub_path: str, found: list[str]
    ) -> None:
        """Walk one skill subtree, appending file paths relative to the skill directory."""
        listing = f"{skill_dir}/{sub_path}" if sub_path else skill_dir
        for entry in await self._fs.list_dir(root, listing):
            relative = f"{sub_path}/{entry.name}" if sub_path else entry.name
            if entry.is_dir:
                await self._collect_files(root, skill_dir, relative, found)
            else:
                found.append(relative)


def _resource_sort_key(path: str) -> tuple[int, str]:
    """Order resources by preferred top-level directory, then path."""
    parts = PurePosixPath(path).parts
    top_level_directory = parts[0] if parts else ""
    return (
        _PREFERRED_RESOURCE_DIRECTORIES.get(
            top_level_directory, len(_PREFERRED_RESOURCE_DIRECTORIES)
        ),
        path,
    )


def _resolve_local_roots(
    *,
    roots: Sequence[str | Path] | None,
    include_default_roots: bool,
) -> tuple[Path, ...]:
    """Normalize configured and default local skill roots to absolute paths."""
    resolved_roots: list[Path] = []
    seen: set[Path] = set()

    configured_roots = tuple(Path(root).expanduser().resolve() for root in roots or ())
    for root in configured_roots:
        if root in seen:
            continue

        seen.add(root)
        resolved_roots.append(root)

    if include_default_roots:
        for root in default_skill_roots():
            if root in seen:
                continue

            seen.add(root)
            resolved_roots.append(root)

    return tuple(resolved_roots)
