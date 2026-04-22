"""Default filesystem-backed skill loader."""

from collections.abc import Sequence
from pathlib import Path

from ._discovery import default_skill_roots
from ._loader import SkillLoader
from ._parser import parse_skill_file
from ._types import LoadedSkill, SkillManifest, SkillResource


class FilesystemSkillLoader(SkillLoader):
    """Default filesystem-backed implementation of `SkillLoader`."""

    def __init__(
        self,
        *,
        roots: Sequence[str | Path] | None = None,
        include_default_roots: bool = True,
    ) -> None:
        self._roots = _resolve_roots(
            roots=roots,
            include_default_roots=include_default_roots,
        )

    async def discover(self) -> Sequence[SkillManifest]:
        """Discover valid skills from the configured filesystem roots."""
        manifests: list[SkillManifest] = []
        seen_names: set[str] = set()

        for root in self._roots:
            if not root.exists() or not root.is_dir():
                continue

            for child in sorted(root.iterdir(), key=lambda path: path.name):
                if not child.is_dir():
                    continue

                skill_file = child / "SKILL.md"
                if not skill_file.is_file():
                    continue

                parsed = parse_skill_file(skill_file)
                if parsed is None:
                    continue

                if parsed.manifest.name in seen_names:
                    continue

                seen_names.add(parsed.manifest.name)
                manifests.append(parsed.manifest)

        return tuple(manifests)

    async def load(self, name: str) -> LoadedSkill:
        """Load one discovered skill by name."""
        for root in self._roots:
            if not root.exists() or not root.is_dir():
                continue

            for child in sorted(root.iterdir(), key=lambda path: path.name):
                if not child.is_dir():
                    continue

                skill_file = child / "SKILL.md"
                if not skill_file.is_file():
                    continue

                parsed = parse_skill_file(skill_file)
                if parsed is None:
                    continue

                if parsed.manifest.name != name:
                    continue

                return LoadedSkill(
                    manifest=parsed.manifest,
                    instructions=parsed.instructions,
                    resources=_list_skill_resources(parsed.manifest.root),
                )

        raise KeyError(name)


def _resolve_roots(
    *,
    roots: Sequence[str | Path] | None,
    include_default_roots: bool,
) -> tuple[Path, ...]:
    """Normalize configured and default skill roots to absolute paths."""
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


def _list_skill_resources(root: Path) -> tuple[SkillResource, ...]:
    """Enumerate bundled skill resources lazily on activation."""
    preferred_directories = {
        "scripts": 0,
        "references": 1,
        "assets": 2,
    }
    skill_file = root / "SKILL.md"
    files = [path for path in root.rglob("*") if path.is_file() and path != skill_file]

    def sort_key(path: Path) -> tuple[int, str]:
        relative_path = path.relative_to(root)
        top_level_directory = relative_path.parts[0] if relative_path.parts else ""
        return (
            preferred_directories.get(top_level_directory, len(preferred_directories)),
            relative_path.as_posix(),
        )

    return tuple(
        SkillResource(path=path.relative_to(root).as_posix())
        for path in sorted(files, key=sort_key)
    )
