"""Catalog container for discovered harness skills."""

from collections.abc import Iterator, Sequence

from ._loader import SkillLoader
from ._types import LoadedSkill, SkillManifest


class SkillCatalog:
    """Developer-facing container for discovered skills."""

    def __init__(
        self,
        *,
        manifests: Sequence[SkillManifest],
        loader: SkillLoader,
    ) -> None:
        self._manifests = tuple(manifests)
        self._loader = loader
        self._by_name = {manifest.name: manifest for manifest in self._manifests}

    def __iter__(self) -> Iterator[SkillManifest]:
        """Iterate discovered skill manifests in stable order."""
        return iter(self._manifests)

    def __len__(self) -> int:
        """Return the number of discovered skills."""
        return len(self._manifests)

    def get(self, name: str) -> SkillManifest | None:
        """Return the manifest for one skill name, if present."""
        return self._by_name.get(name)

    def has(self, name: str) -> bool:
        """Return whether the named skill exists."""
        return name in self._by_name

    def names(self) -> tuple[str, ...]:
        """Return discovered skill names in stable order."""
        return tuple(manifest.name for manifest in self._manifests)

    async def load(self, name: str) -> LoadedSkill:
        """Load one named skill through the catalog's loader."""
        if name not in self._by_name:
            raise KeyError(name)
        return await self._loader.load(name)
