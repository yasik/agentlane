"""Loader interfaces for harness skills."""

from collections.abc import Sequence
from typing import Protocol

from ._types import LoadedSkill, SkillManifest


class SkillLoader(Protocol):
    """Public loader contract for discovering and loading skills."""

    async def discover(self) -> Sequence[SkillManifest]:
        """Return the discovered skill manifests."""
        ...

    async def load(self, name: str) -> LoadedSkill:
        """Load one named skill and return its activated payload."""
        ...
