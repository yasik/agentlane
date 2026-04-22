"""Typed skill primitives for the harness skills package."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SkillManifest:
    """Discovered metadata for one skill plus its canonical file locations."""

    name: str
    """Stable skill name from `SKILL.md`."""

    description: str
    """Short description of what the skill does and when to use it."""

    skill_file: Path
    """Absolute path to the root `SKILL.md` file."""

    root: Path
    """Absolute path to the skill root directory."""

    license: str | None = None
    """Optional license string from the frontmatter."""

    compatibility: str | None = None
    """Optional compatibility note from the frontmatter."""

    metadata: dict[str, str] | None = None
    """Optional extra metadata from the frontmatter."""

    allowed_tools: str | None = None
    """Optional pre-approved tools string from the frontmatter."""


@dataclass(frozen=True, slots=True)
class SkillResource:
    """One bundled file that belongs to an activated skill."""

    path: str
    """Path to the resource file relative to the skill directory."""


@dataclass(frozen=True, slots=True)
class LoadedSkill:
    """Activated skill payload returned by a skill loader."""

    manifest: SkillManifest
    """Discovered manifest for the skill."""

    instructions: str
    """Rendered `SKILL.md` body without frontmatter."""

    resources: tuple[SkillResource, ...]
    """Bundled resources exposed lazily on activation."""
