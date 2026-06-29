"""`SKILL.md` frontmatter parsing for harness skills."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .._frontmatter import (
    coerce_optional_string,
    coerce_required_string,
    load_frontmatter,
    parse_optional_tool_names,
    parse_tool_names,
)
from ._constraints import (
    SKILL_MAX_COMPATIBILITY_LENGTH,
    SKILL_MAX_DESCRIPTION_LENGTH,
    SKILL_MAX_FILE_LINES,
)
from ._types import SkillManifest


@dataclass(frozen=True, slots=True)
class ParsedSkillFile:
    """Parsed `SKILL.md` contents used by the filesystem skill loader."""

    manifest: SkillManifest
    """Discovered skill manifest."""

    instructions: str
    """Markdown body after frontmatter stripping."""


def parse_skill_file(path: Path) -> ParsedSkillFile | None:
    """Parse one `SKILL.md` file into a manifest and instructions body.

    Best-effort for filesystem discovery: returns `None` to skip a file that
    cannot be read, exceeds the size limit, has no parseable frontmatter, or is
    missing a required `name`/`description`, so one bad file never breaks
    discovery of the rest.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    if _exceeds_file_size_limit(text):
        return None

    loaded = load_frontmatter(text)
    if loaded is None:
        return None

    frontmatter, body = loaded
    root = path.parent.resolve()
    skill_file = path.resolve()

    name = coerce_required_string(frontmatter, "name")
    if not name:
        return None

    description = coerce_required_string(frontmatter, "description")
    if not description:
        return None

    manifest = SkillManifest(
        name=name,
        description=description[:SKILL_MAX_DESCRIPTION_LENGTH],
        skill_file=skill_file,
        root=root,
        license=coerce_optional_string(frontmatter, "license"),
        compatibility=_truncate_compatibility(
            coerce_optional_string(frontmatter, "compatibility")
        ),
        metadata=_normalize_metadata(frontmatter.get("metadata")),
        tools=parse_optional_tool_names(frontmatter, key="tools"),
        disallowed_tools=parse_tool_names(frontmatter.get("disallowedTools")),
    )

    return ParsedSkillFile(manifest=manifest, instructions=body.strip())


def _exceeds_file_size_limit(text: str) -> bool:
    """Return whether the skill file exceeds the configured line guidance."""
    return len(text.splitlines()) > SKILL_MAX_FILE_LINES


def _truncate_compatibility(compatibility: str | None) -> str | None:
    """Cap the optional compatibility field at its configured length."""
    if not compatibility:
        return None

    return compatibility[:SKILL_MAX_COMPATIBILITY_LENGTH]


def _normalize_metadata(raw_metadata: object) -> dict[str, str] | None:
    """Normalize the optional metadata mapping to string keys and values."""
    if raw_metadata is None:
        return None

    if not isinstance(raw_metadata, dict):
        return None

    normalized: dict[str, str] = {}
    for key, value in cast(dict[object, object], raw_metadata).items():
        normalized[str(key)] = str(value)

    return normalized
