"""`SKILL.md` frontmatter parsing for harness skills."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from ._constraints import (
    SKILL_MAX_COMPATIBILITY_LENGTH,
    SKILL_MAX_DESCRIPTION_LENGTH,
    SKILL_MAX_FILE_LINES,
    SKILL_MAX_NAME_LENGTH,
)
from ._types import SkillManifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ParsedSkillFile:
    """Parsed `SKILL.md` contents used by the filesystem skill loader."""

    manifest: SkillManifest
    """Discovered skill manifest."""

    instructions: str
    """Markdown body after frontmatter stripping."""


def parse_skill_file(path: Path) -> ParsedSkillFile | None:
    """Parse one `SKILL.md` file into a manifest and instructions body."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        logger.warning("Failed to read skill file %s: %s", path, error)
        return None

    if _exceeds_file_size_limit(text):
        logger.warning(
            "Skipping skill file %s because it exceeds the configured file size guidance.",
            path,
        )
        return None

    split_result = _split_frontmatter(text)
    if split_result is None:
        logger.warning(
            "Skipping skill file %s because YAML frontmatter could not be parsed.",
            path,
        )
        return None

    frontmatter_text, body = split_result

    try:
        raw_frontmatter = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as error:
        logger.warning(
            "Skipping skill file %s because frontmatter YAML is invalid: %s",
            path,
            error,
        )
        return None

    if not isinstance(raw_frontmatter, dict):
        logger.warning(
            "Skipping skill file %s because frontmatter is not a YAML mapping.",
            path,
        )
        return None

    frontmatter = cast(dict[str, object], raw_frontmatter)
    root = path.parent.resolve()
    skill_file = path.resolve()

    name = _coerce_required_string(frontmatter, "name")
    if not name:
        logger.warning(
            "Skipping skill file %s because `name` is missing or empty.", path
        )
        return None

    description = _coerce_required_string(frontmatter, "description")
    if not description:
        logger.warning(
            "Skipping skill file %s because `description` is missing or empty.", path
        )
        return None

    manifest = SkillManifest(
        name=_validate_name(name, root=root, skill_file=skill_file),
        description=_validate_description(description),
        skill_file=skill_file,
        root=root,
        license=_coerce_optional_string(frontmatter, "license"),
        compatibility=_validate_compatibility(
            _coerce_optional_string(frontmatter, "compatibility")
        ),
        metadata=_validate_metadata(frontmatter.get("metadata")),
        allowed_tools=_coerce_optional_string(frontmatter, "allowed-tools"),
    )

    instructions = body.strip()
    return ParsedSkillFile(
        manifest=manifest,
        instructions=instructions,
    )


def _split_frontmatter(text: str) -> tuple[str, str] | None:
    """Return YAML frontmatter and Markdown body from one skill file."""
    if not text.startswith("---"):
        return None

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    end_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break

    if end_index is None:
        return None

    frontmatter = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index + 1 :])
    return frontmatter, body


def _exceeds_file_size_limit(text: str) -> bool:
    """Return whether the skill file exceeds the configured line guidance."""
    return len(text.splitlines()) > SKILL_MAX_FILE_LINES


def _validate_name(name: str, *, root: Path, skill_file: Path) -> str:
    """Validate the required skill description field."""
    if len(name) > SKILL_MAX_NAME_LENGTH:
        logger.warning(
            "Skill `%s` in %s exceeds %d characters; continuing anyway.",
            name,
            skill_file,
            SKILL_MAX_NAME_LENGTH,
        )
    if name.startswith("-") or name.endswith("-"):
        logger.warning(
            "Skill `%s` in %s starts or ends with `-`; continuing anyway.",
            name,
            skill_file,
        )
    if "--" in name:
        logger.warning(
            "Skill `%s` in %s contains consecutive hyphens; continuing anyway.",
            name,
            skill_file,
        )
    for character in name:
        if character == "-":
            continue
        if (character.isalpha() and character.islower()) or character.isdigit():
            continue
        logger.warning(
            "Skill `%s` in %s contains non-compliant characters; continuing anyway.",
            name,
            skill_file,
        )
    if root.name != name:
        logger.warning(
            "Skill `%s` in %s does not match parent directory `%s`; continuing anyway.",
            name,
            skill_file,
            root.name,
        )

    return name


def _validate_description(description: str) -> str:
    """Validate the required skill description field."""
    if len(description) > SKILL_MAX_DESCRIPTION_LENGTH:
        logger.warning(
            "Skill description exceeds %d characters; truncating.",
            SKILL_MAX_DESCRIPTION_LENGTH,
        )

    return description[:SKILL_MAX_DESCRIPTION_LENGTH]


def _validate_compatibility(compatibility: str | None) -> str | None:
    """Validate the optional compatibility field."""
    if not compatibility:
        return None

    if len(compatibility) > SKILL_MAX_COMPATIBILITY_LENGTH:
        logger.warning(
            "Skill compatibility exceeds %d characters; truncating.",
            SKILL_MAX_COMPATIBILITY_LENGTH,
        )

    return compatibility[:SKILL_MAX_COMPATIBILITY_LENGTH]


def _validate_metadata(raw_metadata: object) -> dict[str, str] | None:
    """Validate and normalize the optional metadata mapping."""
    if raw_metadata is None:
        return None

    if not isinstance(raw_metadata, dict):
        logger.warning(
            "Ignoring non-mapping skill metadata of type %s.",
            type(raw_metadata).__name__,
        )
        return {}

    normalized: dict[str, str] = {}
    for key, value in cast(dict[object, object], raw_metadata).items():
        normalized[str(key)] = str(value)
    return normalized


def _coerce_required_string(frontmatter: dict[str, object], key: str) -> str | None:
    """Return one required frontmatter field coerced to string."""
    value = frontmatter.get(key)
    if value is None:
        return None
    return str(value).strip()


def _coerce_optional_string(frontmatter: dict[str, object], key: str) -> str | None:
    """Return one optional frontmatter field coerced to string."""
    value = frontmatter.get(key)
    if value is None:
        return None
    return str(value)
