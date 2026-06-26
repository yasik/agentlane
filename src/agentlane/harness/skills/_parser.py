"""`SKILL.md` frontmatter parsing for harness skills."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import structlog
import yaml

from ._constraints import (
    SKILL_MAX_COMPATIBILITY_LENGTH,
    SKILL_MAX_DESCRIPTION_LENGTH,
    SKILL_MAX_FILE_LINES,
    SKILL_MAX_NAME_LENGTH,
)
from ._types import SkillManifest

LOGGER = structlog.get_logger(log_tag="agentlane.harness.skills.parser")


@dataclass(frozen=True, slots=True)
class ParsedSkillFile:
    """Parsed `SKILL.md` contents used by the filesystem skill loader."""

    manifest: SkillManifest
    """Discovered skill manifest."""

    instructions: str
    """Markdown body after frontmatter stripping."""


def parse_skill_file(path: Path) -> ParsedSkillFile | None:
    """Read and parse one `SKILL.md` file into a manifest and instructions body."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        LOGGER.warning(
            "failed to read skill file",
            skill_file=str(path),
            error=str(error),
        )
        return None

    return parse_skill_text(text, skill_file=path.resolve(), root=path.parent.resolve())


def parse_skill_text(
    text: str, *, skill_file: Path, root: Path
) -> ParsedSkillFile | None:
    """Parse `SKILL.md` text into a manifest and instructions body.

    Holds the frontmatter parsing, validation, and manifest construction shared by every skill
    source. The caller supplies the already-read `text` plus the `skill_file`/`root` paths recorded
    on the manifest (neither is read from here), so a non-filesystem source can reuse the same
    skill contract by reading its own bytes first. `parse_skill_file` is the filesystem caller.
    """
    if _exceeds_file_size_limit(text):
        LOGGER.warning(
            "skipping oversized skill file",
            skill_file=str(skill_file),
            max_lines=SKILL_MAX_FILE_LINES,
        )
        return None

    split_result = _split_frontmatter(text)
    if split_result is None:
        LOGGER.warning(
            "skipping skill file without parseable frontmatter",
            skill_file=str(skill_file),
        )
        return None

    frontmatter_text, body = split_result

    try:
        raw_frontmatter = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as error:
        LOGGER.warning(
            "skipping skill file with invalid frontmatter YAML",
            skill_file=str(skill_file),
            error=str(error),
        )
        return None

    if not isinstance(raw_frontmatter, dict):
        LOGGER.warning(
            "skipping skill file with non-mapping frontmatter",
            skill_file=str(skill_file),
            frontmatter_type=type(raw_frontmatter).__name__,
        )
        return None

    frontmatter = cast(dict[str, object], raw_frontmatter)

    name = _coerce_required_string(frontmatter, "name")
    if not name:
        LOGGER.warning(
            "skipping skill file with missing name",
            skill_file=str(skill_file),
        )
        return None

    description = _coerce_required_string(frontmatter, "description")
    if not description:
        LOGGER.warning(
            "skipping skill file with missing description",
            skill_file=str(skill_file),
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
        tools=_validate_optional_tool_names(
            frontmatter,
            key="tools",
            skill_file=skill_file,
        ),
        disallowed_tools=_validate_tool_names(
            frontmatter.get("disallowedTools"),
            field_name="disallowedTools",
            skill_file=skill_file,
        ),
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
    """Validate the required skill name field."""
    if len(name) > SKILL_MAX_NAME_LENGTH:
        LOGGER.warning(
            "skill name exceeds configured length",
            skill=name,
            skill_file=str(skill_file),
            max_length=SKILL_MAX_NAME_LENGTH,
        )
    if name.startswith("-") or name.endswith("-"):
        LOGGER.warning(
            "skill name starts or ends with hyphen",
            skill=name,
            skill_file=str(skill_file),
        )
    if "--" in name:
        LOGGER.warning(
            "skill name contains consecutive hyphens",
            skill=name,
            skill_file=str(skill_file),
        )
    for character in name:
        if character == "-":
            continue
        if (character.isalpha() and character.islower()) or character.isdigit():
            continue
        LOGGER.warning(
            "skill name contains non-compliant characters",
            skill=name,
            skill_file=str(skill_file),
        )
        break
    if root.name != name:
        LOGGER.warning(
            "skill name does not match parent directory",
            skill=name,
            skill_file=str(skill_file),
            directory=root.name,
        )

    return name


def _validate_description(description: str) -> str:
    """Validate the required skill description field."""
    if len(description) > SKILL_MAX_DESCRIPTION_LENGTH:
        LOGGER.warning(
            "skill description exceeds configured length",
            max_length=SKILL_MAX_DESCRIPTION_LENGTH,
        )

    return description[:SKILL_MAX_DESCRIPTION_LENGTH]


def _validate_compatibility(compatibility: str | None) -> str | None:
    """Validate the optional compatibility field."""
    if not compatibility:
        return None

    if len(compatibility) > SKILL_MAX_COMPATIBILITY_LENGTH:
        LOGGER.warning(
            "skill compatibility exceeds configured length",
            max_length=SKILL_MAX_COMPATIBILITY_LENGTH,
        )

    return compatibility[:SKILL_MAX_COMPATIBILITY_LENGTH]


def _validate_metadata(raw_metadata: object) -> dict[str, str] | None:
    """Validate and normalize the optional metadata mapping."""
    if raw_metadata is None:
        return None

    if not isinstance(raw_metadata, dict):
        LOGGER.warning(
            "ignoring non-mapping skill metadata",
            metadata_type=type(raw_metadata).__name__,
        )
        return None

    normalized: dict[str, str] = {}
    for key, value in cast(dict[object, object], raw_metadata).items():
        normalized[str(key)] = str(value)
    return normalized


def _validate_optional_tool_names(
    frontmatter: dict[str, object],
    *,
    key: str,
    skill_file: Path,
) -> tuple[str, ...] | None:
    """Return parsed tool names when a field is present, preserving omission."""
    if key not in frontmatter:
        return None

    return _validate_tool_names(
        frontmatter.get(key),
        field_name=key,
        skill_file=skill_file,
    )


def _validate_tool_names(
    raw_value: object,
    *,
    field_name: str,
    skill_file: Path,
) -> tuple[str, ...]:
    """Normalize comma-separated or YAML-list tool names."""
    if raw_value is None:
        return ()
    if isinstance(raw_value, str):
        raw_items: Sequence[object] = raw_value.split(",")
    elif isinstance(raw_value, (list, tuple)):
        raw_items = cast(Sequence[object], raw_value)
    else:
        LOGGER.warning(
            "ignoring invalid skill tool field",
            field=field_name,
            skill_file=str(skill_file),
            value_type=type(raw_value).__name__,
        )
        return ()

    names: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        name = str(raw_item).strip()
        if name == "":
            LOGGER.warning(
                "ignoring empty skill tool entry",
                field=field_name,
                skill_file=str(skill_file),
            )
            continue
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return tuple(names)


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
    return str(value).strip()
