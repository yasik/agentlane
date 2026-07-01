"""Shared YAML-frontmatter parsing for markdown definition files.

Both the skills loader (`SKILL.md`) and the agent-definition loader (`AGENT.md`)
read a leading `---`-fenced YAML block followed by a Markdown body. The
format-level parsing is identical across both, so it lives here once and is
imported by each loader rather than duplicated.

These helpers are pure: they return a value (or `None` for "no parseable
frontmatter") and do not log or raise for malformed content. Each caller
decides what `None` means — the skills loader skips the file, the agent loader
raises at its explicit-load boundary.
"""

from collections.abc import Sequence
from typing import cast

import yaml


def split_frontmatter(text: str) -> tuple[str, str] | None:
    """Return the YAML frontmatter and Markdown body from one document.

    The document must start with a line that is exactly `---` and have a
    matching closing `---` line. Returns `(frontmatter_text, body_text)` or
    `None` when no complete fence is present.
    """
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


def load_frontmatter(text: str) -> tuple[dict[str, object], str] | None:
    """Split and parse the frontmatter mapping of one document.

    Returns `(frontmatter_mapping, body)`, or `None` when the document has no
    parseable frontmatter, invalid YAML, or a non-mapping frontmatter value.
    """
    split_result = split_frontmatter(text)
    if split_result is None:
        return None

    frontmatter_text, body = split_result

    try:
        raw_frontmatter = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError:
        return None

    if not isinstance(raw_frontmatter, dict):
        return None

    return cast(dict[str, object], raw_frontmatter), body


def coerce_required_string(frontmatter: dict[str, object], key: str) -> str | None:
    """Return one required frontmatter field coerced to a stripped string."""
    value = frontmatter.get(key)
    if value is None:
        return None

    return str(value).strip()


def coerce_optional_string(frontmatter: dict[str, object], key: str) -> str | None:
    """Return one optional frontmatter field coerced to a stripped string."""
    value = frontmatter.get(key)
    if value is None:
        return None

    return str(value).strip()


def parse_optional_tool_names(
    frontmatter: dict[str, object],
    *,
    key: str,
) -> tuple[str, ...] | None:
    """Return parsed tool names when a field is present, preserving omission.

    Returns `None` when `key` is absent (the caller's signal for "inherit all"),
    and a deduped tuple when the key is present, including an empty tuple for an
    explicit empty list.
    """
    if key not in frontmatter:
        return None

    return parse_tool_names(frontmatter.get(key))


def parse_tool_names(raw_value: object) -> tuple[str, ...]:
    """Normalize a comma-separated string or YAML list of tool names.

    Entries are stripped, empty entries are dropped, and duplicates are removed
    while preserving first-occurrence order. A value that is neither a string
    nor a list yields an empty tuple.
    """
    if raw_value is None:
        return ()

    if isinstance(raw_value, str):
        raw_items: Sequence[object] = raw_value.split(",")
    elif isinstance(raw_value, (list, tuple)):
        raw_items = cast(Sequence[object], raw_value)
    else:
        return ()

    names: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        name = str(raw_item).strip()
        if not name:
            continue
        if name in seen:
            continue

        seen.add(name)
        names.append(name)

    return tuple(names)
