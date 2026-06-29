"""`AGENT.md` frontmatter parsing for markdown agent definitions."""

from pathlib import Path
from typing import Any, cast

from ..._frontmatter import (
    coerce_optional_string,
    load_frontmatter,
    parse_optional_tool_names,
    parse_tool_names,
)
from ._constraints import AGENT_MAX_DESCRIPTION_LENGTH, AGENT_MAX_INSTRUCTIONS_LINES
from ._types import AgentManifest, ParsedAgentFile

_INHERIT_MODEL_SENTINEL = "inherit"


def parse_agent_file(path: Path) -> ParsedAgentFile | None:
    """Parse one `AGENT.md` file into a manifest and system-prompt body.

    Returns `None` only when the file has no parseable frontmatter (no/unclosed
    fence, invalid YAML, or non-mapping frontmatter); the loader turns that into
    a raised error at the explicit-load boundary. Read failures are not caught
    here — a missing or unreadable file raises `OSError` so the caller can fix
    the underlying problem rather than have the agent vanish silently.
    """
    agent_file = path.resolve()

    # Read errors (missing file, permissions) propagate to the caller.
    text = agent_file.read_text(encoding="utf-8")

    loaded = load_frontmatter(text)
    if loaded is None:
        return None

    frontmatter, body = loaded

    manifest = AgentManifest(
        name=coerce_optional_string(frontmatter, "name"),
        description=_truncate_description(
            coerce_optional_string(frontmatter, "description")
        ),
        model_spec=_normalize_model_spec(coerce_optional_string(frontmatter, "model")),
        model_args=_parse_model_args(frontmatter.get("model_args")),
        allowed_tools=parse_optional_tool_names(frontmatter, key="tools"),
        disallowed_tools=parse_tool_names(frontmatter.get("disallowedTools")),
        source_path=agent_file,
    )

    instructions = _truncate_instructions(body, source=agent_file)
    return ParsedAgentFile(manifest=manifest, instructions=instructions)


def _normalize_model_spec(model_spec: str | None) -> str | None:
    """Return the raw provider/model spec, or `None` for inherit/omitted.

    The parser performs no provider routing or validation: the spec is an opaque
    pass-through resolved later by an injected `ModelResolver`. The only
    recognized sentinel is `inherit` (case-insensitive), which maps to `None` so
    the descriptor inherits its parent's model.
    """
    if not model_spec:
        return None

    if model_spec.strip().lower() == _INHERIT_MODEL_SENTINEL:
        return None

    return model_spec


def _parse_model_args(raw_value: object) -> dict[str, Any] | None:
    """Coerce the optional `model_args` mapping forwarded as request kwargs.

    Returns `None` when absent or not a mapping. Conflicting values (such as
    `temperature` with `reasoning_effort`) pass through untouched and surface at
    the model call, which is the layer that owns those rules.
    """
    if raw_value is None:
        return None

    if not isinstance(raw_value, dict):
        return None

    return {
        str(key): value for key, value in cast(dict[object, object], raw_value).items()
    }


def _truncate_description(description: str | None) -> str | None:
    """Cap the description at the configured length.

    The description is a short delegation hint surfaced as the sub-agent tool
    description, so an over-length value is truncated rather than carried whole.
    """
    if not description:
        return None

    return description[:AGENT_MAX_DESCRIPTION_LENGTH]


def _truncate_instructions(body: str, *, source: Path) -> str:
    """Return the body, truncated to the line cap with a pointer when oversized.

    Only the instruction body is bounded — never the frontmatter. An oversized
    body is cut to the line cap and annotated with the source path so the model
    can read the remainder with its own tools instead of losing the agent.
    """
    stripped = body.strip()
    lines = stripped.splitlines()
    if len(lines) <= AGENT_MAX_INSTRUCTIONS_LINES:
        return stripped

    head = "\n".join(lines[:AGENT_MAX_INSTRUCTIONS_LINES]).strip()
    return (
        f"{head}\n\n"
        f"[Instructions truncated after {AGENT_MAX_INSTRUCTIONS_LINES} lines. "
        f"Read the full agent definition at {source} for the rest.]"
    )
