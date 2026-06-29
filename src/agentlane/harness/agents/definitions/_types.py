"""Typed primitives for the agent-definition loader."""

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SubagentLink(enum.Enum):
    """How a markdown sub-agent attaches to its parent agent."""

    AS_TOOL = "as_tool"
    """Expose the sub-agent as an agent-as-tool (subroutine; parent continues)."""

    HANDOFF = "handoff"
    """Expose the sub-agent as a first-class handoff target (control transfer)."""


@dataclass(frozen=True, slots=True)
class AgentManifest:
    """Parsed frontmatter for one `AGENT.md` file.

    Pure data: it holds the raw model spec string (resolved to a live client
    later, outside the parser) and tool names as declared, never live objects.
    """

    name: str | None
    """Agent name from frontmatter, or `None` to use the generated fallback."""

    description: str | None
    """Optional agent description; drives delegation when used as a sub-agent."""

    model_spec: str | None
    """Raw provider/model spec string, or `None` for inherit/omitted."""

    model_args: dict[str, Any] | None
    """Free-form model request arguments forwarded verbatim downstream."""

    allowed_tools: tuple[str, ...] | None
    """Tool allowlist, or `None` when the `tools` key is omitted (inherit-all)."""

    disallowed_tools: tuple[str, ...]
    """Tool denylist removed from the visible set before model exposure."""

    source_path: Path
    """Absolute path to the parsed `AGENT.md` file."""


@dataclass(frozen=True, slots=True)
class ParsedAgentFile:
    """Parsed `AGENT.md` contents: manifest plus the system-prompt body."""

    manifest: AgentManifest
    """Discovered agent manifest."""

    instructions: str
    """Markdown body after frontmatter stripping (the system prompt)."""
