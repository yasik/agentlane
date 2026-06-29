"""Claude-Code-style markdown agent definitions for the harness.

Parse `AGENT.md` files (YAML frontmatter + a system-prompt body) into
`AgentDescriptor` values, with an injectable model resolver and native
AgentLane tool policy. See `AgentDescriptor.from_markdown` and
`DefaultAgent.from_markdown` for the ergonomic entry points.
"""

from ._constraints import (
    AGENT_MAX_DESCRIPTION_LENGTH,
    AGENT_MAX_INSTRUCTIONS_LINES,
    AGENT_MAX_SUBAGENT_DEPTH,
    AgentFileError,
)
from ._loader import descriptor_from_markdown
from ._model import FactoryModelResolver, ModelResolver
from ._parser import parse_agent_file
from ._tools import resolve_tool_config
from ._types import AgentManifest, ParsedAgentFile, SubagentLink

__all__ = [
    "AGENT_MAX_DESCRIPTION_LENGTH",
    "AGENT_MAX_INSTRUCTIONS_LINES",
    "AGENT_MAX_SUBAGENT_DEPTH",
    "AgentFileError",
    "AgentManifest",
    "FactoryModelResolver",
    "ModelResolver",
    "ParsedAgentFile",
    "SubagentLink",
    "descriptor_from_markdown",
    "parse_agent_file",
    "resolve_tool_config",
]
