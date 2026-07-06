"""Map markdown tool frontmatter onto AgentLane tool policy."""

from ..._tooling import (
    INHERIT_TOOLS,
    OVERRIDE_TOOLS,
    RESTRICT_TOOLS,
    ToolConfig,
)
from ...shims import ExcludeToolsShim, Shim


def resolve_tool_config(
    allowed: tuple[str, ...] | None,
    disallowed: tuple[str, ...],
) -> tuple[ToolConfig, tuple[Shim, ...]]:
    """Resolve frontmatter tool names into a tool policy and deny shims.

    Tool names are AgentLane's native names used directly (no aliasing). Names
    not exposed by the parent are tolerated: the allow-list filter and the deny
    shim both ignore absent names, which is correct for custom/MCP tool names.

    Returns:
        A `(ToolConfig, shims)` pair:

        - `allowed is None` (key omitted) → `INHERIT_TOOLS` (inherit-all).
        - `allowed` empty (explicit `tools: []`) → `OVERRIDE_TOOLS` (none).
        - `allowed` non-empty → `RESTRICT_TOOLS.only(*allowed)`.
        - `disallowed` non-empty → an `ExcludeToolsShim` applied each turn.
    """
    shims: tuple[Shim, ...] = ()
    if disallowed:
        shims = (ExcludeToolsShim(names=disallowed),)

    if allowed is None:
        return INHERIT_TOOLS, shims

    if len(allowed) == 0:
        return OVERRIDE_TOOLS, shims

    return RESTRICT_TOOLS.only(*allowed), shims
