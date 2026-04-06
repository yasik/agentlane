"""Tooling helpers for the harness."""

from dataclasses import dataclass
from typing import cast

from agentlane.models import Tools


@dataclass(frozen=True, slots=True)
class _InheritToolsSentinel:
    """Sentinel meaning "inherit tools from the parent agent if available"."""


INHERIT_TOOLS = _InheritToolsSentinel()
"""Default tool policy for descriptors that do not set tools explicitly."""

type ToolConfig = Tools | None | _InheritToolsSentinel
"""Static tool configuration, including the inheritance sentinel."""


def resolve_tools(
    tools: ToolConfig,
    *,
    parent_tools: Tools | None = None,
) -> Tools | None:
    """Resolve one descriptor tool setting against optional parent tools.

    Rules:
    1. ``INHERIT_TOOLS`` -> inherit ``parent_tools`` when present.
    2. ``None`` -> expose no tools explicitly.
    3. ``Tools`` -> expose exactly that tool set.
    """
    if tools is INHERIT_TOOLS:
        return parent_tools
    return cast(Tools | None, tools)
