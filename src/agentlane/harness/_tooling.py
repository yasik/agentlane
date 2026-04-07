"""Tooling helpers for the harness."""

from dataclasses import dataclass
from typing import Any, cast

from agentlane.models import Tools, ToolSpec


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


def merge_tools(
    primary: Tools | None,
    extra: tuple[ToolSpec[Any], ...],
) -> Tools | None:
    """Merge instance-bound extra tools into one base `Tools` config.

    The returned `Tools` preserves all scheduling and loop-control settings from
    the base configuration. If the base configuration is absent, the extra tools
    are exposed with the default `Tools(...)` settings.
    """
    if not extra:
        return primary
    if primary is None:
        return Tools(tools=extra)

    merged_tools = tuple(primary.normalized_tools) + extra
    return Tools(
        tools=merged_tools,
        tool_choice=primary.tool_choice,
        parallel_tool_calls=primary.parallel_tool_calls,
        tool_call_timeout=primary.tool_call_timeout,
        tool_call_max_retries=primary.tool_call_max_retries,
        tool_call_limits=primary.tool_call_limits,
        max_tool_round_trips=primary.max_tool_round_trips,
    )
