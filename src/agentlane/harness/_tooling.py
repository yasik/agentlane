"""Tooling helpers for the harness."""

from dataclasses import dataclass
from typing import Any

from agentlane.models import Tools, ToolSpec


@dataclass(frozen=True, slots=True)
class InheritTools:
    """Tool policy that inherits parent tools and adds optional child tools."""

    tools: Tools | None = None

    def with_tools(self, tools: Tools) -> "InheritTools":
        """Return an inherit policy with child-local tool additions."""
        return InheritTools(tools=tools)


@dataclass(frozen=True, slots=True)
class OverrideTools:
    """Tool policy that ignores parent tools and uses only explicit tools."""

    tools: Tools | None = None

    def with_tools(self, tools: Tools) -> "OverrideTools":
        """Return an override policy with explicit tool replacements."""
        return OverrideTools(tools=tools)


@dataclass(frozen=True, slots=True)
class RestrictTools:
    """Tool policy that filters parent tools and adds optional child tools."""

    names: frozenset[str]
    tools: Tools | None = None

    def with_tools(self, tools: Tools) -> "RestrictTools":
        """Return a restrict policy with child-local tool additions."""
        return RestrictTools(names=self.names, tools=tools)


@dataclass(frozen=True, slots=True)
class RestrictToolsBuilder:
    """Factory for restrict policies that require explicit allowed names."""

    def only(
        self,
        *names: str,
        tools: Tools | None = None,
    ) -> RestrictTools:
        """Return a policy that inherits only named parent tools plus additions."""
        return RestrictTools(names=frozenset(names), tools=tools)


INHERIT_TOOLS = InheritTools()
"""Default tool policy for descriptors that do not set tools explicitly."""

OVERRIDE_TOOLS = OverrideTools()
"""Tool policy that replaces parent tools with no tools by default."""

RESTRICT_TOOLS = RestrictToolsBuilder()
"""Factory for tool policies that filter inherited parent tools by name."""

type ToolConfig = Tools | None | InheritTools | OverrideTools | RestrictTools
"""Static tool configuration for resolving child-visible direct tools.

Bare ``Tools`` and ``None`` are compatibility shorthands for override behavior.
Use ``INHERIT_TOOLS.with_tools(...)`` to inherit and add child-local tools,
``OVERRIDE_TOOLS.with_tools(...)`` to replace parent tools, and
``RESTRICT_TOOLS.only(...)`` to filter parent tools before adding child-local
tools.
"""


def resolve_tools(
    tools: ToolConfig,
    *,
    parent_tools: Tools | None = None,
) -> Tools | None:
    """Resolve one descriptor tool setting against optional parent tools.

    Rules:
    1. ``InheritTools`` -> inherit parent tools and merge optional child tools.
    2. ``OverrideTools`` -> expose explicit tools only, or no tools.
    3. ``RestrictTools`` -> filter parent tools by name, then merge additions.
    4. Bare ``None`` and ``Tools`` preserve legacy override behavior.
    """
    if isinstance(tools, InheritTools):
        return merge_tool_configs(parent_tools, tools.tools)
    if isinstance(tools, OverrideTools):
        return tools.tools
    if isinstance(tools, RestrictTools):
        restricted_parent = filter_tools(parent_tools, names=tools.names)
        return merge_tool_configs(restricted_parent, tools.tools)
    return tools


def merge_tool_configs(
    primary: Tools | None,
    extra: Tools | None,
) -> Tools | None:
    """Merge two `Tools` configs while preserving the primary policy knobs."""
    if extra is None:
        return primary
    if primary is None:
        return extra
    return _with_tools(
        primary,
        _merge_tool_specs(
            primary.normalized_tools,
            extra.normalized_tools,
        ),
    )


def filter_tools(
    tools: Tools | None,
    *,
    names: frozenset[str],
) -> Tools | None:
    """Return a copy containing only tools whose names are allowed."""
    if tools is None:
        return None
    if not names:
        return None

    filtered = tuple(tool for tool in tools.normalized_tools if tool.name in names)
    if not filtered:
        return None

    return _with_tools(tools, filtered)


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
        return Tools(tools=_merge_tool_specs((), extra))

    merged_tools = _merge_tool_specs(primary.normalized_tools, extra)
    return _with_tools(primary, merged_tools)


def _merge_tool_specs(
    primary: tuple[ToolSpec[Any], ...],
    extra: tuple[ToolSpec[Any], ...],
) -> tuple[ToolSpec[Any], ...]:
    """Return named tool specs once, preserving first occurrence."""
    merged_tools: list[ToolSpec[Any]] = []
    seen_names: set[str] = set()

    for tool in primary + extra:
        if tool.name in seen_names:
            continue

        seen_names.add(tool.name)
        merged_tools.append(tool)

    return tuple(merged_tools)


def _with_tools(
    base: Tools,
    tools: tuple[ToolSpec[Any], ...],
) -> Tools:
    """Return `base` settings with a replacement tool sequence."""
    return Tools(
        tools=tools,
        tool_choice=base.tool_choice,
        parallel_tool_calls=base.parallel_tool_calls,
        tool_call_timeout=base.tool_call_timeout,
        tool_call_max_retries=base.tool_call_max_retries,
        tool_call_limits=base.tool_call_limits,
        max_tool_round_trips=base.max_tool_round_trips,
    )
