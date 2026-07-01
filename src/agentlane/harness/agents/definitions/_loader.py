"""Load markdown agent files into `AgentDescriptor` values."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from agentlane.models import Model, ModelResponse, Tools

from ..._handoff import normalize_delegation_tool_name
from ..._lifecycle import AgentDescriptor
from ..._tooling import (
    InheritTools,
    OverrideTools,
    RestrictTools,
    ToolConfig,
    merge_tools,
)
from ._constraints import AGENT_MAX_SUBAGENT_DEPTH
from ._errors import AgentFileError
from ._model import ModelResolver
from ._parser import parse_agent_file
from ._tools import resolve_tool_config
from ._types import AgentManifest, SubagentLink


def descriptor_from_markdown(
    path: str | Path,
    *,
    model_resolver: ModelResolver | None = None,
    subagent_link: SubagentLink = SubagentLink.AS_TOOL,
    subagents: Sequence[AgentDescriptor | str | Path] = (),
) -> AgentDescriptor:
    """Build an `AgentDescriptor` from a markdown agent file.

    The frontmatter maps onto descriptor fields and the body becomes the
    instructions. `model` is resolved to a live client only when both a model
    spec is present and `model_resolver` is supplied; otherwise the descriptor's
    model stays `None` (a sub-agent then inherits its parent's model at runtime).
    `subagents` (descriptors or paths) are attached as agent-as-tools (default)
    or handoffs.

    Args:
        path: Path to the `AGENT.md` file.
        model_resolver: Optional resolver turning the frontmatter `model` spec
            into a live client. When omitted, the descriptor's model stays None.
        subagent_link: How resolved sub-agents attach to this agent.
        subagents: Child descriptors or paths to attach as sub-agents.

    Returns:
        AgentDescriptor: The fully built descriptor (children attached).

    Raises:
        FileNotFoundError: When `path` or a sub-agent path does not exist.
        AgentFileError: When an existing file cannot be parsed, sub-agent nesting
            exceeds the depth cap or forms a cycle, or two sub-agents map to the
            same delegation tool name.
    """
    return _descriptor_from_markdown(
        path,
        model_resolver=model_resolver,
        subagent_link=subagent_link,
        subagents=subagents,
        seen=frozenset(),
        depth=0,
    )


def with_subagents(
    descriptor: AgentDescriptor,
    children: Sequence[AgentDescriptor],
    *,
    link: SubagentLink = SubagentLink.AS_TOOL,
) -> AgentDescriptor:
    """Return `descriptor` with resolved child descriptors attached.

    This is the single place sub-agents become part of a descriptor, shared by
    the markdown loader and `DefaultAgent(subagents=...)`. `AS_TOOL` wires each
    child as an agent-as-tool (the parent calls it and continues); `HANDOFF`
    appends it as a first-class handoff target. Two children whose names
    normalize to the same delegation tool name are rejected, since they would
    silently shadow each other and make runtime dispatch ambiguous.
    """
    children = tuple(children)
    if not children:
        return descriptor

    _reject_duplicate_delegation_names(children)

    if link is SubagentLink.HANDOFF:
        existing = descriptor.handoffs or ()
        return replace(descriptor, handoffs=existing + children)

    child_tools = Tools(tools=tuple(child.as_tool() for child in children))
    return replace(descriptor, tools=_with_child_tools(descriptor.tools, child_tools))


def _descriptor_from_markdown(
    path: str | Path,
    *,
    model_resolver: ModelResolver | None,
    subagent_link: SubagentLink,
    subagents: Sequence[AgentDescriptor | str | Path],
    seen: frozenset[Path],
    depth: int,
) -> AgentDescriptor:
    """Parse one file and recursively attach its sub-agents as values."""
    if depth > AGENT_MAX_SUBAGENT_DEPTH:
        raise AgentFileError(
            f"sub-agent nesting exceeds maximum depth {AGENT_MAX_SUBAGENT_DEPTH}: {path}"
        )

    resolved = Path(path).resolve()
    if resolved in seen:
        raise AgentFileError(f"sub-agent cycle detected at: {resolved}")

    # A missing/unreadable file raises from parse_agent_file; malformed
    # frontmatter returns None and becomes a clear error at this boundary.
    parsed = parse_agent_file(resolved)
    if parsed is None:
        raise AgentFileError(f"could not load agent file: {resolved}")

    manifest = parsed.manifest
    model = _resolve_model(manifest, model_resolver=model_resolver)
    tool_config, deny_shims = resolve_tool_config(
        manifest.allowed_tools,
        manifest.disallowed_tools,
    )

    child_seen = seen | {resolved}
    children = tuple(
        _coerce_subagent(
            item,
            model_resolver=model_resolver,
            subagent_link=subagent_link,
            seen=child_seen,
            depth=depth + 1,
        )
        for item in subagents
    )

    descriptor = AgentDescriptor(
        name=manifest.name or "",
        description=manifest.description,
        model=model,
        instructions=parsed.instructions or None,
        model_args=manifest.model_args,
        tools=tool_config,
        shims=deny_shims or None,
    )

    return with_subagents(descriptor, children, link=subagent_link)


def _coerce_subagent(
    item: AgentDescriptor | str | Path,
    *,
    model_resolver: ModelResolver | None,
    subagent_link: SubagentLink,
    seen: frozenset[Path],
    depth: int,
) -> AgentDescriptor:
    """Return `item` as a child descriptor, loading a path if needed."""
    if isinstance(item, AgentDescriptor):
        return item

    return _descriptor_from_markdown(
        item,
        model_resolver=model_resolver,
        subagent_link=subagent_link,
        subagents=(),
        seen=seen,
        depth=depth,
    )


def _reject_duplicate_delegation_names(children: tuple[AgentDescriptor, ...]) -> None:
    """Raise when two sub-agents map to the same delegation tool name.

    Sub-agents are exposed as delegation tools named by
    `normalize_delegation_tool_name(child.name)` (whether attached as
    agent-as-tools or handoffs). Two children that normalize to the same name
    would silently shadow each other and make runtime dispatch ambiguous.
    """
    seen: set[str] = set()
    for child in children:
        tool_name = normalize_delegation_tool_name(child.name)
        if tool_name in seen:
            raise AgentFileError(
                f"duplicate sub-agent delegation tool name '{tool_name}'"
            )

        seen.add(tool_name)


def _resolve_model(
    manifest: AgentManifest,
    *,
    model_resolver: ModelResolver | None,
) -> Model[ModelResponse] | None:
    """Resolve the manifest model spec into a live client, or `None`.

    `None` means "no model here" — for a sub-agent that is inheritance from the
    parent; for a root agent the runnable boundary (`DefaultAgent.from_markdown`)
    rejects it. A declared spec with no resolver also yields `None`, surfacing at
    that same boundary rather than being resolved here.
    """
    spec = manifest.model_spec
    if spec is None:
        return None

    if model_resolver is None:
        return None

    return model_resolver.resolve(spec, model_args=manifest.model_args or {})


def _with_child_tools(tool_config: ToolConfig, child_tools: Tools) -> ToolConfig:
    """Add sub-agent tools to a tool policy after the base policy applies."""
    if isinstance(tool_config, (InheritTools, OverrideTools, RestrictTools)):
        return tool_config.with_tools(child_tools)

    # Bare `Tools` or `None`: merge the child tools in, preserving any existing.
    return merge_tools(tool_config, child_tools.normalized_tools)
