"""Load markdown agent files into `AgentDescriptor` values."""

from collections.abc import Sequence
from pathlib import Path

from agentlane.models import Model, ModelResponse, Tools

from ..._handoff import normalize_delegation_tool_name
from ..._lifecycle import AgentDescriptor
from ..._tooling import (
    InheritTools,
    OverrideTools,
    RestrictTools,
    ToolConfig,
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
    `subagents` paths are parsed into child descriptors and attached as
    agent-as-tools (default) or handoffs.

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
        AgentFileError: When an existing file cannot be parsed, or sub-agent
            nesting exceeds the depth cap or forms a cycle.
    """
    return _descriptor_from_markdown(
        path,
        model_resolver=model_resolver,
        subagent_link=subagent_link,
        subagents=subagents,
        seen=frozenset(),
        depth=0,
    )


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

    # Handoffs transfer control to the child; agent-as-tools run as subroutines.
    handoffs: tuple[AgentDescriptor, ...] | None = None
    if children:
        _reject_duplicate_delegation_names(children, source=resolved)
        if subagent_link is SubagentLink.HANDOFF:
            handoffs = children
        else:
            child_tools = Tools(tools=tuple(child.as_tool() for child in children))
            tool_config = _with_child_tools(tool_config, child_tools)

    return AgentDescriptor(
        name=manifest.name or "",
        description=manifest.description,
        model=model,
        instructions=parsed.instructions or None,
        model_args=manifest.model_args,
        tools=tool_config,
        shims=deny_shims or None,
        handoffs=handoffs,
    )


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


def _reject_duplicate_delegation_names(
    children: tuple[AgentDescriptor, ...],
    *,
    source: Path,
) -> None:
    """Raise when two sub-agents map to the same delegation tool name.

    Sub-agents are exposed as delegation tools named by
    `normalize_delegation_tool_name(child.name)` (whether attached as
    agent-as-tools or handoffs). Two children that normalize to the same name
    would silently shadow each other and make runtime dispatch ambiguous, so
    fail loudly at load time, matching the cycle/depth error contract.
    """
    seen: set[str] = set()
    for child in children:
        tool_name = normalize_delegation_tool_name(child.name)
        if tool_name in seen:
            raise AgentFileError(
                f"duplicate sub-agent delegation tool name '{tool_name}' "
                f"while loading: {source}"
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
    """Add sub-agent tools to a tool policy after frontmatter rules apply."""
    if isinstance(tool_config, (InheritTools, OverrideTools, RestrictTools)):
        return tool_config.with_tools(child_tools)

    return OverrideTools(tools=child_tools)
