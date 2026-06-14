"""Shim integration for first-party harness tool definitions."""

from collections.abc import Callable, Iterable, Sequence
from itertools import chain
from pathlib import Path
from typing import Any, Literal

from agentlane.models.run import RunContext

from .._run import RunState, ShimState
from .._tooling import merge_tools
from ..shims import BoundShim, PreparedTurn, Shim, ShimBindingContext
from ._agent import agent_tool
from ._bash import bash_tool
from ._find import find_tool
from ._grep import grep_tool
from ._patch import patch_tool
from ._permissions import ToolApprovalCallback, ToolPermissionPolicy
from ._plan import plan_state_key, plan_tool
from ._read import read_tool
from ._types import HarnessToolDefinition
from ._write import write_tool

_PROMPT_MARKER_KEY_SUFFIX = "prompt-appended"
_PLAN_TOOL_NAME = "write_plan"
BASE_TOOL_NAMES: tuple[str, ...] = (
    "read",
    "find",
    "grep",
    "patch",
    "write",
    _PLAN_TOOL_NAME,
    "bash",
    "agent",
)
"""Names of the first-party base harness tools, in standard selection order.

Applications that compose `base_harness_tools` with their own tools can read
this constant instead of re-deriving the set from `base_harness_tools()`. It is
the same set the `include`/`exclude` selectors validate against; pass
application tool names through `extra_names` to let those selectors reference
tools this function does not own.
"""

_BASE_TOOL_NAME_SET = frozenset(BASE_TOOL_NAMES)
type _BaseToolFactory = Callable[[], HarnessToolDefinition]


class _BoundHarnessToolsShim(BoundShim):
    """Bound shim session that contributes tools and prompt metadata."""

    def __init__(
        self,
        *,
        shim_name: str,
        definitions: tuple[HarnessToolDefinition, ...],
        prompt_block: str | None,
    ) -> None:
        self._shim_name = shim_name
        self._current_run_state: RunState | None = None
        self._definitions = tuple(
            self._bind_definition(definition) for definition in definitions
        )
        self._prompt_block = prompt_block

    async def on_run_start(
        self,
        state: RunState,
        transient_state: RunContext[Any],
    ) -> None:
        del transient_state
        self._current_run_state = state

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        self._current_run_state = turn.run_state
        tool_specs = tuple(definition.tool for definition in self._definitions)
        turn.tools = merge_tools(turn.tools, tool_specs)

        if self._prompt_block is None:
            return
        if turn.run_state.turn_count != 1:
            return

        marker_key = _prompt_marker_key(self._shim_name)
        if turn.run_state.shim_state.get(marker_key) is True:
            return

        turn.append_system_instruction(self._prompt_block)
        turn.run_state.shim_state[marker_key] = True

    def _require_shim_state(self) -> ShimState:
        """Return the persisted shim state for the current run."""
        if self._current_run_state is None:
            raise RuntimeError("HarnessToolsShim tool execution requires run state.")
        return self._current_run_state.shim_state

    def _bind_definition(
        self,
        definition: HarnessToolDefinition,
    ) -> HarnessToolDefinition:
        """Attach shim-owned state to stateful first-party tools."""
        if definition.tool.name != _PLAN_TOOL_NAME:
            return definition

        return plan_tool(
            persist_to=self._persist_plan,
            prompt_snippet=definition.prompt_snippet,
            prompt_guidelines=tuple(definition.prompt_guidelines),
        )

    def _persist_plan(self, snapshot: dict[str, object]) -> None:
        """Persist the latest plan snapshot in shim-owned state."""
        self._require_shim_state()[plan_state_key(self._shim_name)] = snapshot


class HarnessToolsShim(Shim):
    """First-party shim that exposes harness tools and prompt guidance."""

    def __init__(
        self,
        definitions: Sequence[HarnessToolDefinition],
        *,
        name: str = "harness-tools",
        prompt_guidelines: Sequence[str] = (),
    ) -> None:
        self._definitions = tuple(definitions)
        self._name = name
        self._prompt_guidelines = tuple(prompt_guidelines)
        _validate_unique_tool_names(self._definitions)

    @property
    def name(self) -> str:
        return self._name

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        del context
        return _BoundHarnessToolsShim(
            shim_name=self._name,
            definitions=self._definitions,
            prompt_block=render_harness_tools_prompt(
                definitions=self._definitions,
                prompt_guidelines=self._prompt_guidelines,
            ),
        )


def base_harness_tools(
    *,
    cwd: str | Path | None = None,
    permissions: ToolPermissionPolicy | None = None,
    approval_callback: ToolApprovalCallback | None = None,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    extra_names: Iterable[str] | None = None,
) -> tuple[HarnessToolDefinition, ...]:
    """Return currently implemented first-party base harness tools.

    Args:
        cwd: Optional working directory passed to each path-aware tool factory.
        permissions: Optional permission policy threaded into every tool.
        approval_callback: Optional approval callback threaded into every tool.
        include: Optional allowlist of tool names to build, in standard order.
        exclude: Optional denylist of tool names to drop.
        extra_names: Optional names of application-added tools that the
            `include`/`exclude` selectors may reference. These names are
            accepted by selector validation but never built here, so an
            application can pass one shared selector across the base tools and
            its own tools without the base selectors raising on names this
            function does not own. See `BASE_TOOL_NAMES` for the owned set.

    Returns:
        tuple[HarnessToolDefinition, ...]: The selected base tools, excluding
        any `extra_names` the selectors referenced.
    """
    selected_names = _select_base_tool_names(
        include=include,
        exclude=exclude,
        extra_names=extra_names,
    )
    factories: dict[str, _BaseToolFactory] = {
        "read": lambda: read_tool(
            cwd=cwd,
            permissions=permissions,
            approval_callback=approval_callback,
        ),
        "find": lambda: find_tool(
            cwd=cwd,
            permissions=permissions,
            approval_callback=approval_callback,
        ),
        "grep": lambda: grep_tool(
            cwd=cwd,
            permissions=permissions,
            approval_callback=approval_callback,
        ),
        "patch": lambda: patch_tool(
            cwd=cwd,
            permissions=permissions,
            approval_callback=approval_callback,
        ),
        "write": lambda: write_tool(
            cwd=cwd,
            permissions=permissions,
            approval_callback=approval_callback,
        ),
        _PLAN_TOOL_NAME: plan_tool,
        "bash": lambda: bash_tool(
            cwd=cwd,
            permissions=permissions,
            approval_callback=approval_callback,
        ),
        "agent": agent_tool,
    }

    return tuple(factories[name]() for name in selected_names)


def render_harness_tools_prompt(
    *,
    definitions: Sequence[HarnessToolDefinition],
    prompt_guidelines: Sequence[str] = (),
) -> str | None:
    """Render the compact system-prompt block for harness tool metadata."""
    snippets = [
        f"- {definition.tool.name}: {definition.prompt_snippet}"
        for definition in definitions
        if definition.prompt_snippet is not None
    ]
    guidelines = _dedupe_preserving_order(
        chain(
            prompt_guidelines,
            (
                guideline
                for definition in definitions
                for guideline in definition.prompt_guidelines
            ),
        )
    )

    sections: list[str] = []
    if snippets:
        sections.append("Available tools:\n" + "\n".join(snippets))
    if guidelines:
        sections.append("Guidelines:\n" + "\n".join(f"- {item}" for item in guidelines))
    if not sections:
        return None
    return "<default_tools>\n" + "\n\n".join(sections) + "\n</default_tools>"


def _validate_unique_tool_names(
    definitions: tuple[HarnessToolDefinition, ...],
) -> None:
    """Reject duplicate tool names before they reach model tool schemas."""
    seen: set[str] = set()
    for definition in definitions:
        tool_name = definition.tool.name
        if tool_name in seen:
            raise ValueError(f"Duplicate harness tool name: {tool_name}")
        seen.add(tool_name)


def _select_base_tool_names(
    *,
    include: Iterable[str] | None,
    exclude: Iterable[str] | None,
    extra_names: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return the requested standard tool names in stable base-tool order.

    Selector names that appear in `extra_names` are accepted but never
    returned, since they belong to application-added tools rather than the
    base set. This keeps base-tool selection valid when an application shares
    one selector across the base tools and its own tools.
    """
    known_names = _BASE_TOOL_NAME_SET | _normalize_extra_names(extra_names)
    include_names = _validate_base_tool_selector(
        include, label="include", known_names=known_names
    )
    exclude_names = _validate_base_tool_selector(
        exclude, label="exclude", known_names=known_names
    )

    overlap = include_names & exclude_names
    if overlap:
        raise ValueError(
            "base_harness_tools include/exclude selectors overlap: "
            f"{_format_tool_names(overlap)}"
        )

    return tuple(
        name
        for name in BASE_TOOL_NAMES
        if (include is None or name in include_names) and name not in exclude_names
    )


def _normalize_extra_names(extra_names: Iterable[str] | None) -> frozenset[str]:
    """Return application-owned selector names accepted by base validation."""
    if extra_names is None:
        return frozenset()
    if isinstance(extra_names, str):
        return frozenset((extra_names,))
    return frozenset(extra_names)


def _validate_base_tool_selector(
    selector: Iterable[str] | None,
    *,
    label: Literal["include", "exclude"],
    known_names: frozenset[str],
) -> set[str]:
    """Validate one base-tool selector collection against the base tool set.

    Only base-tool names are returned; names that resolve to `extra_names`
    application tools are dropped because they are not built here.
    """
    if selector is None:
        return set()

    selected = {selector} if isinstance(selector, str) else set(selector)
    unknown = selected - known_names
    if unknown:
        raise ValueError(
            f"Unknown base_harness_tools {label} selector(s): "
            f"{_format_tool_names(unknown)}. Expected one of: "
            f"{_format_tool_names(BASE_TOOL_NAMES)}"
        )
    return selected & _BASE_TOOL_NAME_SET


def _dedupe_preserving_order(items: Iterable[str]) -> tuple[str, ...]:
    """Return non-empty strings once, preserving first occurrence order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        normalized = item.strip()
        if normalized == "" or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return tuple(deduped)


def _prompt_marker_key(shim_name: str) -> str:
    """Return the persisted shim-state key for prompt append deduplication."""
    return f"{shim_name}:{_PROMPT_MARKER_KEY_SUFFIX}"


def _format_tool_names(names: Iterable[str]) -> str:
    """Return names in deterministic comma-separated form."""
    name_set = set(names)
    ordered = [name for name in BASE_TOOL_NAMES if name in name_set]
    ordered.extend(sorted(name_set - _BASE_TOOL_NAME_SET))
    return ", ".join(ordered)
