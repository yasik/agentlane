"""Shim integration for first-party harness tool definitions."""

from collections.abc import Iterable, Sequence
from typing import Any

from agentlane.models.run import RunContext

from .._run import RunState
from .._tooling import merge_tools
from ..shims import BoundShim, PreparedTurn, Shim, ShimBindingContext
from ._read import read_tool
from ._types import HarnessToolDefinition

_PROMPT_MARKER_KEY_SUFFIX = "prompt-appended"


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
        self._definitions = definitions
        self._prompt_block = prompt_block

    async def on_run_start(
        self,
        state: RunState,
        transient_state: RunContext[Any],
    ) -> None:
        del state
        del transient_state

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        executable_tools = tuple(definition.tool for definition in self._definitions)
        turn.tools = merge_tools(turn.tools, executable_tools)

        if self._prompt_block is None:
            return
        if turn.run_state.turn_count != 1:
            return

        marker_key = _prompt_marker_key(self._shim_name)
        if turn.run_state.shim_state.get(marker_key) is True:
            return

        turn.append_system_instruction(self._prompt_block)
        turn.run_state.shim_state[marker_key] = True


class HarnessToolsShim(Shim):
    """First-party shim that exposes harness tools and prompt guidance."""

    def __init__(
        self,
        definitions: Sequence[HarnessToolDefinition],
        *,
        name: str = "harness-tools",
    ) -> None:
        self._definitions = tuple(definitions)
        self._name = name
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
            ),
        )


def base_harness_tools() -> tuple[HarnessToolDefinition, ...]:
    """Return currently implemented first-party base harness tools."""
    return (read_tool(),)


def render_harness_tools_prompt(
    *,
    definitions: Sequence[HarnessToolDefinition],
) -> str | None:
    """Render the compact system-prompt block for harness tool metadata."""
    snippets = [
        f"- {definition.tool.name}: {definition.prompt_snippet}"
        for definition in definitions
        if definition.prompt_snippet is not None
    ]
    guidelines = _dedupe_preserving_order(
        guideline
        for definition in definitions
        for guideline in definition.prompt_guidelines
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
