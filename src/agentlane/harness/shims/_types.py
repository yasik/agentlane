"""Shared shim context and turn-preparation types."""

from dataclasses import dataclass, field, replace
from typing import Any

from agentlane.models import Tools, ToolSpec, render_instruction_text
from agentlane.models.run import DefaultRunContext, RunContext

from .._run import RunHistoryItem, RunInstructions, RunState, copy_history_item
from .._task import Task
from .._tooling import merge_tools
from ._errors import ToolNameCollisionError


def _default_transient_state() -> RunContext[Any]:
    """Return the default per-run transient state container."""
    return DefaultRunContext()


@dataclass(slots=True)
class ShimBindingContext:
    """Static binding data for one shim on one bound agent instance."""

    task: Task
    """Bound harness task or agent that owns this shim session."""


@dataclass(slots=True)
class PreparedTurn:
    """Mutable working state for one model turn.

    Shims may use this object to adjust the persisted run state, visible tools,
    or model args before the runner builds the next model request.
    """

    run_state: RunState
    """Private working run state for the current run."""

    tools: Tools | None
    """Effective visible tools for this turn."""

    model_args: dict[str, object] | None
    """Effective provider/model arguments for this turn."""

    transient_state: RunContext[Any] = field(default_factory=_default_transient_state)
    """Per-run transient state shared across shim callbacks.

    This state is intentionally ephemeral. It lives for the duration of one
    run, is shared across all turns in that run, and is discarded when the run
    ends. Shims that need resumable state should write to
    `PreparedTurn.run_state.shim_state` instead.
    """

    _excluded_tool_names: set[str] = field(
        default_factory=set[str], init=False, repr=False
    )
    _unique_tool_names: set[str] = field(
        default_factory=set[str], init=False, repr=False
    )
    _contributed_tool_names: set[str] = field(
        default_factory=set[str], init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.tools is not None:
            self._contributed_tool_names.update(
                tool.name for tool in self.tools.normalized_tools
            )

    def add_tools(
        self,
        tools: tuple[ToolSpec[Any], ...],
        *,
        require_unique_names: bool = False,
    ) -> None:
        """Merge tools and reject ambiguous names before deduplication.

        Existing local contributions keep first-wins precedence. Sources that
        require unique names reserve them for the whole preparation pass,
        including after another shim filters the current tool configuration.
        """
        existing: set[str] = self._contributed_tool_names | (
            {tool.name for tool in self.tools.normalized_tools}
            if self.tools is not None
            else set[str]()
        )
        added: set[str] = set()
        for tool in tools:
            if tool.name in self._unique_tool_names or (
                require_unique_names and tool.name in existing | added
            ):
                raise ToolNameCollisionError(
                    f"Tool name {tool.name!r} collides with another tool contribution."
                )
            added.add(tool.name)
        self.tools = merge_tools(self.tools, tools)
        self._contributed_tool_names.update(existing | added)
        if require_unique_names:
            self._unique_tool_names.update(added)

    def exclude_tools(self, names: frozenset[str]) -> None:
        """Exclude named tools now and after all shim contributions.

        Keep an empty configuration during preparation so later contributions
        retain the configured scheduling and execution settings.
        """
        self._excluded_tool_names.update(names)
        self.apply_tool_exclusions()

    def apply_tool_exclusions(self) -> None:
        """Apply exclusions to the current complete tool contribution."""
        if self.tools is None or not self._excluded_tool_names:
            return
        self.tools = replace(
            self.tools,
            tools=tuple(
                tool
                for tool in self.tools.normalized_tools
                if tool.name not in self._excluded_tool_names
            ),
        )

    def set_system_instruction(self, value: RunInstructions) -> None:
        """Replace the single persisted system instruction explicitly."""
        self.run_state.instructions = value

    def append_system_instruction(
        self,
        text: str,
        *,
        separator: str = "\n\n",
    ) -> None:
        """Append text to the tail of the persisted system instruction."""
        current = self.run_state.instructions
        if current is None:
            self.run_state.instructions = text
            return
        if isinstance(current, str):
            self.run_state.instructions = f"{current}{separator}{text}"
            return
        rendered = render_instruction_text(current)
        self.run_state.instructions = f"{rendered}{separator}{text}"

    def append_history_item(self, item: RunHistoryItem) -> None:
        """Append one item to persisted conversation history."""
        self.run_state.history.append(copy_history_item(item))

    def append_history_items(self, items: list[RunHistoryItem]) -> None:
        """Append multiple items to persisted conversation history."""
        for item in items:
            self.append_history_item(item)

    def replace_history(self, items: list[RunHistoryItem]) -> None:
        """Replace persisted conversation history with copied items."""
        self.run_state.history = [copy_history_item(item) for item in items]
