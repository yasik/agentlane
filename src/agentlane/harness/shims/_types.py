"""Shared shim context and turn-preparation types."""

from dataclasses import dataclass, field
from typing import Any

from agentlane.models import PromptSpec, Tools
from agentlane.models.run import DefaultRunContext, RunContext

from .._run import RunHistoryItem, RunInstructions, RunState, copy_history_item
from .._task import Task


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
    ``PreparedTurn.run_state.shim_state`` instead.
    """

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
        rendered = _render_instruction_text(current)
        self.run_state.instructions = f"{rendered}{separator}{text}"

    def append_history_item(self, item: RunHistoryItem) -> None:
        """Append one item to persisted conversation history."""
        self.run_state.history.append(copy_history_item(item))

    def append_history_items(self, items: list[RunHistoryItem]) -> None:
        """Append multiple items to persisted conversation history."""
        for item in items:
            self.append_history_item(item)


def _render_instruction_text(instructions: PromptSpec[Any]) -> str:
    """Render one prompt spec into a single system-instruction string."""
    messages = [
        message
        for message in instructions.template.render_messages(instructions.values)
        if message.get("role") == "system"
    ]
    if not messages:
        raise ValueError("PromptSpec must render at least one system-role message.")

    contents: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            contents.append(content)
            continue
        raise ValueError(
            "System PromptSpec content must render to plain text when appended."
        )
    return "\n\n".join(contents)
