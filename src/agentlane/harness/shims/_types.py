"""Shared shim context and turn-preparation types."""

from dataclasses import dataclass, field
from typing import Any

from agentlane.models import PromptSpec, Tools
from agentlane.models.run import DefaultRunContext, RunContext

from .._run import RunHistoryItem, RunState
from .._task import Task


def _empty_context_items() -> list[RunHistoryItem]:
    """Return one typed empty list for per-turn context items."""
    return []


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

    Shims may use this object to adjust the effective instructions, tools,
    model args, working run state, or one-turn transient context before the
    runner builds the model request.
    """

    run_state: RunState
    """Private working run state for the current run."""

    instructions: str | PromptSpec[Any] | None
    """Effective instructions source for this turn."""

    tools: Tools | None
    """Effective visible tools for this turn."""

    model_args: dict[str, object] | None
    """Effective provider/model arguments for this turn."""

    context_items: list[RunHistoryItem] = field(default_factory=_empty_context_items)
    """Transient items injected into this turn's request only."""

    transient_state: RunContext[Any] = field(default_factory=_default_transient_state)
    """Per-run transient state shared across shim callbacks.

    This state is intentionally ephemeral. It lives for the duration of one
    run, is shared across all turns in that run, and is discarded when the run
    ends. Shims that need resumable state should write to
    ``PreparedTurn.run_state.shim_state`` instead.
    """
