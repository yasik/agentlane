"""Minimal run-state contracts for the harness."""

from dataclasses import dataclass

from agentlane.models import ModelResponse


@dataclass(slots=True)
class RunState:
    """Minimal resumable state for one harness agent run."""

    original_input: str | list[object]
    """Original input that started the run."""

    continuation_history: list[object]
    """Accumulated continuation items for later turns."""

    responses: list[ModelResponse]
    """Raw model responses accumulated across turns."""

    turn_count: int = 0
    """Number of model turns completed for this run."""


type RunInput = str | list[object] | RunState
"""Public input accepted by the default harness agent."""


@dataclass(slots=True)
class RunResult:
    """Minimal final result returned by the default harness agent."""

    final_output: object
    """Final output extracted from the terminal run turn."""

    responses: list[ModelResponse]
    """Raw model responses accumulated across the run."""

    turn_count: int
    """Number of model turns completed for this run."""
