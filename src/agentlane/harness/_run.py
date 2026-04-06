"""Minimal run-state contracts and copy utilities for the harness.

This module defines the canonical data shapes that flow between the agent
lifecycle and the runner.
"""

from dataclasses import dataclass
from typing import cast

from agentlane.models import ModelResponse


@dataclass(slots=True)
class RunState:
    """Minimal resumable state for one harness agent run.

    The lifecycle creates private working copies before handing state to the
    runner, so failed turns never corrupt the persisted baseline.
    """

    original_input: str | list[object]
    """Original input that started the run."""

    continuation_history: list[object]
    """Accumulated continuation items for later turns.

    Items are heterogeneous: strings, ``ModelResponse`` objects (assistant
    turns), ``PromptSpec`` objects, or arbitrary user-side payloads. The
    runner resolves each item into canonical ``MessageDict`` at request time.
    """

    responses: list[ModelResponse]
    """Raw model responses accumulated across turns."""

    turn_count: int = 0
    """Number of model turns completed for this run."""


type RunInput = str | list[object] | RunState
"""Public input accepted by the default harness agent.

A plain ``str`` starts or continues a conversation with a single user
message. A ``list[object]`` provides a richer multi-item payload (e.g.
a ``PromptSpec`` mixed with prior ``ModelResponse`` objects). A
``RunState`` resumes a previously persisted conversation wholesale.
"""


@dataclass(slots=True)
class RunResult:
    """Minimal final result returned by the default harness agent."""

    final_output: object
    """Final output extracted from the terminal run turn."""

    responses: list[ModelResponse]
    """Raw model responses accumulated across the run."""

    turn_count: int
    """Number of model turns completed for this run."""


def copy_run_state(run_state: RunState | None) -> RunState | None:
    """Return an isolated copy of one run state, or ``None`` passthrough."""
    if run_state is None:
        return None

    return RunState(
        original_input=copy_original_input(run_state.original_input),
        continuation_history=[
            copy_item(item) for item in run_state.continuation_history
        ],
        responses=list(run_state.responses),
        turn_count=run_state.turn_count,
    )


def copy_original_input(original_input: str | list[object]) -> str | list[object]:
    """Copy the original input for state ownership.

    Strings are immutable and returned as-is. Lists are shallow-copied so
    the new state owns its own container without deep-cloning every item.
    """
    if isinstance(original_input, str):
        return original_input

    return [copy_item(item) for item in original_input]


def copy_item(item: object) -> object:
    """Copy one generic run item when shallow ownership is needed.

    Only mutable containers (lists) are copied. Everything else —
    strings, ``ModelResponse``, ``PromptSpec`` — is treated as immutable.
    """
    if isinstance(item, list):
        return list(cast(list[object], item))
    if isinstance(item, dict):
        return dict(cast(dict[str, object], item))
    return item
