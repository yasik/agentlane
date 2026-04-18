"""Minimal run-state contracts and copy utilities for the harness.

This module defines the canonical data shapes that flow between the agent
lifecycle and the runner.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel

from agentlane.models import MessageDict, ModelResponse, PromptSpec
from agentlane.models.run import DefaultRunContext


class ShimState(DefaultRunContext):
    """Persisted shim-owned state stored in one harness `RunState`.

    This keeps the same mapping-style access as `DefaultRunContext` while
    making the persisted nature of shim-owned state explicit at the harness
    boundary.
    """

    def __eq__(self, other: object) -> bool:
        """Compare shim state by stored mapping contents."""
        if isinstance(other, DefaultRunContext):
            return self.context == other.context
        if isinstance(other, Mapping):
            other_mapping = cast(Mapping[str, object], other)
            return self.context == dict(other_mapping)
        return NotImplemented


def _empty_shim_state() -> ShimState:
    """Return one typed empty persisted shim-state container."""
    return ShimState()


type RunMessageContent = (
    str | int | float | bool | None | BaseModel | dict[str, object] | list[object]
)
"""Supported non-message content values at the harness run boundary."""


type RunHistoryItem = MessageDict | ModelResponse | PromptSpec[Any] | RunMessageContent
"""Supported heterogeneous items stored in run input and continuation history.

The harness accepts a small set of structured item kinds:

1. canonical message dicts,
2. prior model responses,
3. prompt specs,
4. user-side content values that can be normalized into message content.
"""


@dataclass(slots=True)
class RunState:
    """Minimal resumable state for one harness agent run.

    The lifecycle creates private working copies before handing state to the
    runner, so failed turns never corrupt the persisted baseline.
    """

    original_input: str | list[RunHistoryItem]
    """Original input that started the run."""

    continuation_history: list[RunHistoryItem]
    """Accumulated continuation items for later turns.

    Items may be prior ``ModelResponse`` assistant turns, canonical message
    dicts, prompt specs, or user-side content values. The runner resolves each
    item into canonical ``MessageDict`` at request time.
    """

    responses: list[ModelResponse]
    """Raw model responses accumulated across turns."""

    shim_state: ShimState = field(default_factory=_empty_shim_state)
    """Persisted shim-owned state that must survive resumed runs."""

    turn_count: int = 0
    """Number of model turns completed for this run."""


type RunInput = str | list[RunHistoryItem] | RunState
"""Public input accepted by the default harness agent.

A plain ``str`` starts or continues a conversation with a single user
message. A ``list[RunHistoryItem]`` provides a richer multi-item payload (e.g.
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

    run_state: RunState | None = None
    """Final resumable run state for this completed run when available."""


def copy_run_state(run_state: RunState | None) -> RunState | None:
    """Return an isolated copy of one run state, or ``None`` passthrough."""
    if run_state is None:
        return None

    return RunState(
        original_input=copy_original_input(run_state.original_input),
        continuation_history=[
            copy_history_item(item) for item in run_state.continuation_history
        ],
        responses=list(run_state.responses),
        shim_state=copy_shim_state(run_state.shim_state),
        turn_count=run_state.turn_count,
    )


def copy_shim_state(shim_state: ShimState) -> ShimState:
    """Return an isolated copy of one persisted shim-state container."""
    return ShimState(
        context={key: copy_generic_value(value) for key, value in shim_state.items()}
    )


def copy_original_input(
    original_input: str | list[RunHistoryItem],
) -> str | list[RunHistoryItem]:
    """Copy the original input for state ownership.

    Strings are immutable and returned as-is. Lists are shallow-copied so
    the new state owns its own container without deep-cloning every item.
    """
    if isinstance(original_input, str):
        return original_input

    return [copy_history_item(item) for item in original_input]


def copy_history_item(item: RunHistoryItem) -> RunHistoryItem:
    """Copy one typed run-history item when shallow ownership is needed.

    Mutable containers and structured ``BaseModel`` payloads are copied.
    Everything else — strings, ``ModelResponse``, ``PromptSpec`` — is treated
    as immutable.
    """
    if isinstance(item, list):
        return list(item)
    if isinstance(item, dict):
        return dict(cast(dict[str, object], item))
    if isinstance(item, BaseModel):
        return cast(RunHistoryItem, item.model_copy(deep=True))
    return item


def copy_generic_value(value: object) -> object:
    """Copy one generic shim-state value when shallow ownership is needed."""
    if isinstance(value, list):
        return list(cast(list[object], value))
    if isinstance(value, dict):
        return dict(cast(dict[str, object], value))
    if isinstance(value, BaseModel):
        return value.model_copy(deep=True)
    return value
