"""Minimal run-state contracts and copy utilities for the harness.

This module defines the canonical data shapes that flow between the agent
lifecycle and the runner.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel

from agentlane.messaging import AgentId
from agentlane.models import MessageDict, ModelResponse, PromptSpec, RunStateView
from agentlane.models.run import DefaultRunContext

ACTIVE_SKILL_NAMES_STATE_KEY_SUFFIX = ":active-skill-names"
"""Documented ``shim_state`` key suffix that holds active skill names.

A skills shim records the names of the skills active for the current run under
a key ending in this suffix (the shim name is the prefix, so multiple skills
shims never collide). ``RunStateView.active_skill_names`` reads every such key
so tools can resolve skill-relative resources without coupling to a specific
shim name or reaching for a private key. The value must be a list of strings.
"""


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


type RunInstructions = str | PromptSpec[Any] | None
"""Supported persisted system-instruction source at the harness boundary."""


type RunHistoryItem = MessageDict | ModelResponse | PromptSpec[Any] | RunMessageContent
"""Supported heterogeneous items stored in run input and persisted history.

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

    instructions: RunInstructions
    """Single persisted system instruction for this run."""

    history: list[RunHistoryItem]
    """Append-only persisted conversation history for this run.

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
        instructions=copy_instructions(run_state.instructions),
        history=[copy_history_item(item) for item in run_state.history],
        responses=list(run_state.responses),
        shim_state=copy_shim_state(run_state.shim_state),
        turn_count=run_state.turn_count,
    )


def copy_shim_state(shim_state: ShimState) -> ShimState:
    """Return an isolated copy of one persisted shim-state container."""
    return ShimState(
        context={key: copy_generic_value(value) for key, value in shim_state.items()}
    )


def copy_instructions(instructions: RunInstructions) -> RunInstructions:
    """Copy one persisted instruction source for state ownership."""
    if instructions is None or isinstance(instructions, str):
        return instructions

    copied_values = (
        cast(Any, copy_generic_value(instructions.values))
        if instructions.values is not None
        else None
    )
    return PromptSpec(
        template=instructions.template,
        values=copied_values,
    )


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


@dataclass(frozen=True, slots=True)
class LiveRunStateView(RunStateView):
    """Read-only ``RunStateView`` backed by one live harness ``RunState``.

    Cheap to construct: it wraps the live ``RunState`` and the run's task
    identity by reference, reading through on each access. The runner builds
    one per local tool call and stamps it onto ``ToolExecutionContext`` so tool
    handlers can observe the run without an app-built side channel.
    """

    _run_state: RunState
    _task_id: AgentId

    @property
    def task_id(self) -> str:
        """Return the stable task identity (``str`` of the run's task id)."""
        return str(self._task_id)

    @property
    def shim_state(self) -> Mapping[str, object]:
        """Return the live persisted shim state as a read-only mapping."""
        return self._run_state.shim_state

    @property
    def active_skill_names(self) -> tuple[str, ...]:
        """Return active skill names read from the documented state keys."""
        names: list[str] = []
        for key, value in self._run_state.shim_state.items():
            if not key.endswith(ACTIVE_SKILL_NAMES_STATE_KEY_SUFFIX):
                continue
            if not isinstance(value, list):
                continue
            for entry in cast(list[object], value):
                if isinstance(entry, str) and entry not in names:
                    names.append(entry)
        return tuple(names)
