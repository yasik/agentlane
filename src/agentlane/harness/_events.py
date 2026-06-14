"""High-level harness run events and event-stream handle.

Run events describe one high-level harness run as a single ordered async
stream: model deltas, LLM lifecycle, tool lifecycle, agent lifecycle, approval
traffic, and compact state snapshots.

Lineage and scope semantics (see the "Run Events" section of
``docs/harness/runner.md`` for the full contract):

- Every task-carrying event exposes ``parent_task_id`` and ``is_root``. The
  emitter stamps the first agent run's ``task_id`` as the stream root; events
  whose ``task_id`` matches it are root events (``is_root`` is ``True`` and
  ``parent_task_id`` is ``None``). In the default single-agent-plus-delegation
  topology every event in one stream is a root event, because a delegated child
  runs in its own runtime turn and surfaces to the parent only as a tool call.
- Tool start/end events for agent-as-tool and handoff delegation set
  ``is_delegation`` so consumers identify delegation without name heuristics.
- ``RunAgentEndEvent`` fires for child tasks with that child's result when an
  application shares run hooks across agents; treating it as "the whole run
  ended" records child telemetry as root telemetry. Gate on ``is_root``.
- State snapshots are root-stream-only by contract: a delegated child never
  emits ``RunStateSnapshotEvent`` into the parent stream.
"""

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from agentlane.models import (
    MessageDict,
    ModelResponse,
    ModelStreamEvent,
    PlanStepStatus,
    ToolCall,
    ToolError,
)

from ._hooks import RunnerHooks
from ._run import RunResult, RunState, copy_generic_value
from ._stream_base import BaseRunStream
from ._task import Task
from ._tool_result import as_plan_update, tool_outcome

if TYPE_CHECKING:
    from .tools._approvals import ToolApprovalEvent

_RUN_EVENT_STREAM_END = object()


class RunEventKind(StrEnum):
    """Normalized event kinds emitted by high-level harness run streams."""

    MODEL_STREAM = "model_stream"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOOL_APPROVAL = "tool_approval"
    HANDOFF_START = "handoff_start"
    HANDOFF_END = "handoff_end"
    STATE_SNAPSHOT = "state_snapshot"
    PLAN_UPDATED = "plan_updated"


class RunStateSnapshotBoundary(StrEnum):
    """Stable run boundaries where compact state snapshots are emitted."""

    RUN_START = "run_start"
    TURN_PREPARED = "turn_prepared"
    TOOL_ROUND_END = "tool_round_end"
    RUN_END = "run_end"


@dataclass(frozen=True, slots=True)
class RunStateSnapshot:
    """Compact full state snapshot for a stable point in one run."""

    turn_count: int
    """Number of model turns started so far (see ``RunState.turn_count``)."""
    history_length: int
    response_count: int
    shim_state: dict[str, object]


@dataclass(frozen=True, slots=True)
class RunModelStreamEvent:
    """Run event wrapping one existing model stream event."""

    event: ModelStreamEvent
    kind: Literal[RunEventKind.MODEL_STREAM] = field(
        default=RunEventKind.MODEL_STREAM,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class RunAgentStartEvent:
    """Run event emitted when one agent run starts."""

    task_name: str
    task_id: str
    parent_task_id: str | None = None
    """Task id of the parent run, or ``None`` for the stream root."""
    is_root: bool = True
    """Whether this event belongs to the stream's root run."""
    kind: Literal[RunEventKind.AGENT_START] = field(
        default=RunEventKind.AGENT_START,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class RunAgentEndEvent:
    """Run event emitted when one agent run ends.

    Fires for child tasks too (with that child's ``result``) when run hooks are
    shared across agents. Consumers that report on the whole run must gate on
    ``is_root`` so a finishing child does not clobber root telemetry.
    """

    task_name: str
    task_id: str
    result: RunResult | None
    parent_task_id: str | None = None
    """Task id of the parent run, or ``None`` for the stream root."""
    is_root: bool = True
    """Whether this event belongs to the stream's root run."""
    kind: Literal[RunEventKind.AGENT_END] = field(
        default=RunEventKind.AGENT_END,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class RunLLMStartEvent:
    """Run event emitted before one model request starts."""

    task_name: str
    task_id: str
    messages: list[MessageDict]
    parent_task_id: str | None = None
    """Task id of the parent run, or ``None`` for the stream root."""
    is_root: bool = True
    """Whether this event belongs to the stream's root run."""
    kind: Literal[RunEventKind.LLM_START] = field(
        default=RunEventKind.LLM_START,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class RunLLMEndEvent:
    """Run event emitted after one model request completes."""

    task_name: str
    task_id: str
    response: ModelResponse
    parent_task_id: str | None = None
    """Task id of the parent run, or ``None`` for the stream root."""
    is_root: bool = True
    """Whether this event belongs to the stream's root run."""
    kind: Literal[RunEventKind.LLM_END] = field(
        default=RunEventKind.LLM_END,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class RunToolStartEvent:
    """Run event emitted before one tool call starts."""

    task_name: str
    task_id: str
    tool_call: ToolCall
    is_delegation: bool = False
    """Whether this tool call delegates to another agent (agent-as-tool/handoff)."""
    parent_task_id: str | None = None
    """Task id of the parent run, or ``None`` for the stream root."""
    is_root: bool = True
    """Whether this event belongs to the stream's root run."""
    kind: Literal[RunEventKind.TOOL_START] = field(
        default=RunEventKind.TOOL_START,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class RunToolEndEvent:
    """Run event emitted after one tool call ends.

    The runner derives ``ok``/``error`` from the raw ``result`` so consumers get
    a framework-supplied success/failure signal without reading result wording.
    The model-facing text path is unchanged: ``result`` is the raw handler
    return value, rendered for the model by the tool's own formatter.
    """

    task_name: str
    task_id: str
    tool_call: ToolCall
    result: object
    ok: bool = True
    """Framework-derived success flag for this tool call."""
    error: ToolError | None = None
    """Typed failure payload when the call failed, otherwise ``None``."""
    is_delegation: bool = False
    """Whether this tool call delegated to another agent (agent-as-tool/handoff)."""
    parent_task_id: str | None = None
    """Task id of the parent run, or ``None`` for the stream root."""
    is_root: bool = True
    """Whether this event belongs to the stream's root run."""
    kind: Literal[RunEventKind.TOOL_END] = field(
        default=RunEventKind.TOOL_END,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class RunToolApprovalEvent:
    """Run event wrapping one brokered tool-approval lifecycle event."""

    event: "ToolApprovalEvent"
    kind: Literal[RunEventKind.TOOL_APPROVAL] = field(
        default=RunEventKind.TOOL_APPROVAL,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class RunHandoffStartEvent:
    """Run event emitted before first-class handoff control transfer starts.

    A handoff is itself a delegation; the lineage fields describe the
    transferring (parent) run, which is the stream root in the default topology.
    """

    task_name: str
    task_id: str
    tool_call: ToolCall
    target_name: str
    parent_task_id: str | None = None
    """Task id of the parent run, or ``None`` for the stream root."""
    is_root: bool = True
    """Whether this event belongs to the stream's root run."""
    kind: Literal[RunEventKind.HANDOFF_START] = field(
        default=RunEventKind.HANDOFF_START,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class RunHandoffEndEvent:
    """Run event emitted after first-class handoff control transfer ends.

    A handoff is itself a delegation; the lineage fields describe the
    transferring (parent) run, which is the stream root in the default topology.
    """

    task_name: str
    task_id: str
    tool_call: ToolCall
    target_name: str
    result: RunResult
    parent_task_id: str | None = None
    """Task id of the parent run, or ``None`` for the stream root."""
    is_root: bool = True
    """Whether this event belongs to the stream's root run."""
    kind: Literal[RunEventKind.HANDOFF_END] = field(
        default=RunEventKind.HANDOFF_END,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class RunStateSnapshotEvent:
    """Run event emitted with a compact full state snapshot."""

    boundary: RunStateSnapshotBoundary
    snapshot: RunStateSnapshot
    kind: Literal[RunEventKind.STATE_SNAPSHOT] = field(
        default=RunEventKind.STATE_SNAPSHOT,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class RunPlanItem:
    """One step in a plan-updated run event payload."""

    step: str
    """Concise description of this plan step."""

    status: PlanStepStatus
    """Current status of this plan step."""


@dataclass(frozen=True, slots=True)
class RunPlanUpdatedEvent:
    """Run event emitted when the first-party plan tool records a new plan.

    Carries the structured plan so consumers render plan UX from typed data
    instead of string-matching the plan tool's private success message.
    """

    task_name: str
    task_id: str
    tool_call: ToolCall
    plan: tuple[RunPlanItem, ...]
    explanation: str | None = None
    """Optional model-supplied reason for this plan update."""
    parent_task_id: str | None = None
    """Task id of the parent run, or ``None`` for the stream root."""
    is_root: bool = True
    """Whether this event belongs to the stream's root run."""
    kind: Literal[RunEventKind.PLAN_UPDATED] = field(
        default=RunEventKind.PLAN_UPDATED,
        init=False,
    )


type RunEvent = (
    RunModelStreamEvent
    | RunAgentStartEvent
    | RunAgentEndEvent
    | RunLLMStartEvent
    | RunLLMEndEvent
    | RunToolStartEvent
    | RunToolEndEvent
    | RunToolApprovalEvent
    | RunHandoffStartEvent
    | RunHandoffEndEvent
    | RunStateSnapshotEvent
    | RunPlanUpdatedEvent
)
"""Public union of high-level harness run events."""


class RunEventStream(BaseRunStream[RunEvent]):
    """Async stream handle for one high-level harness event run."""

    def __init__(
        self,
        *,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """Initialize one run event stream handle."""
        super().__init__(end_sentinel=_RUN_EVENT_STREAM_END, on_close=on_close)


class _RunLineage:
    """Tracks the stream's root run so events can be stamped with lineage.

    The first agent run observed in one stream is treated as the root. Every
    later event compares its ``task_id`` against that latched root: a match is a
    root event; a mismatch (only possible when an application shares run hooks
    across delegated agents) is a child event whose parent is the stream root.
    """

    def __init__(self) -> None:
        self._root_task_id: str | None = None

    def observe_root(self, task_id: str) -> None:
        """Latch the first observed run as the stream root."""
        if self._root_task_id is None:
            self._root_task_id = task_id

    def of(self, task_id: str) -> tuple[str | None, bool]:
        """Return ``(parent_task_id, is_root)`` for one task id."""
        if self._root_task_id is None or task_id == self._root_task_id:
            return None, True
        return self._root_task_id, False


class RunEventEmitter:
    """Internal adapter that converts runner callbacks into run events."""

    def __init__(self, emit: Callable[[RunEvent], None]) -> None:
        self._emit = emit
        self._lineage = _RunLineage()
        self._delegation_tool_names: set[str] = set()
        self._hooks = _RunEventHookAdapter(emit, self._lineage, self)

    @property
    def hooks(self) -> RunnerHooks:
        """Return hook adapter used to observe existing runner callbacks."""
        return self._hooks

    def register_delegation_tool_names(self, names: frozenset[str]) -> None:
        """Record which visible tool names delegate to another agent.

        The runner classifies delegation tools structurally (by tool-definition
        type), so the emitter can tag delegation tool events without inspecting
        tool names heuristically.
        """
        self._delegation_tool_names |= names

    def is_delegation_tool(self, tool_call: ToolCall) -> bool:
        """Return whether a tool call targets a registered delegation tool."""
        return (tool_call.function.name or "") in self._delegation_tool_names

    def model_stream_event(self, event: ModelStreamEvent) -> None:
        """Emit a wrapped model stream event."""
        self._emit(RunModelStreamEvent(event=event))

    def tool_approval_event(self, event: "ToolApprovalEvent") -> None:
        """Emit a wrapped tool-approval lifecycle event."""
        self._emit(RunToolApprovalEvent(event=event))

    def state_snapshot(
        self,
        boundary: RunStateSnapshotBoundary,
        state: RunState,
    ) -> None:
        """Emit one compact state snapshot event."""
        self._emit(
            RunStateSnapshotEvent(
                boundary=boundary,
                snapshot=_compact_state_snapshot(state),
            )
        )

    def plan_updated(
        self,
        *,
        task: Task,
        tool_call: ToolCall,
        plan: tuple[RunPlanItem, ...],
        explanation: str | None,
    ) -> None:
        """Emit one structured plan-updated event."""
        task_id = str(task.task_id)
        parent_task_id, is_root = self._lineage.of(task_id)
        self._emit(
            RunPlanUpdatedEvent(
                task_name=_task_name(task),
                task_id=task_id,
                tool_call=tool_call,
                plan=plan,
                explanation=explanation,
                parent_task_id=parent_task_id,
                is_root=is_root,
            )
        )

    def handoff_start(
        self,
        *,
        task: Task,
        tool_call: ToolCall,
        target_name: str,
    ) -> None:
        """Emit one handoff-start event."""
        task_id = str(task.task_id)
        parent_task_id, is_root = self._lineage.of(task_id)
        self._emit(
            RunHandoffStartEvent(
                task_name=_task_name(task),
                task_id=task_id,
                tool_call=tool_call,
                target_name=target_name,
                parent_task_id=parent_task_id,
                is_root=is_root,
            )
        )

    def handoff_end(
        self,
        *,
        task: Task,
        tool_call: ToolCall,
        target_name: str,
        result: RunResult,
    ) -> None:
        """Emit one handoff-end event."""
        task_id = str(task.task_id)
        parent_task_id, is_root = self._lineage.of(task_id)
        self._emit(
            RunHandoffEndEvent(
                task_name=_task_name(task),
                task_id=task_id,
                tool_call=tool_call,
                target_name=target_name,
                result=result,
                parent_task_id=parent_task_id,
                is_root=is_root,
            )
        )


class _RunEventHookAdapter(RunnerHooks):
    """Runner hook adapter that emits high-level run events."""

    def __init__(
        self,
        emit: Callable[[RunEvent], None],
        lineage: _RunLineage,
        emitter: RunEventEmitter,
    ) -> None:
        self._emit = emit
        self._lineage = lineage
        self._emitter = emitter

    async def on_agent_start(
        self,
        task: Task,
        state: RunState,
    ) -> None:
        del state
        task_id = str(task.task_id)
        self._lineage.observe_root(task_id)
        parent_task_id, is_root = self._lineage.of(task_id)
        self._emit(
            RunAgentStartEvent(
                task_name=_task_name(task),
                task_id=task_id,
                parent_task_id=parent_task_id,
                is_root=is_root,
            )
        )

    async def on_agent_end(
        self,
        task: Task,
        result: RunResult | None,
    ) -> None:
        task_id = str(task.task_id)
        parent_task_id, is_root = self._lineage.of(task_id)
        self._emit(
            RunAgentEndEvent(
                task_name=_task_name(task),
                task_id=task_id,
                result=result,
                parent_task_id=parent_task_id,
                is_root=is_root,
            )
        )

    async def on_llm_start(
        self,
        task: Task,
        messages: list[MessageDict],
    ) -> None:
        task_id = str(task.task_id)
        parent_task_id, is_root = self._lineage.of(task_id)
        self._emit(
            RunLLMStartEvent(
                task_name=_task_name(task),
                task_id=task_id,
                messages=[dict(message) for message in messages],
                parent_task_id=parent_task_id,
                is_root=is_root,
            )
        )

    async def on_llm_end(
        self,
        task: Task,
        response: ModelResponse,
    ) -> None:
        task_id = str(task.task_id)
        parent_task_id, is_root = self._lineage.of(task_id)
        self._emit(
            RunLLMEndEvent(
                task_name=_task_name(task),
                task_id=task_id,
                response=response,
                parent_task_id=parent_task_id,
                is_root=is_root,
            )
        )

    async def on_tool_call_start(
        self,
        task: Task,
        tool_call: ToolCall,
    ) -> None:
        task_id = str(task.task_id)
        parent_task_id, is_root = self._lineage.of(task_id)
        self._emit(
            RunToolStartEvent(
                task_name=_task_name(task),
                task_id=task_id,
                tool_call=tool_call,
                is_delegation=self._emitter.is_delegation_tool(tool_call),
                parent_task_id=parent_task_id,
                is_root=is_root,
            )
        )

    async def on_tool_call_end(
        self,
        task: Task,
        tool_call: ToolCall,
        result: object,
    ) -> None:
        task_id = str(task.task_id)
        parent_task_id, is_root = self._lineage.of(task_id)
        outcome = tool_outcome(result)
        self._emit(
            RunToolEndEvent(
                task_name=_task_name(task),
                task_id=task_id,
                tool_call=tool_call,
                result=result,
                ok=outcome.ok,
                error=outcome.error,
                is_delegation=self._emitter.is_delegation_tool(tool_call),
                parent_task_id=parent_task_id,
                is_root=is_root,
            )
        )
        plan_update = as_plan_update(result)
        if plan_update is not None:
            self._emitter.plan_updated(
                task=task,
                tool_call=tool_call,
                plan=tuple(
                    RunPlanItem(step=step, status=status)
                    for step, status in plan_update.plan_steps
                ),
                explanation=plan_update.plan_explanation,
            )


def _compact_state_snapshot(state: RunState) -> RunStateSnapshot:
    """Return a compact full snapshot of one run state."""
    return RunStateSnapshot(
        turn_count=state.turn_count,
        history_length=len(state.history),
        response_count=len(state.responses),
        shim_state={
            key: copy_generic_value(value) for key, value in state.shim_state.items()
        },
    )


def _task_name(task: Task) -> str:
    """Return the task's configured name when available."""
    name = getattr(task, "name", None)
    if isinstance(name, str):
        return name
    return type(task).__name__


async def forward_tool_approval_events(
    approval_events: AsyncIterator["ToolApprovalEvent"],
    *,
    event_emitter: RunEventEmitter,
) -> None:
    """Forward brokered approval lifecycle events into a run-event stream."""
    async for event in approval_events:
        event_emitter.tool_approval_event(event)
