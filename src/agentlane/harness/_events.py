"""High-level harness run events and event-stream handle."""

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from agentlane.models import MessageDict, ModelResponse, ModelStreamEvent, ToolCall

from ._hooks import RunnerHooks
from ._run import RunResult, RunState, copy_generic_value
from ._stream_base import BaseRunStream
from ._task import Task

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
    kind: Literal[RunEventKind.AGENT_START] = field(
        default=RunEventKind.AGENT_START,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class RunAgentEndEvent:
    """Run event emitted when one agent run ends."""

    task_name: str
    task_id: str
    result: RunResult | None
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
    kind: Literal[RunEventKind.TOOL_START] = field(
        default=RunEventKind.TOOL_START,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class RunToolEndEvent:
    """Run event emitted after one tool call ends."""

    task_name: str
    task_id: str
    tool_call: ToolCall
    result: object
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
    """Run event emitted before first-class handoff control transfer starts."""

    task_name: str
    task_id: str
    tool_call: ToolCall
    target_name: str
    kind: Literal[RunEventKind.HANDOFF_START] = field(
        default=RunEventKind.HANDOFF_START,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class RunHandoffEndEvent:
    """Run event emitted after first-class handoff control transfer ends."""

    task_name: str
    task_id: str
    tool_call: ToolCall
    target_name: str
    result: RunResult
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


class RunEventEmitter:
    """Internal adapter that converts runner callbacks into run events."""

    def __init__(self, emit: Callable[[RunEvent], None]) -> None:
        self._emit = emit
        self._hooks = _RunEventHookAdapter(emit)

    @property
    def hooks(self) -> RunnerHooks:
        """Return hook adapter used to observe existing runner callbacks."""
        return self._hooks

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

    def handoff_start(
        self,
        *,
        task: Task,
        tool_call: ToolCall,
        target_name: str,
    ) -> None:
        """Emit one handoff-start event."""
        self._emit(
            RunHandoffStartEvent(
                task_name=_task_name(task),
                task_id=str(task.task_id),
                tool_call=tool_call,
                target_name=target_name,
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
        self._emit(
            RunHandoffEndEvent(
                task_name=_task_name(task),
                task_id=str(task.task_id),
                tool_call=tool_call,
                target_name=target_name,
                result=result,
            )
        )


class _RunEventHookAdapter(RunnerHooks):
    """Runner hook adapter that emits high-level run events."""

    def __init__(self, emit: Callable[[RunEvent], None]) -> None:
        self._emit = emit

    async def on_agent_start(
        self,
        task: Task,
        state: RunState,
    ) -> None:
        del state
        self._emit(
            RunAgentStartEvent(
                task_name=_task_name(task),
                task_id=str(task.task_id),
            )
        )

    async def on_agent_end(
        self,
        task: Task,
        result: RunResult | None,
    ) -> None:
        self._emit(
            RunAgentEndEvent(
                task_name=_task_name(task),
                task_id=str(task.task_id),
                result=result,
            )
        )

    async def on_llm_start(
        self,
        task: Task,
        messages: list[MessageDict],
    ) -> None:
        self._emit(
            RunLLMStartEvent(
                task_name=_task_name(task),
                task_id=str(task.task_id),
                messages=[dict(message) for message in messages],
            )
        )

    async def on_llm_end(
        self,
        task: Task,
        response: ModelResponse,
    ) -> None:
        self._emit(
            RunLLMEndEvent(
                task_name=_task_name(task),
                task_id=str(task.task_id),
                response=response,
            )
        )

    async def on_tool_call_start(
        self,
        task: Task,
        tool_call: ToolCall,
    ) -> None:
        self._emit(
            RunToolStartEvent(
                task_name=_task_name(task),
                task_id=str(task.task_id),
                tool_call=tool_call,
            )
        )

    async def on_tool_call_end(
        self,
        task: Task,
        tool_call: ToolCall,
        result: object,
    ) -> None:
        self._emit(
            RunToolEndEvent(
                task_name=_task_name(task),
                task_id=str(task.task_id),
                tool_call=tool_call,
                result=result,
            )
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
