"""Encode AgentLane run events into process-bridge protocol events."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, cast

import structlog
from pydantic import BaseModel

from agentlane.harness import (
    RunAgentEndEvent,
    RunAgentStartEvent,
    RunEvent,
    RunEventKind,
    RunHandoffEndEvent,
    RunHandoffStartEvent,
    RunLLMEndEvent,
    RunLLMStartEvent,
    RunModelStreamEvent,
    RunPlanUpdatedEvent,
    RunResult,
    RunStateSnapshotEvent,
    RunToolApprovalEvent,
    RunToolEndEvent,
    RunToolStartEvent,
)
from agentlane.harness.tools import (
    PLAN_TOOL_NAME,
    ToolApprovalEvent,
    ToolApprovalStatus,
    ToolPermissionRequest,
)
from agentlane.models import (
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    ToolCall,
    get_content_or_none,
    get_reasoning_phase,
    get_usage_totals,
)

from ._protocol import ERROR_SCOPE_RUN, MAX_TOOL_RESULT_PREVIEW_CHARS, BridgeEventType

_logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BridgeRunEvent:
    """Encoded bridge event."""

    type: BridgeEventType
    payload: dict[str, object]


class ApprovalRequestPayload(BaseModel):
    """Typed approval request payload embedded in approval bridge events."""

    tool_name: str
    operation: str
    cwd: str
    path: str | None
    command: str | None
    skill_name: str | None
    reason: str | None
    run_id: str | None
    agent_name: str | None
    tool_call_id: str | None
    metadata: dict[str, object]

    def to_bridge_payload(self) -> dict[str, object]:
        """Return a JSON-compatible payload for the NDJSON event writer."""
        return cast(dict[str, object], self.model_dump(mode="json"))


def _bridge_events(*event_types: BridgeEventType) -> frozenset[BridgeEventType]:
    return frozenset(event_types)


class RunEventEncodingContext(Protocol):
    """Encoder state exposed to bridge event handlers."""

    @property
    def next_turn_count(self) -> int:
        """Return the turn number to attach to the next agent-start event."""
        ...

    def observe_turn_count(self, turn_count: int) -> None:
        """Record the authoritative turn count observed from a state snapshot."""
        ...


@dataclass(frozen=True, slots=True)
class RunEventBridgeHandler(ABC):
    """One extension point for mapping an AgentLane run event into bridge events."""

    kind: RunEventKind
    event_type: type[object]
    bridge_event_types: frozenset[BridgeEventType]

    @abstractmethod
    def encode(
        self,
        encoder: RunEventEncodingContext,
        event: object,
    ) -> BridgeRunEvent | None:
        """Encode one upstream run event into zero or one bridge event.

        Returning ``None`` means the event was intentionally consumed without a
        downstream protocol event, such as a completed stream marker.
        """


class _LineageEvent(Protocol):
    @property
    def task_id(self) -> str:
        """Return the stable task identifier for this run event."""
        ...

    @property
    def parent_task_id(self) -> str | None:
        """Return the parent task identifier, or ``None`` for root tasks."""
        ...

    @property
    def is_root(self) -> bool:
        """Return whether this event belongs to the root task."""
        ...


class RunEventEncoder:
    """Stateful encoder for one bridge conversation."""

    def __init__(
        self,
        *,
        handlers: tuple[RunEventBridgeHandler, ...] | None = None,
    ) -> None:
        self._handlers = RUN_EVENT_BRIDGE_HANDLERS if handlers is None else handlers
        self._latest_turn_count = 0

    def reset_turns(self) -> None:
        """Reset next-turn tracking after a conversation reset."""
        self._latest_turn_count = 0

    @property
    def next_turn_count(self) -> int:
        """Return the turn count that the next agent-start event should report."""
        return self._latest_turn_count + 1

    def observe_turn_count(self, turn_count: int) -> None:
        """Record the latest turn count reported by AgentLane state."""
        self._latest_turn_count = turn_count

    def encode(self, event: object) -> BridgeRunEvent | None:
        """Encode one AgentLane run event into a bridge event."""
        for handler in self._handlers:
            if isinstance(event, handler.event_type):
                return handler.encode(self, event)

        # `run_event` is the explicit diagnostic event for framework events
        # that do not have downstream UI semantics in this bridge yet.
        return BridgeRunEvent(
            BridgeEventType.RUN_EVENT,
            {"run_event_type": type(event).__name__},
        )


class ModelStreamRunEventHandler(RunEventBridgeHandler):
    """Encode provider/model stream events into bridge stream events."""

    def __init__(self) -> None:
        super().__init__(
            kind=RunEventKind.MODEL_STREAM,
            event_type=RunModelStreamEvent,
            bridge_event_types=_bridge_events(
                BridgeEventType.ASSISTANT_DELTA,
                BridgeEventType.REASONING_DELTA,
                BridgeEventType.TOOL_ARGUMENTS_DELTA,
                BridgeEventType.PROVIDER_EVENT,
                BridgeEventType.ERROR,
            ),
        )

    def encode(
        self,
        encoder: RunEventEncodingContext,
        event: object,
    ) -> BridgeRunEvent | None:
        del encoder
        run_event = cast(RunModelStreamEvent, event)

        return self._encode_stream_event(run_event.event)

    def _encode_stream_event(self, event: ModelStreamEvent) -> BridgeRunEvent | None:
        if event.kind == ModelStreamEventKind.TEXT_DELTA:
            # Providers may emit empty text deltas as stream bookkeeping. The
            # UI should not render a visible empty assistant update.
            if not event.text:
                return None

            return BridgeRunEvent(BridgeEventType.ASSISTANT_DELTA, {"text": event.text})

        if event.kind == ModelStreamEventKind.REASONING:
            # Some provider reasoning events only carry native metadata. Bridge
            # consumers expect reasoning_delta to include text-like content.
            if event.reasoning is None:
                return None

            return BridgeRunEvent(
                BridgeEventType.REASONING_DELTA,
                {
                    "text": str(event.reasoning),
                    "provider_event_type": event.provider_event_type,
                    "reasoning_signature": event.reasoning_signature,
                },
            )

        if event.kind == ModelStreamEventKind.TOOL_CALL_ARGUMENTS_DELTA:
            return BridgeRunEvent(
                BridgeEventType.TOOL_ARGUMENTS_DELTA,
                {
                    "tool_call_id": event.tool_call_id or "",
                    "tool_call_index": event.tool_call_index,
                    "delta": event.arguments_delta or "",
                },
            )

        if event.kind == ModelStreamEventKind.PROVIDER:
            # Preserve provider-native metadata without forcing every provider
            # event into assistant text or tool argument deltas.
            phase = get_reasoning_phase(event)
            return BridgeRunEvent(
                BridgeEventType.PROVIDER_EVENT,
                {
                    "provider_event_type": event.provider_event_type,
                    "item_index": event.item_index,
                    "item_type": event.item_type,
                    "phase": None if phase is None else phase.value,
                },
            )

        if event.kind == ModelStreamEventKind.ERROR:
            return BridgeRunEvent(
                BridgeEventType.ERROR,
                {
                    "message": "" if event.error is None else str(event.error),
                    "scope": ERROR_SCOPE_RUN,
                },
            )

        if event.kind == ModelStreamEventKind.COMPLETED:
            # The run stream itself already provides completion ordering; there
            # is no separate downstream event for provider stream completion.
            return None

        _logger.warning(
            "model_stream_event_unhandled",
            kind=str(event.kind),
            provider_event_type=event.provider_event_type,
        )
        return None


class AgentStartRunEventHandler(RunEventBridgeHandler):
    """Encode agent task start events."""

    def __init__(self) -> None:
        super().__init__(
            kind=RunEventKind.AGENT_START,
            event_type=RunAgentStartEvent,
            bridge_event_types=_bridge_events(BridgeEventType.AGENT_START),
        )

    def encode(self, encoder: RunEventEncodingContext, event: object) -> BridgeRunEvent:
        run_event = cast(RunAgentStartEvent, event)

        return BridgeRunEvent(
            BridgeEventType.AGENT_START,
            {
                **_lineage_payload(run_event),
                "agent": run_event.task_name,
                "next_turn": encoder.next_turn_count,
            },
        )


class AgentEndRunEventHandler(RunEventBridgeHandler):
    """Encode agent task completion events."""

    def __init__(self) -> None:
        super().__init__(
            kind=RunEventKind.AGENT_END,
            event_type=RunAgentEndEvent,
            bridge_event_types=_bridge_events(BridgeEventType.AGENT_END),
        )

    def encode(self, encoder: RunEventEncodingContext, event: object) -> BridgeRunEvent:
        del encoder
        run_event = cast(RunAgentEndEvent, event)

        return BridgeRunEvent(
            BridgeEventType.AGENT_END,
            {
                **_lineage_payload(run_event),
                "agent": run_event.task_name,
                "final_preview": _result_preview(run_event.result),
            },
        )


class LLMStartRunEventHandler(RunEventBridgeHandler):
    """Encode LLM request start events."""

    def __init__(self) -> None:
        super().__init__(
            kind=RunEventKind.LLM_START,
            event_type=RunLLMStartEvent,
            bridge_event_types=_bridge_events(BridgeEventType.LLM_START),
        )

    def encode(self, encoder: RunEventEncodingContext, event: object) -> BridgeRunEvent:
        del encoder
        run_event = cast(RunLLMStartEvent, event)

        return BridgeRunEvent(
            BridgeEventType.LLM_START,
            {
                **_lineage_payload(run_event),
                "agent": run_event.task_name,
                "message_count": len(run_event.messages),
            },
        )


class LLMEndRunEventHandler(RunEventBridgeHandler):
    """Encode LLM request completion events."""

    def __init__(self) -> None:
        super().__init__(
            kind=RunEventKind.LLM_END,
            event_type=RunLLMEndEvent,
            bridge_event_types=_bridge_events(BridgeEventType.LLM_END),
        )

    def encode(self, encoder: RunEventEncodingContext, event: object) -> BridgeRunEvent:
        del encoder
        run_event = cast(RunLLMEndEvent, event)

        return BridgeRunEvent(
            BridgeEventType.LLM_END,
            {
                **_lineage_payload(run_event),
                "agent": run_event.task_name,
                "output_preview": _response_preview(run_event.response),
                "usage": _usage_payload(run_event.response),
            },
        )


class ToolStartRunEventHandler(RunEventBridgeHandler):
    """Encode tool execution start events."""

    def __init__(self) -> None:
        super().__init__(
            kind=RunEventKind.TOOL_START,
            event_type=RunToolStartEvent,
            bridge_event_types=_bridge_events(BridgeEventType.TOOL_START),
        )

    def encode(self, encoder: RunEventEncodingContext, event: object) -> BridgeRunEvent:
        del encoder
        run_event = cast(RunToolStartEvent, event)
        tool_name = _tool_name(run_event.tool_call)

        return BridgeRunEvent(
            BridgeEventType.TOOL_START,
            {
                **_lineage_payload(run_event),
                "agent": run_event.task_name,
                "tool": tool_name,
                "tool_call_id": run_event.tool_call.id,
                "is_plan": tool_name == PLAN_TOOL_NAME,
                "is_delegation": run_event.is_delegation,
                "arguments": _format_tool_arguments(
                    run_event.tool_call.function.arguments,
                ),
            },
        )


class ToolEndRunEventHandler(RunEventBridgeHandler):
    """Encode tool execution completion events."""

    def __init__(self) -> None:
        super().__init__(
            kind=RunEventKind.TOOL_END,
            event_type=RunToolEndEvent,
            bridge_event_types=_bridge_events(BridgeEventType.TOOL_END),
        )

    def encode(self, encoder: RunEventEncodingContext, event: object) -> BridgeRunEvent:
        del encoder
        run_event = cast(RunToolEndEvent, event)
        tool_name = _tool_name(run_event.tool_call)

        return BridgeRunEvent(
            BridgeEventType.TOOL_END,
            {
                **_lineage_payload(run_event),
                "agent": run_event.task_name,
                "tool": tool_name,
                "tool_call_id": run_event.tool_call.id,
                "is_plan": tool_name == PLAN_TOOL_NAME,
                "is_delegation": run_event.is_delegation,
                "ok": run_event.ok,
                "result": _preview_text(
                    str(run_event.result),
                    limit=MAX_TOOL_RESULT_PREVIEW_CHARS,
                ),
                "error": (
                    None
                    if run_event.error is None
                    else {
                        "message": run_event.error.message,
                        "kind": run_event.error.kind,
                    }
                ),
            },
        )


class PlanUpdatedRunEventHandler(RunEventBridgeHandler):
    """Encode plan update tool events."""

    def __init__(self) -> None:
        super().__init__(
            kind=RunEventKind.PLAN_UPDATED,
            event_type=RunPlanUpdatedEvent,
            bridge_event_types=_bridge_events(BridgeEventType.PLAN_UPDATED),
        )

    def encode(self, encoder: RunEventEncodingContext, event: object) -> BridgeRunEvent:
        del encoder
        run_event = cast(RunPlanUpdatedEvent, event)

        return BridgeRunEvent(
            BridgeEventType.PLAN_UPDATED,
            {
                **_lineage_payload(run_event),
                "agent": run_event.task_name,
                "tool_call_id": run_event.tool_call.id,
                "explanation": run_event.explanation,
                "steps": [
                    {"status": item.status, "step": item.step}
                    for item in run_event.plan
                ],
                "title": None,
                "raw": None,
            },
        )


class ToolApprovalRunEventHandler(RunEventBridgeHandler):
    """Encode tool approval request and resolution events."""

    def __init__(self) -> None:
        super().__init__(
            kind=RunEventKind.TOOL_APPROVAL,
            event_type=RunToolApprovalEvent,
            bridge_event_types=_bridge_events(
                BridgeEventType.APPROVAL_REQUEST,
                BridgeEventType.APPROVAL_RESOLVED,
            ),
        )

    def encode(self, encoder: RunEventEncodingContext, event: object) -> BridgeRunEvent:
        del encoder
        run_event = cast(RunToolApprovalEvent, event)

        return self._encode_approval_event(run_event.event)

    def _encode_approval_event(self, event: ToolApprovalEvent) -> BridgeRunEvent:
        record = event.record
        request = _approval_request_payload(record.request).to_bridge_payload()

        if event.status == ToolApprovalStatus.PENDING:
            # Pending approvals are the only approval events that require the
            # host app to render a decision UI.
            return BridgeRunEvent(
                BridgeEventType.APPROVAL_REQUEST,
                {
                    "id": record.request_id,
                    "request": request,
                    "reason": record.approval_required_decision.reason,
                },
            )

        final_decision = record.final_decision
        # Resolution events are emitted for both allow and deny outcomes. If a
        # record has no final decision, treat it as not allowed rather than
        # implying the client granted permission.
        return BridgeRunEvent(
            BridgeEventType.APPROVAL_RESOLVED,
            {
                "id": record.request_id,
                "allowed": bool(final_decision and final_decision.allowed),
                "request": request,
                "reason": None if final_decision is None else final_decision.reason,
            },
        )


class StateSnapshotRunEventHandler(RunEventBridgeHandler):
    """Encode run-state snapshot events."""

    def __init__(self) -> None:
        super().__init__(
            kind=RunEventKind.STATE_SNAPSHOT,
            event_type=RunStateSnapshotEvent,
            bridge_event_types=_bridge_events(BridgeEventType.STATE_SNAPSHOT),
        )

    def encode(self, encoder: RunEventEncodingContext, event: object) -> BridgeRunEvent:
        run_event = cast(RunStateSnapshotEvent, event)
        snapshot = run_event.snapshot
        # Agent-start events are emitted before the state snapshot that confirms
        # the final turn count, so the encoder keeps this compact cross-event
        # state for the next turn.
        encoder.observe_turn_count(snapshot.turn_count)

        return BridgeRunEvent(
            BridgeEventType.STATE_SNAPSHOT,
            {
                "boundary": run_event.boundary.value,
                "turn_count": snapshot.turn_count,
                "history_length": snapshot.history_length,
                "response_count": snapshot.response_count,
                "shim_state": snapshot.shim_state,
            },
        )


class HandoffStartRunEventHandler(RunEventBridgeHandler):
    """Encode handoff start events."""

    def __init__(self) -> None:
        super().__init__(
            kind=RunEventKind.HANDOFF_START,
            event_type=RunHandoffStartEvent,
            bridge_event_types=_bridge_events(BridgeEventType.HANDOFF_START),
        )

    def encode(self, encoder: RunEventEncodingContext, event: object) -> BridgeRunEvent:
        del encoder
        run_event = cast(RunHandoffStartEvent, event)

        return BridgeRunEvent(
            BridgeEventType.HANDOFF_START,
            {
                **_lineage_payload(run_event),
                "agent": run_event.task_name,
                "target": run_event.target_name,
                "tool": _tool_name(run_event.tool_call),
                "tool_call_id": run_event.tool_call.id,
            },
        )


class HandoffEndRunEventHandler(RunEventBridgeHandler):
    """Encode handoff completion events."""

    def __init__(self) -> None:
        super().__init__(
            kind=RunEventKind.HANDOFF_END,
            event_type=RunHandoffEndEvent,
            bridge_event_types=_bridge_events(BridgeEventType.HANDOFF_END),
        )

    def encode(self, encoder: RunEventEncodingContext, event: object) -> BridgeRunEvent:
        del encoder
        run_event = cast(RunHandoffEndEvent, event)

        return BridgeRunEvent(
            BridgeEventType.HANDOFF_END,
            {
                **_lineage_payload(run_event),
                "agent": run_event.task_name,
                "target": run_event.target_name,
                "tool": _tool_name(run_event.tool_call),
                "tool_call_id": run_event.tool_call.id,
                "final_preview": _result_preview(run_event.result),
            },
        )


RUN_EVENT_BRIDGE_HANDLERS: tuple[RunEventBridgeHandler, ...] = (
    ModelStreamRunEventHandler(),
    AgentStartRunEventHandler(),
    AgentEndRunEventHandler(),
    LLMStartRunEventHandler(),
    LLMEndRunEventHandler(),
    ToolStartRunEventHandler(),
    ToolEndRunEventHandler(),
    PlanUpdatedRunEventHandler(),
    ToolApprovalRunEventHandler(),
    StateSnapshotRunEventHandler(),
    HandoffStartRunEventHandler(),
    HandoffEndRunEventHandler(),
)
"""Default run-event handlers that define the bridge's event coverage."""


RUN_EVENT_KIND_BRIDGE_EVENT_TYPES: dict[RunEventKind, frozenset[BridgeEventType]] = {
    handler.kind: handler.bridge_event_types for handler in RUN_EVENT_BRIDGE_HANDLERS
}
"""Map each upstream run-event kind to the downstream bridge events it can emit."""


def encode_run_event(event: RunEvent) -> BridgeRunEvent | None:
    """Encode one run event without retaining cross-event turn state."""
    return RunEventEncoder().encode(event)


def _lineage_payload(event: _LineageEvent) -> dict[str, object]:
    return {
        "task_id": event.task_id,
        "parent_task_id": event.parent_task_id,
        "is_root": event.is_root,
        "is_subagent": not event.is_root,
    }


def _tool_name(tool_call: ToolCall) -> str:
    return tool_call.function.name or "unknown"


def _format_tool_arguments(arguments: str) -> object:
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        # Tool arguments are usually JSON, but malformed or provider-native
        # argument strings are still useful to show exactly as received.
        return arguments


def _result_preview(result: RunResult | None) -> str | None:
    if result is None:
        return None
    return _preview_text(str(result.final_output))


def _response_preview(response: ModelResponse) -> str | None:
    return _preview_text(get_content_or_none(response))


def _usage_payload(response: ModelResponse) -> dict[str, int] | None:
    """Encode the response token totals, or None when the provider omits them.

    None means "no usage reported" (a provider may omit it) so downstream
    telemetry treats the value as missing data rather than a real zero.
    """
    totals = get_usage_totals(response)
    if totals is None:
        return None

    return {
        "prompt_tokens": totals.prompt_tokens,
        "completion_tokens": totals.completion_tokens,
        "total_tokens": totals.total_tokens,
    }


def _preview_text(text: str | None, *, limit: int = 500) -> str | None:
    if text is None:
        return None

    if len(text) <= limit:
        return text

    omitted = len(text) - limit
    return text[:limit].rstrip() + f"\n[truncated, +{omitted} more chars]"


def _approval_request_payload(request: ToolPermissionRequest) -> ApprovalRequestPayload:
    return ApprovalRequestPayload(
        tool_name=request.tool_name,
        operation=request.operation.value,
        cwd=str(request.cwd),
        path=None if request.path is None else str(request.path),
        command=request.command,
        skill_name=request.skill_name,
        reason=request.reason,
        run_id=request.run_id,
        agent_name=request.agent_name,
        tool_call_id=request.tool_call_id,
        metadata={str(key): item for key, item in request.metadata.items()},
    )
