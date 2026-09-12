"""Representative events for public serialization tests."""

import json
from dataclasses import replace
from pathlib import Path

from agentlane.harness import (
    RunAgentEndEvent,
    RunAgentStartEvent,
    RunEvent,
    RunHandoffEndEvent,
    RunHandoffStartEvent,
    RunLLMEndEvent,
    RunLLMStartEvent,
    RunModelStreamEvent,
    RunPlanItem,
    RunPlanUpdatedEvent,
    RunResult,
    RunState,
    RunStateSnapshot,
    RunStateSnapshotBoundary,
    RunStateSnapshotEvent,
    RunToolApprovalEvent,
    RunToolEndEvent,
    RunToolStartEvent,
)
from agentlane.harness.tools import (
    ToolApprovalEvent,
    ToolApprovalRecord,
    ToolApprovalStatus,
    ToolOperation,
    ToolPermissionDecision,
    ToolPermissionRequest,
)
from agentlane.models import (
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    PromptSpec,
    PromptTemplate,
    ToolCall,
)


def make_model_response() -> ModelResponse:
    """Provide an SDK response with extension fields at multiple nesting levels."""
    return ModelResponse.model_validate(
        {
            "id": "response-1",
            "object": "chat.completion",
            "created": 1,
            "model": "scripted",
            "provider_extra": {"kept": None},
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Answer",
                        "phase": "final_answer",
                    },
                }
            ],
        }
    )


def make_tool_call() -> ToolCall:
    """Provide a correlated call with unchanged argument JSON."""
    return ToolCall.model_validate(
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "analyze", "arguments": '{"task": "Summarize"}'},
        }
    )


def make_run_result(model_response: ModelResponse) -> RunResult:
    """Include real prompt-backed state and mapping-like shim state."""
    state = RunState(
        instructions=PromptSpec(
            template=PromptTemplate(system_template="Explain {{ topic }}"),
            values={"topic": "sample", "unused": "must survive"},
        ),
        history=[
            PromptSpec(
                template=PromptTemplate(user_template="Question: {{ task }}"),
                values={"task": "summary"},
            ),
            model_response,
        ],
        responses=[model_response],
        turn_count=1,
        revision=2,
    )
    state.shim_state["progress"] = {"items": list(range(1000)), "note": "x" * 6000}
    return RunResult(
        final_output={"answer": "Answer"},
        responses=[model_response],
        turn_count=1,
        run_state=state,
    )


def make_run_events(
    model_response: ModelResponse, tool_call: ToolCall, run_result: RunResult
) -> list[RunEvent]:
    """Exercise every public run-event class with nested framework values."""
    return [
        RunAgentStartEvent(task_name="qa", task_id="root"),
        RunLLMStartEvent(
            task_name="qa", task_id="root", messages=[{"role": "user", "content": "Hi"}]
        ),
        RunModelStreamEvent(
            event=ModelStreamEvent(
                kind=ModelStreamEventKind.TEXT_DELTA, text="Preamble"
            )
        ),
        RunLLMEndEvent(task_name="qa", task_id="root", response=model_response),
        RunToolStartEvent(
            task_name="qa", task_id="root", tool_call=tool_call, is_delegation=True
        ),
        RunToolEndEvent(
            task_name="qa",
            task_id="root",
            tool_call=tool_call,
            result={"sections": list(range(1000)), "text": "x" * 6000},
            is_delegation=True,
        ),
        RunToolApprovalEvent(
            event=ToolApprovalEvent(
                record=ToolApprovalRecord(
                    request_id="approval-1",
                    request=ToolPermissionRequest(
                        tool_name="read",
                        operation=ToolOperation.READ_FILE,
                        cwd=Path("/workspace"),
                        path=Path("/workspace/report.json"),
                        tool_call_id="call-1",
                        metadata={"details": list(range(1000))},
                    ),
                    approval_required_decision=ToolPermissionDecision.require_approval(),
                    status=ToolApprovalStatus.PENDING,
                )
            )
        ),
        RunHandoffStartEvent(
            task_name="qa", task_id="root", tool_call=tool_call, target_name="analyst"
        ),
        RunHandoffEndEvent(
            task_name="qa",
            task_id="root",
            tool_call=tool_call,
            target_name="analyst",
            result=run_result,
        ),
        RunStateSnapshotEvent(
            boundary=RunStateSnapshotBoundary.TOOL_ROUND_END,
            snapshot=RunStateSnapshot(
                turn_count=1,
                history_length=2,
                response_count=1,
                shim_state={"progress": [None, 1, True]},
            ),
        ),
        RunPlanUpdatedEvent(
            task_name="qa",
            task_id="root",
            tool_call=tool_call,
            plan=(RunPlanItem(step="Analyze", status="completed"),),
            explanation="Done",
        ),
        RunAgentEndEvent(task_name="qa", task_id="root", result=run_result),
    ]


def native_events() -> list[RunEvent]:
    """Build real source records for cross-language contract verification."""
    response = make_model_response()
    events = make_run_events(response, make_tool_call(), make_run_result(response))
    for kind in ModelStreamEventKind:
        events.append(
            RunModelStreamEvent(
                event=ModelStreamEvent(
                    kind=kind,
                    raw={"sdk_extra": {"unicode": "你好\n", "null": None}},
                    provider_event_type="response.output_item.done",
                    item_index=2,
                    item_type="reasoning",
                    text="",
                    tool_call_id="call-1",
                    tool_call_index=0,
                    arguments_delta="",
                    reasoning={"summary": ["你好\n"], "encrypted_content": "opaque"},
                    reasoning_signature="signed",
                    response=response,
                    error=(
                        RuntimeError("model failed")
                        if kind == ModelStreamEventKind.ERROR
                        else None
                    ),
                )
            )
        )
    approval = next(
        event for event in events if isinstance(event, RunToolApprovalEvent)
    )
    events.append(
        replace(
            approval,
            event=ToolApprovalEvent(
                record=replace(
                    approval.event.record,
                    status=ToolApprovalStatus.RESOLVED,
                    final_decision=ToolPermissionDecision.allow(),
                )
            ),
        )
    )
    events.append(
        RunAgentStartEvent(
            task_name="child", task_id="child", parent_task_id="root", is_root=False
        )
    )
    return events


def write_native_fixtures() -> None:
    """Replace generated native fixtures and keep the authored control examples."""
    path = Path(__file__).parents[1] / "fixtures" / "protocol" / "events.json"
    controls = [
        event for event in json.loads(path.read_text()) if event["type"] != "run_event"
    ]
    records = [
        {
            "protocol_version": "1.0",
            "type": "run_event",
            "ts": 100 + index,
            "event": event.to_dict(),
        }
        for index, event in enumerate(native_events())
    ]
    path.write_text(
        json.dumps([*controls, *records], ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    write_native_fixtures()
