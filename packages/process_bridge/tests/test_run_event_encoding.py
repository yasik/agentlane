from pathlib import Path

from agentlane_process_bridge import (
    BridgeEventType,
    BridgeRunEvent,
    RunEventBridgeHandler,
    RunEventEncoder,
    RunEventEncodingContext,
)

from agentlane.harness import (
    RunAgentEndEvent,
    RunAgentStartEvent,
    RunEventKind,
    RunHandoffEndEvent,
    RunHandoffStartEvent,
    RunLLMEndEvent,
    RunLLMStartEvent,
    RunModelStreamEvent,
    RunPlanItem,
    RunPlanUpdatedEvent,
    RunResult,
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
    Choice,
    Message,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    ToolCall,
    ToolError,
    Usage,
)


def test_encoder_maps_model_stream_events() -> None:
    encoder = RunEventEncoder()

    text = encoder.encode(
        RunModelStreamEvent(
            event=ModelStreamEvent(kind=ModelStreamEventKind.TEXT_DELTA, text="hi")
        )
    )
    provider = encoder.encode(
        RunModelStreamEvent(
            event=ModelStreamEvent(
                kind=ModelStreamEventKind.PROVIDER,
                provider_event_type="response.output_item.added",
                item_index=1,
                item_type="reasoning",
            )
        )
    )
    reasoning = encoder.encode(
        RunModelStreamEvent(
            event=ModelStreamEvent(
                kind=ModelStreamEventKind.REASONING,
                reasoning="thinking",
                reasoning_signature="sig",
            )
        )
    )
    arguments = encoder.encode(
        RunModelStreamEvent(
            event=ModelStreamEvent(
                kind=ModelStreamEventKind.TOOL_CALL_ARGUMENTS_DELTA,
                tool_call_id="call_1",
                tool_call_index=0,
                arguments_delta='{"path"',
            )
        )
    )
    error = encoder.encode(
        RunModelStreamEvent(
            event=ModelStreamEvent(
                kind=ModelStreamEventKind.ERROR,
                error=RuntimeError("model failed"),
            )
        )
    )
    completed = encoder.encode(
        RunModelStreamEvent(event=ModelStreamEvent(kind=ModelStreamEventKind.COMPLETED))
    )

    assert text is not None and text.type == BridgeEventType.ASSISTANT_DELTA
    assert text.payload["text"] == "hi"
    assert provider is not None and provider.type == BridgeEventType.PROVIDER_EVENT
    assert provider.payload["item_type"] == "reasoning"
    assert reasoning is not None and reasoning.type == BridgeEventType.REASONING_DELTA
    assert reasoning.payload["reasoning_signature"] == "sig"
    assert (
        arguments is not None and arguments.type == BridgeEventType.TOOL_ARGUMENTS_DELTA
    )
    assert arguments.payload["tool_call_id"] == "call_1"
    assert error is not None and error.type == BridgeEventType.ERROR
    assert error.payload["scope"] == "run"
    assert completed is None


def test_encoder_accepts_explicit_handler_registry() -> None:
    class CustomAgentStartHandler(RunEventBridgeHandler):
        def __init__(self) -> None:
            super().__init__(
                kind=RunEventKind.AGENT_START,
                event_type=RunAgentStartEvent,
                bridge_event_types=frozenset({BridgeEventType.RUN_EVENT}),
            )

        def encode(
            self,
            encoder: RunEventEncodingContext,
            event: object,
        ) -> BridgeRunEvent:
            del encoder, event

            return BridgeRunEvent(
                BridgeEventType.RUN_EVENT,
                {"run_event_type": "custom_agent_start"},
            )

    handler = CustomAgentStartHandler()
    encoder = RunEventEncoder(handlers=(handler,))

    encoded = encoder.encode(RunAgentStartEvent(task_name="Root", task_id="task-root"))

    assert encoded is not None
    assert encoded.type == BridgeEventType.RUN_EVENT
    assert encoded.payload == {"run_event_type": "custom_agent_start"}


def test_encoder_preserves_lineage_and_tool_error_fields() -> None:
    encoder = RunEventEncoder()
    call = _tool_call(arguments='{"path":"README.md"}')

    agent_start = encoder.encode(
        RunAgentStartEvent(
            task_name="Root",
            task_id="task-root",
        )
    )
    tool_start = encoder.encode(
        RunToolStartEvent(
            task_name="Child",
            task_id="task-child",
            parent_task_id="task-root",
            is_root=False,
            tool_call=call,
            is_delegation=True,
        )
    )
    tool_end = encoder.encode(
        RunToolEndEvent(
            task_name="Child",
            task_id="task-child",
            parent_task_id="task-root",
            is_root=False,
            tool_call=call,
            result="failed",
            ok=False,
            error=ToolError(message="boom", kind="timeout"),
            is_delegation=True,
        )
    )

    assert agent_start is not None and agent_start.payload["next_turn"] == 1
    assert tool_start is not None and tool_start.type == BridgeEventType.TOOL_START
    assert tool_start.payload["parent_task_id"] == "task-root"
    assert tool_start.payload["is_subagent"] is True
    assert tool_start.payload["arguments"] == {"path": "README.md"}
    assert tool_end is not None and tool_end.type == BridgeEventType.TOOL_END
    assert tool_end.payload["ok"] is False
    assert tool_end.payload["error"] == {"message": "boom", "kind": "timeout"}


def test_encoder_maps_llm_and_handoff_lifecycle_events() -> None:
    encoder = RunEventEncoder()
    call = _tool_call(arguments="{}")

    llm_start = encoder.encode(
        RunLLMStartEvent(
            task_name="Root",
            task_id="task-root",
            messages=[{"role": "user", "content": "hello"}],
        )
    )
    llm_end = encoder.encode(
        RunLLMEndEvent(
            task_name="Root",
            task_id="task-root",
            response=_model_response("done"),
        )
    )
    handoff_start = encoder.encode(
        RunHandoffStartEvent(
            task_name="Root",
            task_id="task-root",
            tool_call=call,
            target_name="Child",
        )
    )
    handoff_end = encoder.encode(
        RunHandoffEndEvent(
            task_name="Root",
            task_id="task-root",
            tool_call=call,
            target_name="Child",
            result=RunResult(final_output="child done", responses=[], turn_count=1),
        )
    )

    assert llm_start is not None and llm_start.type == BridgeEventType.LLM_START
    assert llm_start.payload["message_count"] == 1
    assert llm_end is not None and llm_end.type == BridgeEventType.LLM_END
    assert llm_end.payload["output_preview"] == "done"
    # A response without provider usage encodes usage as None (missing data),
    # never a synthesized zero.
    assert llm_end.payload["usage"] is None
    assert (
        handoff_start is not None
        and handoff_start.type == BridgeEventType.HANDOFF_START
    )
    assert handoff_start.payload["target"] == "Child"
    assert handoff_end is not None and handoff_end.type == BridgeEventType.HANDOFF_END
    assert handoff_end.payload["final_preview"] == "child done"


def test_encoder_maps_plan_approval_snapshot_and_fallback_events() -> None:
    encoder = RunEventEncoder()
    call = _tool_call(arguments="{}")
    request = ToolPermissionRequest(
        tool_name="write",
        operation=ToolOperation.CREATE_FILE,
        cwd=Path("/workspace"),
        path=Path("/workspace/a.txt"),
        reason="review write",
    )
    record = ToolApprovalRecord(
        request_id="approval-1",
        request=request,
        approval_required_decision=ToolPermissionDecision.require_approval(
            "review write"
        ),
        status=ToolApprovalStatus.PENDING,
    )
    resolved_record = ToolApprovalRecord(
        request_id="approval-1",
        request=request,
        approval_required_decision=ToolPermissionDecision.require_approval(
            "review write"
        ),
        status=ToolApprovalStatus.RESOLVED,
        final_decision=ToolPermissionDecision.allow(),
    )

    plan = encoder.encode(
        RunPlanUpdatedEvent(
            task_name="Root",
            task_id="task-root",
            tool_call=call,
            plan=(RunPlanItem(step="ship", status="in_progress"),),
            explanation="because",
        )
    )
    approval = encoder.encode(RunToolApprovalEvent(event=ToolApprovalEvent(record)))
    approval_resolved = encoder.encode(
        RunToolApprovalEvent(event=ToolApprovalEvent(resolved_record))
    )
    snapshot = encoder.encode(
        RunStateSnapshotEvent(
            boundary=RunStateSnapshotBoundary.TURN_PREPARED,
            snapshot=RunStateSnapshot(
                turn_count=2,
                history_length=3,
                response_count=1,
                shim_state={"skill": ["x"]},
            ),
        )
    )
    agent_end = encoder.encode(
        RunAgentEndEvent(
            task_name="Root",
            task_id="task-root",
            result=RunResult(final_output="done", responses=[], turn_count=2),
        )
    )
    unknown = encoder.encode(object())

    assert plan is not None and plan.type == BridgeEventType.PLAN_UPDATED
    assert plan.payload["steps"] == [{"status": "in_progress", "step": "ship"}]
    assert approval is not None and approval.type == BridgeEventType.APPROVAL_REQUEST
    assert approval.payload["request"] == {
        "tool_name": "write",
        "operation": "create_file",
        "cwd": "/workspace",
        "path": "/workspace/a.txt",
        "command": None,
        "skill_name": None,
        "reason": "review write",
        "run_id": None,
        "agent_name": None,
        "tool_call_id": None,
        "metadata": {},
    }
    assert (
        approval_resolved is not None
        and approval_resolved.type == BridgeEventType.APPROVAL_RESOLVED
    )
    assert approval_resolved.payload["allowed"] is True
    assert snapshot is not None and snapshot.type == BridgeEventType.STATE_SNAPSHOT
    assert snapshot.payload["turn_count"] == 2
    assert agent_end is not None and agent_end.payload["final_preview"] == "done"
    assert unknown is not None and unknown.type == BridgeEventType.RUN_EVENT
    assert unknown.payload["run_event_type"] == "object"


def _tool_call(*, arguments: str) -> ToolCall:
    return ToolCall.model_validate(
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "write_plan",
                "arguments": arguments,
            },
        }
    )


def test_llm_end_encodes_provider_token_usage() -> None:
    encoder = RunEventEncoder()

    llm_end = encoder.encode(
        RunLLMEndEvent(
            task_name="Root",
            task_id="task-root",
            response=_model_response(
                "done",
                usage=Usage(
                    prompt_tokens=1200,
                    completion_tokens=340,
                    total_tokens=1540,
                ),
            ),
        )
    )

    assert llm_end is not None and llm_end.type == BridgeEventType.LLM_END
    assert llm_end.payload["usage"] == {
        "prompt_tokens": 1200,
        "completion_tokens": 340,
        "total_tokens": 1540,
    }


def _model_response(content: str, *, usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(
        id="resp_test",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=Message(role="assistant", content=content),
            )
        ],
        created=0,
        model="test",
        object="chat.completion",
        usage=usage,
    )
