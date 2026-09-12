"""Representative events for public serialization tests."""

import json
from pathlib import Path

import pytest

from agentlane.harness import (
    AgentDescriptor,
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
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.shims import PreparedTurn, Shim
from agentlane.harness.tools import (
    ToolApprovalEvent,
    ToolApprovalRecord,
    ToolApprovalStatus,
    ToolOperation,
    ToolPermissionDecision,
    ToolPermissionRequest,
)
from agentlane.models import (
    MessageDict,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    PromptSpec,
    PromptTemplate,
    ToolCall,
    Tools,
)

from ..tools_test_utils import (
    SequenceModel,
    echo_tool,
    make_assistant_response,
    make_tool_call,
)


class _RewriteSystemMessageShim(Shim):
    """Change request messages without changing the stored prompt."""

    @property
    def name(self) -> str:
        """Return the shim identifier."""
        return "rewrite-system-message"

    async def transform_messages(
        self, turn: PreparedTurn, messages: list[MessageDict]
    ) -> list[MessageDict]:
        """Replace the rendered system message in the model request only."""
        del turn
        return [
            (
                {**message, "content": "Explain the transformed request"}
                if message.get("role") == "system"
                else dict(message)
            )
            for message in messages
        ]


@pytest.fixture(name="transformed_request_model")
def fixture_transformed_request_model() -> SequenceModel:
    """Record the actual model request without calling a provider."""
    return SequenceModel([make_assistant_response("Complete")])


@pytest.fixture(name="transformed_request_agent")
def fixture_transformed_request_agent(
    transformed_request_model: SequenceModel,
) -> DefaultAgent:
    """Keep prompt-backed state while a shim transforms the rendered request."""
    return DefaultAgent(
        descriptor=AgentDescriptor(
            name="Transformed request",
            model=transformed_request_model,
            instructions=PromptSpec(
                template=PromptTemplate(system_template="Explain {{ topic }}"),
                values={"topic": "stored prompt", "unused": "retained"},
            ),
            shims=(_RewriteSystemMessageShim(),),
        )
    )


@pytest.fixture(name="model_response")
def fixture_model_response() -> ModelResponse:
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


@pytest.fixture(name="tool_call")
def fixture_tool_call() -> ToolCall:
    """Provide a correlated call with unchanged argument JSON."""
    return ToolCall.model_validate(
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "analyze", "arguments": '{"task": "Summarize"}'},
        }
    )


@pytest.fixture(name="run_result")
def fixture_run_result(model_response: ModelResponse) -> RunResult:
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


@pytest.fixture(name="run_events")
def fixture_run_events(
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


@pytest.fixture(name="serializable_agent")
def fixture_serializable_agent() -> DefaultAgent:
    """Build a real tool-using agent with prompt-backed state and no provider calls."""
    text = "Full tool result.\n" * 1000
    model = SequenceModel(
        [
            make_assistant_response(
                "Calling the tool",
                tool_calls=[
                    make_tool_call(
                        tool_id="echo-1",
                        name="echo",
                        arguments=json.dumps({"text": text}),
                    )
                ],
            ),
            make_assistant_response("Complete"),
        ]
    )
    return DefaultAgent(
        descriptor=AgentDescriptor(
            name="Example",
            model=model,
            instructions=PromptSpec(
                template=PromptTemplate(system_template="Explain {{ topic }}"),
                values={"topic": "serialization"},
            ),
            tools=Tools(tools=[echo_tool("echo")]),
        )
    )
