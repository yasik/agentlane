"""Source-to-JSON fidelity checks for the public AgentLane contracts."""

import json
from dataclasses import fields, replace
from typing import Any, cast, get_args
from uuid import UUID

import pytest
from openai.types.responses import ResponseOutputItemAddedEvent, ResponseOutputMessage
from pydantic import BaseModel, RootModel

from agentlane.harness import (
    RunAgentEndEvent,
    RunEvent,
    RunEventKind,
    RunModelStreamEvent,
    RunResult,
    RunToolApprovalEvent,
    RunToolEndEvent,
)
from agentlane.harness.tools import ToolApprovalStatus, ToolPermissionDecision
from agentlane.models import (
    ImagePart,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    MultiPartPromptTemplate,
    OutputSchema,
    PromptSpec,
    PromptTemplate,
    PromptTemplateBase,
    TextPart,
    ToolCall,
    ToolError,
)


def test_serialize_run_event_all_public_classes_preserve_fields(
    run_events: list[RunEvent],
) -> None:
    """Upstream union additions require a representative fixture and coverage review."""
    assert {type(event) for event in run_events} == set(get_args(RunEvent.__value__))
    assert {event.kind for event in run_events} == set(RunEventKind)

    for event in run_events:
        record = event.to_dict()
        event_type, payload = record["type"], record["payload"]
        assert event_type == event.kind.value
        assert json.loads(json.dumps(payload, allow_nan=False)) == payload
        assert payload.keys() == {field.name for field in fields(event)}
        assert payload["kind"] == event.kind.value

        if isinstance(event, (RunModelStreamEvent, RunToolApprovalEvent)):
            contents = cast(dict[str, Any], payload["event"])
            assert contents.keys() == {field.name for field in fields(event.event)}


@pytest.mark.parametrize("kind", list(ModelStreamEventKind))
def test_serialize_model_event_preserves_complete_payload(
    kind: ModelStreamEventKind,
    model_response: ModelResponse,
) -> None:
    """Deltas retain correlation, raw data, reasoning signatures, and complete responses."""
    source = ModelStreamEvent(
        kind=kind,
        raw={"phase": "commentary", "nested": [None, True, "x" * 6000]},
        provider_event_type="response.output_item.done",
        item_index=2,
        item_type="message",
        text="  Preamble\n",
        tool_call_id="call-1",
        tool_call_index=0,
        arguments_delta='{"task":',
        reasoning={"summary": "Check", "encrypted_content": "opaque"},
        reasoning_signature="signed",
        response=model_response,
        error=(
            ValueError("source error") if kind == ModelStreamEventKind.ERROR else None
        ),
    )
    event = RunModelStreamEvent(event=source)
    record = event.to_dict()
    event_type, payload = record["type"], record["payload"]

    assert json.loads(event.dumps()) == record
    assert event_type == RunEventKind.MODEL_STREAM.value
    assert payload.keys() == {"kind", "event"}
    assert payload["kind"] == RunEventKind.MODEL_STREAM.value
    contents = cast(dict[str, Any], payload["event"])
    assert contents.keys() == {field.name for field in fields(source)}
    assert contents["raw"] == source.raw
    assert contents["reasoning"] == source.reasoning
    assert contents["response"] == model_response.model_dump(mode="json")
    for name in (
        "provider_event_type",
        "item_index",
        "item_type",
        "text",
        "tool_call_id",
        "tool_call_index",
        "arguments_delta",
        "reasoning_signature",
    ):
        assert contents[name] == getattr(source, name)

    assert contents["kind"] == kind.value
    assert contents["error"] == (
        {"type": "ValueError", "message": "source error"}
        if kind == ModelStreamEventKind.ERROR
        else None
    )


@pytest.mark.parametrize("status", list(ToolApprovalStatus))
def test_serialize_approval_preserves_record(
    run_events: list[RunEvent], status: ToolApprovalStatus
) -> None:
    """Both approval statuses retain the wrapper, full requests, and decisions."""
    source = next(
        event for event in run_events if isinstance(event, RunToolApprovalEvent)
    )
    approval_record = replace(
        source.event.record,
        status=status,
        final_decision=(
            ToolPermissionDecision.allow()
            if status == ToolApprovalStatus.RESOLVED
            else None
        ),
    )
    event = replace(source, event=replace(source.event, record=approval_record))
    record = event.to_dict()
    event_type, payload = record["type"], record["payload"]

    assert json.loads(event.dumps()) == record
    assert event_type == RunEventKind.TOOL_APPROVAL.value
    assert payload.keys() == {"kind", "event"}
    assert payload["kind"] == RunEventKind.TOOL_APPROVAL.value
    contents = cast(dict[str, Any], payload["event"])
    assert contents.keys() == {"record"}
    record_payload = contents["record"]
    assert record_payload.keys() == {field.name for field in fields(approval_record)}
    assert record_payload["request"]["path"] == "/workspace/report.json"
    assert record_payload["request"]["metadata"]["details"] == list(range(1000))
    assert record_payload["status"] == status.value
    assert record_payload["final_decision"] == (
        {"outcome": "allow", "reason": None}
        if status == ToolApprovalStatus.RESOLVED
        else None
    )


def test_serialize_result_preserves_prompts_history_and_shim_state(
    run_result: RunResult,
) -> None:
    """Executable prompts use the documented rendered view without flattening history."""
    record = RunAgentEndEvent(
        task_name="qa", task_id="root", result=run_result
    ).to_dict()
    payload = record["payload"]
    state = cast(dict[str, Any], payload["result"])["run_state"]

    assert state["instructions"] == {
        "messages": [{"role": "system", "content": "Explain sample"}],
        "values": {"topic": "sample", "unused": "must survive"},
        "response_format": None,
    }
    assert state["history"][0] == {
        "messages": [{"role": "user", "content": "Question: summary"}],
        "values": {"task": "summary"},
        "response_format": None,
    }
    assert state["history"][1] == run_result.responses[0].model_dump(mode="json")
    assert state["shim_state"]["progress"] == {
        "items": list(range(1000)),
        "note": "x" * 6000,
    }
    assert state["revision"] == 2
    assert run_result.run_state is not None
    assert isinstance(run_result.run_state.instructions, PromptSpec)


class _Answer(BaseModel):
    """Small structured prompt output for schema fidelity tests."""

    answer: str
    """Synthetic answer text."""


def test_serialize_multipart_prompt_preserves_content_and_schema(
    tool_call: ToolCall,
) -> None:
    """The public template interface retains multimodal messages and structured output."""
    template = MultiPartPromptTemplate[dict[str, str], _Answer](
        system_parts=[TextPart(template="Explain {{ topic }}")],
        user_parts=[ImagePart(base64_data="YWJj", media_type="image/png")],
        output_schema=OutputSchema(_Answer),
    )
    prompt = PromptSpec(template=template, values={"topic": "sample"})
    record = RunToolEndEvent(
        task_name="qa", task_id="root", tool_call=tool_call, result=prompt
    ).to_dict()
    payload = record["payload"]

    assert payload["result"] == {
        "messages": template.render_messages(prompt.values),
        "values": prompt.values,
        "response_format": template.response_format(),
    }


@pytest.mark.parametrize(
    "value",
    [
        object(),
        b"bytes",
        {1: "key"},
        {1, 2},
        float("nan"),
        float("inf"),
        PromptTemplate[dict[str, str], str](system_template="Hello"),
    ],
)
def test_serialize_unsupported_value_fails_explicitly(
    value: object, tool_call: ToolCall
) -> None:
    """Unsupported data never becomes a preview, empty substitute, or invalid JSON."""
    event = RunToolEndEvent(
        task_name="qa", task_id="root", tool_call=tool_call, result=value
    )
    for convert in (event.to_dict, event.dumps):
        with pytest.raises((TypeError, ValueError)):
            convert()


def test_serialize_cycle_fails_but_shared_value_is_preserved(
    tool_call: ToolCall,
) -> None:
    """Only ancestor cycles fail; repeated references are legitimate JSON values."""
    shared = {"values": [1, None]}
    event = RunToolEndEvent(
        task_name="qa", task_id="root", tool_call=tool_call, result=[shared, shared]
    )
    assert event.to_dict()["payload"]["result"] == [shared, shared]

    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="Cyclic"):
        replace(event, result=cycle).to_dict()


def test_serialize_child_tool_failure_retains_lineage_and_error(
    tool_call: ToolCall,
) -> None:
    """Shared-hook child events remain visible and retain structured failure metadata."""
    event = RunToolEndEvent(
        task_name="child",
        task_id="child-id",
        parent_task_id="root",
        is_root=False,
        tool_call=tool_call,
        result={"retryable": True},
        ok=False,
        error=ToolError(message="Unavailable", kind="timeout"),
    )
    record = event.to_dict()
    event_type, payload = record["type"], record["payload"]

    assert event_type == RunEventKind.TOOL_END.value
    assert payload["parent_task_id"] == "root"
    assert payload["is_root"] is False
    assert payload["error"] == {"message": "Unavailable", "kind": "timeout"}
    assert payload["ok"] is False


def test_serialize_identifier_uses_string(tool_call: ToolCall) -> None:
    """Opaque UUID values retain their full string identity."""
    identity = UUID("00000000-0000-4000-8000-000000000001")
    event = RunToolEndEvent(
        task_name="qa", task_id="root", tool_call=tool_call, result=identity
    )
    assert event.to_dict()["payload"]["result"] == str(identity)


def test_serialize_provider_sdk_payload_retains_extras_and_nulls() -> None:
    """A real provider event retains nested phase and future SDK extension fields."""
    raw = ResponseOutputItemAddedEvent(
        type="response.output_item.added",
        sequence_number=1,
        output_index=0,
        item=ResponseOutputMessage(
            id="message-1",
            type="message",
            role="assistant",
            content=[],
            status="in_progress",
            phase="commentary",
        ),
    ).model_copy(update={"extension": {"future": None}})
    record = RunModelStreamEvent(
        event=ModelStreamEvent(kind=ModelStreamEventKind.PROVIDER, raw=raw)
    ).to_dict()
    payload = record["payload"]

    assert cast(dict[str, Any], payload["event"])["raw"] == raw.model_dump(mode="json")


class _CustomTemplate(PromptTemplateBase[dict[str, str], str]):
    """Non-Jinja template proving the transport uses only the public interface."""

    def render_messages(
        self, ctx: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        """Return the full rendered message sequence, including an empty context."""
        return [{"role": "user", "content": ctx["task"] if ctx else "Default task"}]

    def response_format(self) -> dict[str, Any] | None:
        """Use unstructured output."""
        return None


@pytest.mark.parametrize("values", [None, {"task": "Analyze", "unused": "retained"}])
def test_serialize_custom_prompt_uses_public_rendering(
    values: dict[str, str] | None, tool_call: ToolCall
) -> None:
    """Custom prompt implementations need no codec registration or private inspection."""
    prompt = PromptSpec(template=_CustomTemplate(), values=values)
    record = RunToolEndEvent(
        task_name="qa",
        task_id="root",
        tool_call=tool_call,
        result=RootModel[Any](prompt),
    ).to_dict()
    payload = record["payload"]

    assert payload["result"] == {
        "messages": prompt.template.render_messages(values),
        "values": values,
        "response_format": None,
    }


def test_serialize_prompt_invalid_values_fail(tool_call: ToolCall) -> None:
    """Even unused prompt values cannot be silently omitted from the rendered view."""
    prompt = PromptSpec(
        template=PromptTemplate(system_template="Hello"), values={"unused": object()}
    )
    with pytest.raises(TypeError, match="Unsupported"):

        RunToolEndEvent(
            task_name="qa", task_id="root", tool_call=tool_call, result=prompt
        ).to_dict()


def test_serialize_pydantic_nested_values_preserve_subclass_fields(
    tool_call: ToolCall,
) -> None:
    """Recursive field iteration retains subclass fields without SDK serializer warmup."""
    source = RootModel[BaseModel](_Answer(answer="Full value"))
    record = RunToolEndEvent(
        task_name="qa", task_id="root", tool_call=tool_call, result=source
    ).to_dict()
    payload = record["payload"]
    assert payload["result"] == {"answer": "Full value"}


def test_run_event_dumps_all_public_classes_match_records(
    run_events: list[RunEvent],
) -> None:
    """Both methods use one conversion, without adding framing or instance storage."""
    for event in run_events:
        record = event.to_dict()
        encoded = event.dumps()
        assert set(record) == {"type", "payload"}
        assert type(record["type"]) is str
        assert json.loads(encoded) == record
        assert not encoded.endswith("\n")
        assert not hasattr(event, "__dict__")


def test_run_event_dumps_unicode_and_detached_payload_preserve_source(
    tool_call: ToolCall,
) -> None:
    """JSON preserves text and newlines; callers can change a record without changing the event."""
    result = {"text": "\u03b1\n", "items": [None, True, {"value": 1}]}
    event = RunToolEndEvent(
        task_name="qa", task_id="root", tool_call=tool_call, result=result
    )
    expected = event.to_dict()
    record = event.to_dict()
    cast(dict[str, Any], record["payload"]["result"])["items"].clear()

    assert result["items"] == [None, True, {"value": 1}]
    assert json.loads(event.dumps()) == expected
    assert "\u03b1" in event.dumps()
