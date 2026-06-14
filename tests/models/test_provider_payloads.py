"""Tests for typed provider-native payload accessors."""

from typing import Literal

from openai.types.responses.response_output_item_done_event import (
    ResponseOutputItemDoneEvent,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem

from agentlane.models import (
    Choice,
    Message,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    ReasoningPhase,
    Usage,
    get_reasoning_phase,
    get_usage_totals,
)


def _output_item_event(
    phase: Literal["commentary", "final_answer"] | None,
) -> ModelStreamEvent:
    """Build a stream event whose raw payload carries an output message phase."""
    message = ResponseOutputMessage(
        id="msg_1",
        content=[],
        role="assistant",
        status="completed",
        type="message",
        phase=phase,
    )
    raw = ResponseOutputItemDoneEvent(
        item=message,
        output_index=0,
        sequence_number=1,
        type="response.output_item.done",
    )
    return ModelStreamEvent(kind=ModelStreamEventKind.PROVIDER, raw=raw)


def _response_with_usage(usage: Usage | None) -> ModelResponse:
    """Build a minimal ModelResponse carrying the given usage payload."""
    return ModelResponse(
        id="resp_test",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=Message(role="assistant", content="done"),
            )
        ],
        created=0,
        model="test",
        object="chat.completion",
        usage=usage,
    )


def test_get_reasoning_phase_returns_final_answer_phase() -> None:
    event = _output_item_event("final_answer")

    assert get_reasoning_phase(event) is ReasoningPhase.FINAL_ANSWER


def test_get_reasoning_phase_returns_commentary_phase() -> None:
    event = _output_item_event("commentary")

    assert get_reasoning_phase(event) is ReasoningPhase.COMMENTARY


def test_get_reasoning_phase_returns_none_when_phase_absent() -> None:
    event = _output_item_event(None)

    assert get_reasoning_phase(event) is None


def test_get_reasoning_phase_returns_none_for_non_output_item_raw() -> None:
    event = ModelStreamEvent(
        kind=ModelStreamEventKind.TEXT_DELTA,
        raw=None,
        text="hello",
    )

    assert get_reasoning_phase(event) is None


def test_get_reasoning_phase_returns_none_for_non_message_item() -> None:
    reasoning_item = ResponseReasoningItem(
        id="rs_1",
        summary=[],
        type="reasoning",
    )
    raw = ResponseOutputItemDoneEvent(
        item=reasoning_item,
        output_index=0,
        sequence_number=1,
        type="response.output_item.done",
    )
    event = ModelStreamEvent(kind=ModelStreamEventKind.PROVIDER, raw=raw)

    assert get_reasoning_phase(event) is None


def test_get_reasoning_phase_returns_none_for_unrecognized_phase() -> None:
    # Provider payloads preserved on `raw` are not re-validated, so a provider
    # may surface a phase string this enum does not model. `model_construct`
    # bypasses validation to mirror that preserved-payload shape.
    message = ResponseOutputMessage.model_construct(
        id="msg_1",
        content=[],
        role="assistant",
        status="completed",
        type="message",
        phase="some_future_phase",
    )
    raw = ResponseOutputItemDoneEvent(
        item=message,
        output_index=0,
        sequence_number=1,
        type="response.output_item.done",
    )
    event = ModelStreamEvent(kind=ModelStreamEventKind.PROVIDER, raw=raw)

    assert get_reasoning_phase(event) is None


def test_get_usage_totals_exposes_typed_token_counts() -> None:
    response = _response_with_usage(
        Usage(completion_tokens=10, prompt_tokens=20, total_tokens=30)
    )

    totals = get_usage_totals(response)

    assert totals is not None
    assert totals.prompt_tokens == 20
    assert totals.completion_tokens == 10
    assert totals.total_tokens == 30


def test_get_usage_totals_returns_none_when_usage_missing() -> None:
    response = _response_with_usage(None)

    assert get_usage_totals(response) is None


def test_get_usage_totals_returns_none_for_missing_response() -> None:
    assert get_usage_totals(None) is None
