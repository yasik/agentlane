"""Tests for the harness run result shape."""

from typing import Any, cast

from agentlane.harness import RunResult
from agentlane.models import Choice, Message, ModelResponse


def _response(reasoning: str | None) -> ModelResponse:
    """Build one assistant response, optionally carrying reasoning content."""
    response = ModelResponse(
        id="resp_test",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=Message(role="assistant", content="answer"),
            )
        ],
        created=0,
        model="test",
        object="chat.completion",
    )
    if reasoning is not None:
        cast(Any, response).reasoning_content = reasoning
    return response


def _result(responses: list[ModelResponse]) -> RunResult:
    """Build one run result around the given responses."""
    return RunResult(
        final_output="answer",
        responses=responses,
        turn_count=len(responses),
    )


def test_reasoning_returns_last_response_carrying_reasoning() -> None:
    """The final turn may carry no reasoning, so an earlier turn's is returned."""
    result = _result([_response("thinking hard"), _response(None)])

    reasoning = result.reasoning

    assert reasoning is not None
    assert str(reasoning) == "thinking hard"


def test_reasoning_prefers_most_recent_reasoning_turn() -> None:
    """When several turns carry reasoning, the most recent one wins."""
    result = _result([_response("earlier"), _response("later"), _response(None)])

    reasoning = result.reasoning

    assert reasoning is not None
    assert str(reasoning) == "later"


def test_reasoning_returns_none_when_no_response_carries_reasoning() -> None:
    """A run whose responses carry no reasoning reports no reasoning."""
    result = _result([_response(None), _response(None)])

    assert result.reasoning is None


def test_reasoning_returns_none_for_empty_responses() -> None:
    """A run with no responses reports no reasoning."""
    result = _result([])

    assert result.reasoning is None
