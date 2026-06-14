import copy
import pickle

from pydantic import BaseModel

from agentlane.harness import (
    PlanUpdateResult,
    ToolError,
    ToolFailure,
    ToolOutcome,
    tool_outcome,
)
from agentlane.harness._tool_result import as_plan_update
from agentlane.harness.tools import PLAN_UPDATED_MESSAGE, PlanUpdate
from agentlane.models import Tool, ToolExecutionContext
from agentlane.runtime import CancellationToken


class _Args(BaseModel):
    pass


def test_tool_failure_is_str_equal_to_its_text() -> None:
    failure = ToolFailure(text="boom", error=ToolError(message="boom", kind="x"))

    assert isinstance(failure, str)
    assert failure == "boom"
    assert failure.text == "boom"
    assert failure.error == ToolError(message="boom", kind="x")


def test_default_formatter_renders_tool_failure_text_unchanged() -> None:
    # ToolFailure relies on the default str-passthrough formatter to keep the
    # model-facing text byte-for-byte unchanged; no custom formatter is needed.
    async def handler(
        args: _Args,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> str | ToolFailure:
        del args, cancellation_token, context
        return ToolFailure(text="ok", error=ToolError(message="c"))

    tool: Tool[_Args, str | ToolFailure] = Tool(
        name="t",
        description="d",
        args_model=_Args,
        handler=handler,
    )
    failure = ToolFailure(text="[Command cancelled]", error=ToolError(message="c"))

    assert tool.return_value_as_string(failure) == "[Command cancelled]"
    assert tool.return_value_as_string("plain result") == "plain result"


def test_tool_failure_deepcopy_preserves_text_and_error() -> None:
    failure = ToolFailure(text="boom", error=ToolError(message="boom", kind="timeout"))

    copied = copy.deepcopy(failure)

    assert isinstance(copied, ToolFailure)
    assert copied == "boom"
    assert copied.error == ToolError(message="boom", kind="timeout")


def test_tool_failure_pickle_round_trip_preserves_text_and_error() -> None:
    failure = ToolFailure(text="boom", error=ToolError(message="boom", kind="timeout"))

    # Round-trips trusted, locally serialized data to exercise __getnewargs_ex__.
    restored = pickle.loads(pickle.dumps(failure))  # noqa: S301

    assert isinstance(restored, ToolFailure)
    assert restored == "boom"
    assert restored.error == ToolError(message="boom", kind="timeout")


def test_plan_update_deepcopy_preserves_text_and_payload() -> None:
    update = PlanUpdate(
        explanation="why",
        steps=(("a", "pending"), ("b", "in_progress")),
    )

    copied = copy.deepcopy(update)

    assert isinstance(copied, PlanUpdate)
    assert copied == PLAN_UPDATED_MESSAGE
    assert copied.plan_explanation == "why"
    assert copied.plan_steps == (("a", "pending"), ("b", "in_progress"))


def test_plan_update_pickle_round_trip_preserves_text_and_payload() -> None:
    update = PlanUpdate(
        explanation="why",
        steps=(("a", "pending"), ("b", "in_progress")),
    )

    # Round-trips trusted, locally serialized data to exercise __getnewargs_ex__.
    restored = pickle.loads(pickle.dumps(update))  # noqa: S301

    assert isinstance(restored, PlanUpdate)
    assert restored == PLAN_UPDATED_MESSAGE
    assert restored.plan_explanation == "why"
    assert restored.plan_steps == (("a", "pending"), ("b", "in_progress"))


def test_tool_outcome_marks_tool_failure_as_failed() -> None:
    failure = ToolFailure(
        text="nope",
        error=ToolError(message="nope", kind="timeout"),
    )

    outcome = tool_outcome(failure)

    assert outcome == ToolOutcome(ok=False, error=ToolError("nope", "timeout"))


def test_tool_outcome_marks_exception_result_as_failed() -> None:
    outcome = tool_outcome(ValueError("oops"))

    assert outcome.ok is False
    assert outcome.error == ToolError(message="oops")


def test_tool_outcome_reads_error_mapping_without_wording() -> None:
    outcome = tool_outcome({"error": "rate limited", "data": None})

    assert outcome.ok is False
    assert outcome.error == ToolError(message="rate limited")


def test_tool_outcome_treats_plain_string_as_success() -> None:
    assert tool_outcome("failed to do the thing") == ToolOutcome(ok=True)
    assert tool_outcome({"data": 1}) == ToolOutcome(ok=True)


def test_plan_update_is_str_and_satisfies_plan_update_result() -> None:
    update = PlanUpdate(
        explanation="why",
        steps=(("a", "pending"), ("b", "in_progress")),
    )

    assert isinstance(update, str)
    assert update == PLAN_UPDATED_MESSAGE
    assert isinstance(update, PlanUpdateResult)
    assert update.plan_message == PLAN_UPDATED_MESSAGE
    assert update.plan_explanation == "why"
    assert update.plan_steps == (("a", "pending"), ("b", "in_progress"))


def test_as_plan_update_narrows_only_plan_results() -> None:
    update = PlanUpdate(explanation=None, steps=(("a", "completed"),))

    assert as_plan_update(update) is update
    assert as_plan_update("Plan updated") is None
    assert as_plan_update(ToolFailure(text="x", error=ToolError("x"))) is None
