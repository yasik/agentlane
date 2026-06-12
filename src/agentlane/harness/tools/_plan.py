"""Plan tool implementation for first-party harness base tools."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Self

from pydantic import BaseModel, Field

from agentlane.models import PlanStepStatus, Tool, ToolExecutionContext
from agentlane.runtime import CancellationToken

from ._types import HarnessToolDefinition

PLAN_TOOL_NAME = "write_plan"
"""Public, stable name of the first-party plan tool."""

PLAN_UPDATED_MESSAGE = "Plan updated"
"""Public, stable model-facing success message returned by the plan tool."""

_TOOL_NAME = PLAN_TOOL_NAME
_TOOL_DESCRIPTION = """Writes or updates the task plan.
Provide an optional explanation and a list of plan items, each with a step and status.
At most one step can be in_progress at a time.
"""
_TOOL_PROMPT_SNIPPET = "Write or update the task plan"
_TOOL_PROMPT_GUIDELINE = """
Use `write_plan` to maintain a visible, step-by-step plan for non-trivial tasks.
The plan demonstrates your understanding and approach, and gives the user checkpoints for feedback.

**Use a plan when:**
- The task has multiple logical phases or dependencies where sequencing matters
- The user asked for several distinct things in one prompt
- The work is ambiguous and benefits from outlined high-level goals
- You discover additional steps mid-task that you'll do before yielding
- The user explicitly asks for a plan or "TODOs"

**Don't use a plan for:**
- Single-step or trivially answerable requests
- Padding simple work with filler steps
- Steps you can't actually execute

### Plan quality

Steps should be meaningful, logically ordered, and easy to verify. Each step is one sentence,
max 5–7 words. Aim for substance over ceremony.

Good:
1. Add CLI entry with file args
2. Parse Markdown via CommonMark library
3. Apply semantic HTML template
4. Handle code blocks, images, links
5. Add error handling for invalid files

Bad (vague, low-information):
1. Create CLI tool
2. Add Markdown parser
3. Convert to HTML

### Mechanics

Each step has a `status`: `pending`, `in_progress`, or `completed`. Exactly one step is `in_progress` until the task is done.

- Mark steps `completed` as you finish them; set the next one `in_progress` in the same call. Multiple completions per call is fine.
- If the plan changes mid-task, call `write_plan` with the updated plan and include an `explanation` of why.
- When all steps are done, call `write_plan` once more to mark everything `completed`.
- After calling `write_plan`, do **not** repeat the plan in your reply — the harness renders it. Briefly note the change or next step instead.
- If a single implementation pass completes everything, mark all steps `completed` in one call.
"""

_GENERIC_PLAN_ERROR = "failed to update plan"


class PlanUpdate(str):
    """Successful plan-tool result carrying the structured plan.

    Subclasses ``str`` so the model-facing path is byte-for-byte unchanged: the
    value equals ``PLAN_UPDATED_MESSAGE`` and renders as that text in the
    conversation. The structured plan rides alongside as attributes, satisfying
    :class:`agentlane.harness.PlanUpdateResult` so the runner can emit a typed
    ``RunPlanUpdatedEvent`` without consumers string-matching the message.
    """

    __slots__ = ("_explanation", "_steps")

    _explanation: str | None
    _steps: tuple[tuple[str, PlanStepStatus], ...]

    def __new__(
        cls,
        *,
        explanation: str | None,
        steps: tuple[tuple[str, PlanStepStatus], ...],
    ) -> Self:
        instance = super().__new__(cls, PLAN_UPDATED_MESSAGE)
        instance._explanation = explanation
        instance._steps = steps
        return instance

    def __getnewargs_ex__(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Return the keyword-only ``__new__`` arguments for copy and pickle.

        ``PlanUpdate`` is a ``str`` subclass with a keyword-only ``__new__``.
        Without this hook ``copy.deepcopy`` and ``pickle`` reconstruct the value
        with positional ``str`` newargs, which the keyword-only signature
        rejects. Returning the keyword form preserves the structured plan across
        a round-trip.
        """
        return (), {"explanation": self._explanation, "steps": self._steps}

    @property
    def plan_message(self) -> str:
        """Return the model-facing success text for the plan update."""
        return PLAN_UPDATED_MESSAGE

    @property
    def plan_explanation(self) -> str | None:
        """Return the optional model-supplied reason for the plan update."""
        return self._explanation

    @property
    def plan_steps(self) -> tuple[tuple[str, PlanStepStatus], ...]:
        """Return ordered ``(step, status)`` pairs for the current plan."""
        return self._steps


class _PlanItem(BaseModel):
    """Model-visible plan item."""

    step: str = Field(description="Concise description of this step.")
    status: PlanStepStatus = Field(description="Current step status.")


class _ToolArgs(BaseModel):
    """Model-visible arguments for the plan tool."""

    explanation: str | None = Field(
        default=None,
        description="Optional brief reason for this plan update.",
    )
    plan: list[_PlanItem] = Field(description="The list of steps.")


def plan_tool(
    *,
    persist_to: Callable[[dict[str, object]], None] | None = None,
    prompt_snippet: str | None = _TOOL_PROMPT_SNIPPET,
    prompt_guidelines: Sequence[str] = (_TOOL_PROMPT_GUIDELINE,),
) -> HarnessToolDefinition:
    """Build the first-party task-plan harness tool.

    Args:
        persist_to: Optional callback that receives the latest plan update.
            `HarnessToolsShim` uses this to persist state in `RunState`.
        prompt_snippet: Optional prompt snippet rendered by `HarnessToolsShim`.
        prompt_guidelines: Prompt guidance rendered by `HarnessToolsShim`.

    Returns:
        HarnessToolDefinition: Executable plan tool with prompt metadata.
    """

    async def run_tool(
        args: _ToolArgs,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> str:
        del context
        del cancellation_token
        try:
            return _write_plan(
                args,
                persist_to=persist_to,
            )
        except Exception:
            return _GENERIC_PLAN_ERROR

    return HarnessToolDefinition(
        tool=_build_plan_tool(handler=run_tool),
        prompt_snippet=prompt_snippet,
        prompt_guidelines=tuple(prompt_guidelines),
    )


def plan_state_key(shim_name: str) -> str:
    """Return the persisted shim-state key for the latest plan."""
    return f"{shim_name}:plan"


def _write_plan(
    args: _ToolArgs,
    *,
    persist_to: Callable[[dict[str, object]], None] | None,
) -> str:
    """Persist one plan update and return the structured success result.

    The success value is a :class:`PlanUpdate`, which is a ``str`` equal to the
    model-facing success message and additionally carries the structured plan
    for typed run-event emission.
    """
    validation_error = _validate_plan(args)
    if validation_error is not None:
        return validation_error

    if persist_to is not None:
        persist_to(_plan_snapshot(args))

    return PlanUpdate(
        explanation=args.explanation,
        steps=tuple((item.step, item.status) for item in args.plan),
    )


def _validate_plan(args: _ToolArgs) -> str | None:
    """Return one model-facing validation error when the plan is invalid."""
    if not args.plan:
        return "plan must contain at least one item"

    in_progress_count = 0
    for item in args.plan:
        if item.step.strip() == "":
            return "plan steps must not be empty"
        if item.status == "in_progress":
            in_progress_count += 1

    if in_progress_count > 1:
        return "at most one plan step can be in_progress"
    return None


def _plan_snapshot(args: _ToolArgs) -> dict[str, object]:
    """Return the serialized form persisted by the shim."""
    return {
        "explanation": args.explanation,
        "plan": [{"step": item.step, "status": item.status} for item in args.plan],
    }


def _build_plan_tool(
    *,
    handler: Callable[
        [_ToolArgs, CancellationToken, ToolExecutionContext],
        Awaitable[str],
    ],
) -> Tool[_ToolArgs, str]:
    """Build a concrete executable plan tool."""
    return Tool(
        name=_TOOL_NAME,
        description=_TOOL_DESCRIPTION,
        args_model=_ToolArgs,
        handler=handler,
    )
