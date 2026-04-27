"""Plan tool implementation for first-party harness base tools."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from agentlane.models import Tool
from agentlane.runtime import CancellationToken

from ._types import HarnessToolDefinition

_TOOL_NAME = "write_plan"
_TOOL_DESCRIPTION = (
    "Creates or replaces a concise task plan with pending, in_progress, and "
    "completed steps. At most one step may be in_progress."
)
_TOOL_PROMPT_SNIPPET = "Create or update a plan based on the user request"
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

Steps should be meaningful, logically ordered, and easy to verify. Each step is one sentence, max 5–7 words. Aim for substance over ceremony.

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


class _PlanItem(BaseModel):
    """Model-visible plan item."""

    step: str = Field(description="Concise description of this step.")
    status: Literal["pending", "in_progress", "completed"] = Field(
        description="Current step status."
    )


class _ToolArgs(BaseModel):
    """Model-visible arguments for the plan tool."""

    task: str = Field(description="Short name for the current task.")
    items: list[_PlanItem] = Field(
        description="Authoritative replacement list of plan items."
    )
    explanation: str | None = Field(
        default=None,
        description="Optional brief reason for this plan update.",
    )


@dataclass(frozen=True, slots=True)
class _ValidatedPlan:
    """Validated plan content ready to render and persist."""

    task: str
    items: tuple[_PlanItem, ...]
    explanation: str | None


def plan_tool(
    *,
    persist_to: Callable[[dict[str, object]], None] | None = None,
    prompt_snippet: str | None = _TOOL_PROMPT_SNIPPET,
    prompt_guidelines: Sequence[str] = (_TOOL_PROMPT_GUIDELINE,),
) -> HarnessToolDefinition:
    """Build the first-party task-plan harness tool.

    Args:
        persist_to: Optional callback that receives the latest validated plan.
            `HarnessToolsShim` uses this to persist state in `RunState`.
        prompt_snippet: Optional prompt snippet rendered by `HarnessToolsShim`.
        prompt_guidelines: Prompt guidance rendered by `HarnessToolsShim`.

    Returns:
        HarnessToolDefinition: Executable plan tool with prompt metadata.
    """

    async def run_tool(
        args: _ToolArgs,
        cancellation_token: CancellationToken,
    ) -> str:
        del cancellation_token
        try:
            return _update_plan(
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


def _update_plan(
    args: _ToolArgs,
    *,
    persist_to: Callable[[dict[str, object]], None] | None,
) -> str:
    """Validate, optionally persist, and render one plan update."""
    validation_result = _validate_plan(args)
    if isinstance(validation_result, str):
        return validation_result

    if persist_to is not None:
        persist_to(_plan_snapshot(validation_result))

    return _format_plan_output(validation_result)


def _validate_plan(args: _ToolArgs) -> _ValidatedPlan | str:
    """Return validated plan content or a model-facing error."""
    task = args.task.strip()
    if task == "":
        return "task must not be empty"
    if not args.items:
        return "plan must include at least one item"

    in_progress_count = 0
    items: list[_PlanItem] = []
    for item in args.items:
        step = item.step.strip()
        if step == "":
            return "plan item step must not be empty"
        if item.status == "in_progress":
            in_progress_count += 1
        items.append(_PlanItem(step=step, status=item.status))

    if in_progress_count > 1:
        return "plan may contain at most one in_progress item"

    explanation = _normalize_explanation(args.explanation)
    return _ValidatedPlan(
        task=task,
        items=tuple(items),
        explanation=explanation,
    )


def _normalize_explanation(value: str | None) -> str | None:
    """Normalize optional explanation text."""
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    return stripped


def _plan_snapshot(plan: _ValidatedPlan) -> dict[str, object]:
    """Return the serialized form persisted by the shim."""
    return {
        "task": plan.task,
        "items": [{"step": item.step, "status": item.status} for item in plan.items],
        "explanation": plan.explanation,
    }


def _format_plan_output(plan: _ValidatedPlan) -> str:
    """Render one validated plan as plain text for model-visible output."""
    lines = [f"Plan: {plan.task}", ""]
    if plan.explanation is not None:
        lines.extend((f"Explanation: {plan.explanation}", ""))

    lines.extend(f"{_status_marker(item.status)} {item.step}" for item in plan.items)
    counts = _status_counts(plan.items)
    lines.extend(
        (
            "",
            "Status: "
            f"{counts['pending']} pending, "
            f"{counts['in_progress']} in progress, "
            f"{counts['completed']} completed.",
        )
    )
    return "\n".join(lines)


def _build_plan_tool(
    *,
    handler: Callable[[_ToolArgs, CancellationToken], Awaitable[str]],
) -> Tool[_ToolArgs, str]:
    """Build a concrete executable plan tool."""
    return Tool(
        name=_TOOL_NAME,
        description=_TOOL_DESCRIPTION,
        args_model=_ToolArgs,
        handler=handler,
    )


def _status_marker(status: Literal["pending", "in_progress", "completed"]) -> str:
    if status == "pending":
        return "- [ ]"
    if status == "in_progress":
        return "- [~]"
    return "- [x]"


def _status_counts(items: tuple[_PlanItem, ...]) -> dict[str, int]:
    counts = {
        "pending": 0,
        "in_progress": 0,
        "completed": 0,
    }
    for item in items:
        counts[item.status] += 1
    return counts
