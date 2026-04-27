"""Plan tool implementation for first-party harness base tools."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

from pydantic import BaseModel, Field

from agentlane.models import Tool
from agentlane.runtime import CancellationToken

from ._types import HarnessToolDefinition

_TOOL_NAME = "update_plan"
_TOOL_DESCRIPTION = """Updates the task plan.
Provide an optional explanation and a list of plan items, each with a step and status.
At most one step can be in_progress at a time.
"""
_TOOL_PROMPT_SNIPPET = "Update the task plan"
_TOOL_PROMPT_GUIDELINE = """
Use `update_plan` to maintain a visible, step-by-step plan for non-trivial tasks.
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
- If the plan changes mid-task, call `update_plan` with the updated plan and include an `explanation` of why.
- When all steps are done, call `update_plan` once more to mark everything `completed`.
- After calling `update_plan`, do **not** repeat the plan in your reply — the harness renders it. Briefly note the change or next step instead.
- If a single implementation pass completes everything, mark all steps `completed` in one call.
"""

_GENERIC_PLAN_ERROR = "failed to update plan"
_PLAN_UPDATED_MESSAGE = "Plan updated"


class _PlanItem(BaseModel):
    """Model-visible plan item."""

    step: str = Field(description="Concise description of this step.")
    status: Literal["pending", "in_progress", "completed"] = Field(
        description="Current step status."
    )


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
    """Persist one plan update and return the model-facing success message."""
    if persist_to is not None:
        persist_to(_plan_snapshot(args))

    return _PLAN_UPDATED_MESSAGE


def _plan_snapshot(args: _ToolArgs) -> dict[str, object]:
    """Return the serialized form persisted by the shim."""
    return {
        "explanation": args.explanation,
        "plan": [{"step": item.step, "status": item.status} for item in args.plan],
    }


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
