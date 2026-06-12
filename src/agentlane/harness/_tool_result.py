"""Framework-side derivation over the tool success/failure envelope.

The typed envelope primitives a tool returns —  :class:`ToolError`,
:class:`ToolFailure`, and the :data:`PlanStepStatus` alias — live in
``agentlane.models`` so the model-layer tool executor can build a structured
failure without importing from the harness package above it. This module
re-exports them unchanged (so ``from agentlane.harness import ToolFailure`` keeps
working) and adds the framework-side halves the runner uses:

1. ``tool_outcome`` derives a success/failure :class:`ToolOutcome` for every tool
   result so each ``RunToolEndEvent`` carries an ``ok`` flag and an optional
   typed error, without consumers reflecting over result internals.
2. ``PlanUpdateResult`` / ``as_plan_update`` narrow a tool result to its
   structured plan view so the runner can emit a typed plan-updated event.

Typical usage example:

  async def run_tool(...) -> str | ToolFailure:
      if timed_out:
          return ToolFailure(text="[Command timed out]", error=ToolError(...))
      return "ok"
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

from ..models import PlanStepStatus, ToolError, ToolFailure

__all__ = [
    "PlanStepStatus",
    "PlanUpdateResult",
    "ToolError",
    "ToolFailure",
    "ToolOutcome",
    "as_plan_update",
    "tool_outcome",
]


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """Framework-derived success/failure summary for one tool result."""

    ok: bool
    """Whether the tool call is considered successful."""

    error: ToolError | None = None
    """Typed failure payload when the call failed, otherwise ``None``."""


@runtime_checkable
class PlanUpdateResult(Protocol):
    """Structural result a tool returns to publish a structured plan update.

    The first-party plan tool returns a value satisfying this protocol so the
    runner can emit a typed plan-updated run event without consumers
    string-matching the tool's model-facing success message. ``plan_message``
    is the unchanged model-facing text rendered back into the conversation.
    """

    @property
    def plan_message(self) -> str:
        """Model-facing success text for the plan update."""
        ...

    @property
    def plan_explanation(self) -> str | None:
        """Optional model-supplied reason for the plan update."""
        ...

    @property
    def plan_steps(self) -> Sequence[tuple[str, PlanStepStatus]]:
        """Ordered ``(step, status)`` pairs describing the current plan."""
        ...


def tool_outcome(result: object) -> ToolOutcome:
    """Derive a success/failure outcome from one raw tool result.

    The runner calls this for every executed tool so consumers receive a
    framework-derived ``ok`` flag and typed error instead of inferring failure
    from result wording. Recognized failure shapes, in order:

    1. :class:`ToolFailure` — the explicit, typed tool failure signal.
    2. ``BaseException`` — a raised-and-captured tool error.
    3. A mapping with a truthy ``"error"`` entry — the structured-mapping
       convention third-party tools commonly return.

    Anything else is treated as success. First-party tools that render failures
    as plain strings stay ``ok`` because result wording is not a contract; tools
    that need a failure marked should return :class:`ToolFailure`.

    Args:
        result: The raw value returned by a tool handler (or a captured error).

    Returns:
        The derived :class:`ToolOutcome`.
    """
    if isinstance(result, ToolFailure):
        return ToolOutcome(ok=False, error=result.error)
    if isinstance(result, BaseException):
        return ToolOutcome(
            ok=False,
            error=ToolError(message=str(result) or type(result).__name__),
        )
    if isinstance(result, Mapping):
        error_value = cast(Mapping[object, object], result).get("error")
        if error_value:
            return ToolOutcome(ok=False, error=ToolError(message=str(error_value)))
    return ToolOutcome(ok=True)


def as_plan_update(result: object) -> PlanUpdateResult | None:
    """Return the plan-update view of a tool result, or ``None``.

    ``PlanUpdate`` from the plan tool is a ``str`` subclass; this narrows a raw
    tool result to its structured plan view so the runner can emit a typed
    plan-updated event.
    """
    if isinstance(result, PlanUpdateResult):
        return result
    return None
