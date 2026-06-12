"""Structured success/failure envelope for harness tool results.

Tool handlers return model-facing text by default, which leaves consumers with
no framework-derived way to tell a successful call from a failed one. This
module adds a small, typed failure signal plus the derivation the runner uses
to populate ``ok``/``error`` on tool-end run events.

Two halves work together:

1. ``ToolFailure`` is the one public, typed way a tool implementation signals
   failure. A tool returns it instead of a plain string; because it is a ``str``
   subclass the model still sees ``ToolFailure.text`` verbatim (the default tool
   formatter renders ``str`` values unchanged), while the framework reads the
   attached ``ToolError``.
2. ``tool_outcome`` is the framework-side derivation. The runner calls it for
   every tool result so every ``RunToolEndEvent`` carries an ``ok`` flag and an
   optional typed error, without consumers reflecting over result internals.

Typical usage example:

  async def run_tool(...) -> str | ToolFailure:
      if timed_out:
          return ToolFailure(text="[Command timed out]", error=ToolError(...))
      return "ok"
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, Self, cast, runtime_checkable


@dataclass(frozen=True, slots=True)
class ToolError:
    """Typed, framework-derived failure payload for one tool call.

    The payload stays intentionally small: a human-readable ``message`` plus an
    optional machine-stable ``kind`` consumers can branch on (for example
    ``"timeout"`` or ``"cancelled"``) without parsing the model-facing text.
    """

    message: str
    """Human-readable description of why the tool call failed."""

    kind: str | None = None
    """Optional stable failure category (e.g. ``"timeout"``, ``"cancelled"``)."""


class ToolFailure(str):
    """Public, typed failure result a tool returns to signal an unsuccessful call.

    ``ToolFailure`` is a ``str`` equal to its model-facing ``text``: it renders,
    compares, and persists exactly like the plain string a tool would otherwise
    return, so the conversation contract is unchanged. The attached
    :class:`ToolError` is what the framework reads to mark the call as failed on
    the tool-end run event. Returning ``ToolFailure`` is the one public, typed
    way a tool implementation signals failure.
    """

    __slots__ = ("_error",)

    _error: ToolError

    def __new__(cls, *, text: str, error: ToolError) -> Self:
        instance = super().__new__(cls, text)
        instance._error = error
        return instance

    @property
    def text(self) -> str:
        """Return the exact model-facing text for this failure."""
        return str(self)

    @property
    def error(self) -> ToolError:
        """Return the typed failure payload surfaced on the tool-end event."""
        return self._error


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """Framework-derived success/failure summary for one tool result."""

    ok: bool
    """Whether the tool call is considered successful."""

    error: ToolError | None = None
    """Typed failure payload when the call failed, otherwise ``None``."""


type PlanStepStatus = Literal["pending", "in_progress", "completed"]
"""Status values a plan step may carry."""


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
