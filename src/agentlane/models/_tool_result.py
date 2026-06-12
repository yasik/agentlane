"""Typed success/failure envelope primitives for tool results.

A tool handler returns model-facing text by default, which leaves consumers with
no framework-derived way to tell a successful call from a failed one. This
module defines the small, typed failure signal a tool returns to mark a call as
failed, plus the plan-step status alias shared by the plan tool and its run
events.

These primitives live in ``models`` rather than ``harness`` so the model-layer
tool executor can build a structured failure without importing from the harness
package above it. The harness re-exports them from ``agentlane.harness`` so every
existing import path keeps working.

Typical usage example:

  async def run_tool(...) -> str | ToolFailure:
      if timed_out:
          return ToolFailure(text="[Command timed out]", error=ToolError(...))
      return "ok"
"""

from dataclasses import dataclass
from typing import Any, Literal, Self


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

    def __getnewargs_ex__(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Return the keyword-only ``__new__`` arguments for copy and pickle.

        ``ToolFailure`` is a ``str`` subclass with a keyword-only ``__new__``.
        Without this hook ``copy.deepcopy`` and ``pickle`` reconstruct the value
        with positional ``str`` newargs, which the keyword-only signature
        rejects. Returning the keyword form preserves both the model-facing text
        and the structured :class:`ToolError` across a round-trip.
        """
        return (), {"text": str(self), "error": self._error}

    @property
    def text(self) -> str:
        """Return the exact model-facing text for this failure."""
        return str(self)

    @property
    def error(self) -> ToolError:
        """Return the typed failure payload surfaced on the tool-end event."""
        return self._error


type PlanStepStatus = Literal["pending", "in_progress", "completed"]
"""Status values a plan step may carry."""
