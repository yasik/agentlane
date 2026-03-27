"""Context management for current trace and span."""

import contextvars
from typing import Any

# Context variables for tracking current span and trace
_current_span: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "agentlane_tracing_current_span", default=None
)

_current_trace: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "agentlane_tracing_current_trace", default=None
)


def get_current_span() -> Any | None:
    """Get the currently active span.

    Returns:
        The current span or None.
    """
    return _current_span.get()


def set_current_span(
    span: Any | None,
) -> contextvars.Token[Any | None]:
    """Set the current span.

    Args:
        span: The span to set as current.

    Returns:
        A token for resetting the span.
    """
    return _current_span.set(span)


def reset_current_span(token: contextvars.Token[Any | None]) -> None:
    """Reset the current span using a token.

    Args:
        token: The token from set_current_span.
    """
    _current_span.reset(token)


def get_current_trace() -> Any | None:
    """Get the currently active trace.

    Returns:
        The current trace or None.
    """
    return _current_trace.get()


def set_current_trace(trace: Any | None) -> contextvars.Token[Any | None]:
    """Set the current trace.

    Args:
        trace: The trace to set as current.

    Returns:
        A token for resetting the trace.
    """
    return _current_trace.set(trace)


def reset_current_trace(token: contextvars.Token[Any | None]) -> None:
    """Reset the current trace using a token.

    Args:
        token: The token from set_current_trace.
    """
    _current_trace.reset(token)
