"""Helpers for propagating tracing context across async task boundaries."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ._scope import (
    get_current_span,
    get_current_trace,
    reset_current_span,
    reset_current_trace,
    set_current_span,
    set_current_trace,
)
from ._span import NoOpSpan, Span
from ._trace import NoOpTrace, Trace


@dataclass(slots=True)
class _SpanContextSnapshot:
    """Captured trace/span context for later adoption."""

    trace: Trace
    span: Span[Any] | None


_MESSAGE_CONTEXTS: dict[str, _SpanContextSnapshot] = {}
_MESSAGE_CONTEXT_LOCK = threading.Lock()


def capture_parent_context(message_id: str) -> None:
    """Store the currently active trace/span under the given message ID."""
    trace = get_current_trace()
    if trace is None or isinstance(trace, NoOpTrace):
        return

    span = get_current_span()
    if span is not None and isinstance(span, NoOpSpan):
        span = None

    with _MESSAGE_CONTEXT_LOCK:
        _MESSAGE_CONTEXTS[message_id] = _SpanContextSnapshot(trace=trace, span=span)


def discard_parent_context(message_id: str) -> None:
    """Remove any stored context for the given message ID."""
    with _MESSAGE_CONTEXT_LOCK:
        _MESSAGE_CONTEXTS.pop(message_id, None)


@contextmanager
def adopt_parent_context(message_id: str) -> Iterator[None]:
    """Temporarily adopt the context captured for the message ID, if any."""
    with _MESSAGE_CONTEXT_LOCK:
        snapshot = _MESSAGE_CONTEXTS.pop(message_id, None)

    if snapshot is None:
        yield
        return

    trace_token = set_current_trace(snapshot.trace)
    span_token = set_current_span(snapshot.span)
    try:
        yield
    finally:
        reset_current_span(span_token)
        reset_current_trace(trace_token)
