"""Processor interface for handling trace and span lifecycle events."""

import abc
from typing import Any


class TracingProcessor(abc.ABC):
    """Abstract base class for processing trace and span events."""

    @abc.abstractmethod
    def on_trace_start(self, trace: Any) -> None:
        """Called when a trace is started.

        Args:
            trace: The trace that was started.
        """

    @abc.abstractmethod
    def on_trace_end(self, trace: Any) -> None:
        """Called when a trace is finished.

        Args:
            trace: The trace that was finished.
        """

    @abc.abstractmethod
    def on_span_start(self, span: Any) -> None:
        """Called when a span is started.

        Args:
            span: The span that was started.
        """

    @abc.abstractmethod
    def on_span_end(self, span: Any) -> None:
        """Called when a span is finished.

        Args:
            span: The span that was finished.
        """

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Shutdown the processor and clean up resources."""

    @abc.abstractmethod
    def force_flush(self) -> None:
        """Force flush any buffered data."""
