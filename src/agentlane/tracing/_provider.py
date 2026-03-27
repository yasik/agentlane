# pylint: disable=W0603

"""Trace provider interface and implementation."""

import abc
import uuid
from datetime import UTC, datetime
from typing import Any, TypeVar

import structlog

from ._processor_interface import TracingProcessor
from ._processors import SynchronousMultiTracingProcessor
from ._scope import get_current_span, get_current_trace
from ._setup._global import set_trace_provider_factory
from ._span import NoOpSpan, Span, SpanImpl
from ._span_data import SpanData
from ._trace import NoOpTrace, Trace, TraceImpl

LOGGER = structlog.get_logger(log_tag="tracing.provider")

TSpanData = TypeVar("TSpanData", bound=SpanData)


class TraceProvider(abc.ABC):
    """Abstract interface for creating traces and spans."""

    @abc.abstractmethod
    def register_processor(self, processor: TracingProcessor) -> None:
        """Add a processor that will receive all traces and spans.

        Args:
            processor: The processor to register.
        """

    @abc.abstractmethod
    def set_processors(self, processors: list[TracingProcessor]) -> None:
        """Replace the list of processors.

        Args:
            processors: New list of processors.
        """

    @abc.abstractmethod
    def get_current_trace(self) -> Trace | None:
        """Return the currently active trace, if any.

        Returns:
            The current trace or None.
        """

    @abc.abstractmethod
    def get_current_span(self) -> Span[Any] | None:
        """Return the currently active span, if any.

        Returns:
            The current span or None.
        """

    @abc.abstractmethod
    def set_disabled(self, disabled: bool) -> None:
        """Enable or disable tracing globally.

        Args:
            disabled: Whether to disable tracing.
        """

    @abc.abstractmethod
    def time_iso(self) -> str:
        """Return the current time in ISO 8601 format.

        Returns:
            ISO formatted timestamp.
        """

    @abc.abstractmethod
    def gen_trace_id(self) -> str:
        """Generate a new trace identifier.

        Returns:
            A unique trace ID.
        """

    @abc.abstractmethod
    def gen_span_id(self) -> str:
        """Generate a new span identifier.

        Returns:
            A unique span ID.
        """

    @abc.abstractmethod
    def gen_group_id(self) -> str:
        """Generate a new group identifier.

        Returns:
            A unique group ID.
        """

    @abc.abstractmethod
    def create_trace(
        self,
        name: str,
        trace_id: str | None = None,
        group_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        disabled: bool = False,
    ) -> Trace:
        """Create a new trace.

        Args:
            name: Name of the workflow being traced.
            trace_id: Optional trace ID (generated if not provided).
            group_id: Optional group ID for linking traces.
            metadata: Optional metadata dictionary.
            disabled: Whether to create a no-op trace.

        Returns:
            A new Trace instance.
        """

    @abc.abstractmethod
    def create_span(
        self,
        span_data: TSpanData,
        span_id: str | None = None,
        parent: Trace | Span[Any] | None = None,
        disabled: bool = False,
    ) -> Span[TSpanData]:
        """Create a new span.

        Args:
            span_data: The span data.
            span_id: Optional span ID (generated if not provided).
            parent: Optional parent trace or span.
            disabled: Whether to create a no-op span.

        Returns:
            A new Span instance.
        """

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Clean up any resources used by the provider."""


class DefaultTraceProvider(TraceProvider):
    """Default implementation of TraceProvider with OTEL integration."""

    def __init__(self, disabled: bool = False) -> None:
        """Initialize the default trace provider.

        Args:
            disabled: Whether to disable tracing.
        """
        self._multi_processor = SynchronousMultiTracingProcessor()
        self._disabled = disabled

    def register_processor(self, processor: TracingProcessor) -> None:
        """Add a processor to receive all traces and spans."""
        self._multi_processor.add_processor(processor)

    def set_processors(self, processors: list[TracingProcessor]) -> None:
        """Replace all processors with the given list."""
        self._multi_processor.set_processors(processors)

    def get_current_trace(self) -> Trace | None:
        """Return the currently active trace."""
        return get_current_trace()

    def get_current_span(self) -> Span[Any] | None:
        """Return the currently active span."""
        return get_current_span()

    def set_disabled(self, disabled: bool) -> None:
        """Enable or disable tracing globally."""
        self._disabled = disabled

    def time_iso(self) -> str:
        """Return the current time in ISO 8601 format."""
        return datetime.now(UTC).isoformat()

    def gen_trace_id(self) -> str:
        """Generate a new trace ID."""
        return f"trace_{uuid.uuid4().hex}"

    def gen_span_id(self) -> str:
        """Generate a new span ID."""
        return f"span_{uuid.uuid4().hex[:24]}"

    def gen_group_id(self) -> str:
        """Generate a new group ID."""
        return f"group_{uuid.uuid4().hex[:24]}"

    def create_trace(
        self,
        name: str,
        trace_id: str | None = None,
        group_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        disabled: bool = False,
    ) -> Trace:
        """Create a new trace."""
        if self._disabled or disabled:
            return NoOpTrace()

        return TraceImpl(
            name=name,
            trace_id=trace_id or self.gen_trace_id(),
            group_id=group_id,
            metadata=metadata,
            processor=self._multi_processor,
        )

    def create_span(
        self,
        span_data: TSpanData,
        span_id: str | None = None,
        parent: Trace | Span[Any] | None = None,
        disabled: bool = False,
    ) -> Span[TSpanData]:
        """Create a new span."""
        if self._disabled or disabled:
            return NoOpSpan(span_data)

        resolved_trace_id: str
        parent_id: str | None = None

        if parent is None:
            current_span = get_current_span()
            current_trace = get_current_trace()
            if current_trace is None:
                LOGGER.warning(
                    "No active trace. Make sure to start a trace with `trace()` first. "
                    "Returning NoOpSpan."
                )
                return NoOpSpan(span_data)

            if isinstance(current_trace, NoOpTrace) or isinstance(
                current_span, NoOpSpan
            ):
                return NoOpSpan(span_data)

            parent_id = current_span.span_id if current_span is not None else None
            resolved_trace_id = current_trace.trace_id
        elif isinstance(parent, Trace):
            if isinstance(parent, NoOpTrace):
                return NoOpSpan(span_data)
            resolved_trace_id = parent.trace_id
        else:
            if isinstance(parent, NoOpSpan):
                return NoOpSpan(span_data)
            parent_id = parent.span_id
            resolved_trace_id = parent.trace_id

        return SpanImpl(
            trace_id=resolved_trace_id,
            span_id=span_id or self.gen_span_id(),
            parent_id=parent_id,
            processor=self._multi_processor,
            span_data=span_data,
        )

    def shutdown(self) -> None:
        """Shutdown the provider and all processors."""
        if self._disabled:
            return

        try:
            self._multi_processor.shutdown()
        except Exception as error:  # noqa: BLE001
            LOGGER.error(f"Error shutting down trace provider: {error}")


set_trace_provider_factory(DefaultTraceProvider)
