"""Span interface and implementations."""

import abc
import asyncio
import contextvars
import traceback
from types import TracebackType
from typing import Any, Self

import structlog

from ._processor_interface import TracingProcessor
from ._scope import reset_current_span, set_current_span
from ._setup import get_trace_provider
from ._span_data import SpanData

LOGGER = structlog.get_logger(log_tag="tracing.span")


class SpanError(dict[str, Any]):
    """Error information for a span."""

    def __init__(self, message: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, data=data)


class Span[TSpanData: SpanData](abc.ABC):
    """Abstract base class for spans."""

    @property
    @abc.abstractmethod
    def trace_id(self) -> str:
        """Get the trace ID."""

    @property
    @abc.abstractmethod
    def span_id(self) -> str:
        """Get the span ID."""

    @property
    @abc.abstractmethod
    def span_data(self) -> TSpanData:
        """Get the span data."""

    @property
    @abc.abstractmethod
    def parent_id(self) -> str | None:
        """Get the parent span ID."""

    @abc.abstractmethod
    def start(self, mark_as_current: bool = False) -> None:
        """Start the span.

        Args:
            mark_as_current: Whether to mark as the current span.
        """

    @abc.abstractmethod
    def finish(self, reset_current: bool = False) -> None:
        """Finish the span.

        Args:
            reset_current: Whether to reset the current span.
        """

    @abc.abstractmethod
    def __enter__(self) -> Self:
        """Enter the span context."""

    @abc.abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the span context."""

    @abc.abstractmethod
    def set_error(self, error: SpanError) -> None:
        """Set an error on the span."""

    @property
    @abc.abstractmethod
    def error(self) -> SpanError | None:
        """Get the span error."""

    @abc.abstractmethod
    def export(self) -> dict[str, Any] | None:
        """Export the span as a dictionary."""

    @property
    @abc.abstractmethod
    def started_at(self) -> str | None:
        """Get the start time."""

    @property
    @abc.abstractmethod
    def ended_at(self) -> str | None:
        """Get the end time."""


class NoOpSpan[TSpanData: SpanData](Span[TSpanData]):
    """No-operation span for when tracing is disabled."""

    def __init__(self, span_data: TSpanData) -> None:
        self._span_data = span_data
        self._prev_span_token: contextvars.Token[Span[Any] | None] | None = None

    @property
    def trace_id(self) -> str:
        return "no-op"

    @property
    def span_id(self) -> str:
        return "no-op"

    @property
    def span_data(self) -> TSpanData:
        return self._span_data

    @property
    def parent_id(self) -> str | None:
        return None

    def start(self, mark_as_current: bool = False) -> None:
        if mark_as_current:
            self._prev_span_token = set_current_span(self)

    def finish(self, reset_current: bool = False) -> None:
        if reset_current and self._prev_span_token is not None:
            reset_current_span(self._prev_span_token)
            self._prev_span_token = None

    def __enter__(self) -> Self:
        self.start(mark_as_current=True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        del exc_val, exc_tb
        reset_current = exc_type is not GeneratorExit
        self.finish(reset_current=reset_current)

    def set_error(self, error: SpanError) -> None:
        del error

    @property
    def error(self) -> SpanError | None:
        return None

    def export(self) -> dict[str, Any] | None:
        return None

    @property
    def started_at(self) -> str | None:
        return None

    @property
    def ended_at(self) -> str | None:
        return None


class SpanImpl[TSpanData: SpanData](Span[TSpanData]):
    """Concrete span implementation."""

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        parent_id: str | None,
        processor: TracingProcessor,
        span_data: TSpanData,
    ) -> None:
        self._trace_id = trace_id
        self._span_id = span_id
        self._parent_id = parent_id
        self._started_at: str | None = None
        self._ended_at: str | None = None
        self._processor = processor
        self._error: SpanError | None = None
        self._prev_span_token: contextvars.Token[Span[Any] | None] | None = None
        self._span_data = span_data

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def span_id(self) -> str:
        return self._span_id

    @property
    def span_data(self) -> TSpanData:
        return self._span_data

    @property
    def parent_id(self) -> str | None:
        return self._parent_id

    def start(self, mark_as_current: bool = False) -> None:
        if self._started_at is not None:
            LOGGER.warning("Span already started")
            return

        self._started_at = get_trace_provider().time_iso()
        self._processor.on_span_start(self)

        if mark_as_current:
            self._prev_span_token = set_current_span(self)

    def finish(self, reset_current: bool = False) -> None:
        if self._ended_at is not None:
            LOGGER.warning("Span already finished")
            return

        self._ended_at = get_trace_provider().time_iso()
        self._processor.on_span_end(self)

        if reset_current and self._prev_span_token is not None:
            reset_current_span(self._prev_span_token)
            self._prev_span_token = None

    def __enter__(self) -> Self:
        self.start(mark_as_current=True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None and exc_type is not GeneratorExit:
            if issubclass(exc_type, (asyncio.CancelledError, TimeoutError)):
                self.set_error(
                    SpanError(
                        message=(
                            f"{exc_type.__module__}.{exc_type.__name__}: "
                            f"{exc_val or 'cancelled'}"
                        )
                    )
                )
            else:
                error_data = {
                    "error": "".join(
                        traceback.format_exception(exc_type, exc_val, exc_tb)
                    ),
                }
                self.set_error(
                    SpanError(
                        message=(
                            str(exc_val)
                            if exc_val is not None
                            else f"Exception: {exc_type.__name__}"
                        ),
                        data=error_data,
                    )
                )

        self.finish(reset_current=exc_type is not GeneratorExit)

    def set_error(self, error: SpanError) -> None:
        self._error = error

    @property
    def error(self) -> SpanError | None:
        return self._error

    @property
    def started_at(self) -> str | None:
        return self._started_at

    @property
    def ended_at(self) -> str | None:
        return self._ended_at

    def export(self) -> dict[str, Any] | None:
        return {
            "object": "trace.span",
            "id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self._parent_id,
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "span_data": self.span_data.export(),
            "error": self._error,
        }
