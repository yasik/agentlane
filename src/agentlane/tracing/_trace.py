"""Trace interface and implementations."""

import abc
import contextvars
from collections.abc import Mapping
from types import TracebackType
from typing import Any, Self

import structlog

from ._processor_interface import TracingProcessor
from ._scope import reset_current_trace as scope_reset_current_trace
from ._scope import set_current_trace as scope_set_current_trace
from ._setup import get_trace_provider

LOGGER = structlog.get_logger(log_tag="tracing.trace")


class Trace(abc.ABC):
    """Abstract base class for traces (root workflow spans)."""

    @abc.abstractmethod
    def __enter__(self) -> Self:
        """Enter the trace context."""

    @abc.abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the trace context."""

    @abc.abstractmethod
    def start(self, mark_as_current: bool = False) -> None:
        """Start the trace.

        Args:
            mark_as_current: Whether to mark as the current trace.
        """

    @abc.abstractmethod
    def finish(self, reset_current: bool = False) -> None:
        """Finish the trace.

        Args:
            reset_current: Whether to reset the current trace.
        """

    @abc.abstractmethod
    def set_first_input(self, first_input: Mapping[str, Any]) -> None:
        """Set the first input of the trace."""

    @abc.abstractmethod
    def set_last_output(self, last_output: Mapping[str, Any]) -> None:
        """Set the last output of the trace."""

    @property
    @abc.abstractmethod
    def trace_id(self) -> str:
        """Get the trace ID."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Get the name of the workflow being traced."""

    @property
    @abc.abstractmethod
    def first_input(self) -> Mapping[str, Any] | None:
        """Get the first input of the trace."""

    @property
    @abc.abstractmethod
    def last_output(self) -> Mapping[str, Any] | None:
        """Get the last output of the trace."""

    @property
    @abc.abstractmethod
    def metadata(self) -> dict[str, Any] | None:
        """Get the metadata of the trace."""

    @property
    @abc.abstractmethod
    def started_at(self) -> str | None:
        """Get the start time."""

    @property
    @abc.abstractmethod
    def ended_at(self) -> str | None:
        """Get the end time."""

    @abc.abstractmethod
    def export(self) -> dict[str, Any] | None:
        """Export the trace as a dictionary."""


class NoOpTrace(Trace):
    """No-operation trace for when tracing is disabled."""

    def __init__(self) -> None:
        self._started = False
        self._prev_context_token: contextvars.Token[Trace | None] | None = None

    def __enter__(self) -> Self:
        if self._started:
            if self._prev_context_token is None:
                LOGGER.error("Trace already started but no context token set")
            return self

        self._started = True
        self.start(mark_as_current=True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        del exc_type, exc_val, exc_tb
        self.finish(reset_current=True)

    def start(self, mark_as_current: bool = False) -> None:
        if mark_as_current:
            self._prev_context_token = scope_set_current_trace(self)

    def finish(self, reset_current: bool = False) -> None:
        if reset_current and self._prev_context_token is not None:
            scope_reset_current_trace(self._prev_context_token)
            self._prev_context_token = None

    def set_first_input(self, first_input: Mapping[str, Any]) -> None:
        del first_input

    def set_last_output(self, last_output: Mapping[str, Any]) -> None:
        del last_output

    @property
    def trace_id(self) -> str:
        return "no-op"

    @property
    def name(self) -> str:
        return "no-op"

    @property
    def started_at(self) -> str | None:
        return None

    @property
    def ended_at(self) -> str | None:
        return None

    @property
    def first_input(self) -> Mapping[str, Any] | None:
        return None

    @property
    def last_output(self) -> Mapping[str, Any] | None:
        return None

    @property
    def metadata(self) -> dict[str, Any] | None:
        return None

    def export(self) -> dict[str, Any] | None:
        return None


class TraceImpl(Trace):
    """Concrete trace implementation."""

    def __init__(
        self,
        name: str,
        trace_id: str,
        group_id: str | None,
        metadata: dict[str, Any] | None,
        processor: TracingProcessor,
    ) -> None:
        self._name = name
        self._trace_id = trace_id
        self._group_id = group_id
        self._metadata = metadata or {}
        self._processor = processor
        self._started_at: str | None = None
        self._ended_at: str | None = None
        self._started = False
        self._prev_context_token: contextvars.Token[Trace | None] | None = None
        self._first_input: Mapping[str, Any] | None = None
        self._last_output: Mapping[str, Any] | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def started_at(self) -> str | None:
        return self._started_at

    @property
    def ended_at(self) -> str | None:
        return self._ended_at

    @property
    def first_input(self) -> Mapping[str, Any] | None:
        return self._first_input

    @property
    def last_output(self) -> Mapping[str, Any] | None:
        return self._last_output

    def set_first_input(self, first_input: Mapping[str, Any]) -> None:
        self._first_input = first_input

    def set_last_output(self, last_output: Mapping[str, Any]) -> None:
        self._last_output = last_output

    def __enter__(self) -> Self:
        if self._started:
            if self._prev_context_token is None:
                LOGGER.error("Trace already started but no context token set")
            return self

        self._started = True
        self.start(mark_as_current=True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        del exc_type, exc_val, exc_tb
        self.finish(reset_current=True)

    def start(self, mark_as_current: bool = False) -> None:
        if self._started_at is not None:
            LOGGER.warning("Trace already started")
            return

        self._started_at = get_trace_provider().time_iso()
        self._processor.on_trace_start(self)

        if mark_as_current:
            self._prev_context_token = scope_set_current_trace(self)

    def finish(self, reset_current: bool = False) -> None:
        if self._ended_at is not None:
            LOGGER.warning("Trace already finished")
            return

        self._ended_at = get_trace_provider().time_iso()
        self._processor.on_trace_end(self)

        if reset_current and self._prev_context_token is not None:
            scope_reset_current_trace(self._prev_context_token)
            self._prev_context_token = None

    @property
    def metadata(self) -> dict[str, Any] | None:
        return self._metadata

    def export(self) -> dict[str, Any] | None:
        data: dict[str, Any] = {
            "object": "trace",
            "id": self.trace_id,
            "name": self.name,
            "started_at": self._started_at,
            "ended_at": self._ended_at,
        }

        if self._group_id:
            data["group_id"] = self._group_id

        if self._first_input:
            data["first_input"] = self._first_input

        if self._last_output:
            data["last_output"] = self._last_output

        if self._metadata:
            data["metadata"] = self._metadata

        return data
