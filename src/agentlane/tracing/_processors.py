"""Concrete processor implementations for trace and span handling."""

import threading
from typing import Any

import structlog

from ._metrics import remove_collector
from ._processor_interface import TracingProcessor

LOGGER = structlog.get_logger(log_tag="tracing.processors")


class SynchronousMultiTracingProcessor(TracingProcessor):
    """Forwards all calls to a list of TracingProcessors in order."""

    def __init__(self) -> None:
        """Initialize the multi-processor with thread safety."""
        # Using tuple for thread-safe iteration
        self._processors: tuple[TracingProcessor, ...] = ()
        self._lock = threading.Lock()

    def add_processor(self, processor: TracingProcessor) -> None:
        """Add a processor to receive all traces and spans.

        Args:
            processor: The processor to add.
        """
        with self._lock:
            # Skip if the exact same instance is already registered
            if any(processor is p for p in self._processors):
                LOGGER.warning(
                    f"Processor {processor.__class__.__name__} already registered; skipping"
                )
                return
            self._processors += (processor,)

    def set_processors(self, processors: list[TracingProcessor]) -> None:
        """Replace all processors with the given list.

        Args:
            processors: New list of processors.
        """
        with self._lock:
            # De-duplicate processors by identity while preserving order
            seen: set[int] = set()
            unique_processors: list[TracingProcessor] = []
            for p in processors:
                pid = id(p)
                if pid not in seen:
                    seen.add(pid)
                    unique_processors.append(p)
            self._processors = tuple(unique_processors)

    def on_trace_start(self, trace: Any) -> None:
        """Forward trace start to all processors."""
        for processor in self._processors:
            try:
                processor.on_trace_start(trace)
            except Exception as e:
                LOGGER.error(
                    f"Error in processor {processor.__class__.__name__} "
                    f"during on_trace_start: {e}"
                )

    def on_trace_end(self, trace: Any) -> None:
        """Forward trace end to all processors."""
        for processor in self._processors:
            try:
                processor.on_trace_end(trace)
            except Exception as e:
                LOGGER.error(
                    f"Error in processor {processor.__class__.__name__} "
                    f"during on_trace_end: {e}"
                )

        # Clean up collector after all processors have accessed it
        if trace.trace_id != "no-op":
            remove_collector(trace.trace_id)

    def on_span_start(self, span: Any) -> None:
        """Forward span start to all processors."""
        for processor in self._processors:
            try:
                processor.on_span_start(span)
            except Exception as e:
                LOGGER.error(
                    f"Error in processor {processor.__class__.__name__} "
                    f"during on_span_start: {e}"
                )

    def on_span_end(self, span: Any) -> None:
        """Forward span end to all processors."""
        for processor in self._processors:
            try:
                processor.on_span_end(span)
            except Exception as e:
                LOGGER.error(
                    f"Error in processor {processor.__class__.__name__} "
                    f"during on_span_end: {e}"
                )

    def shutdown(self) -> None:
        """Shutdown all processors."""
        for processor in self._processors:
            LOGGER.debug(f"Shutting down processor {processor.__class__.__name__}")
            try:
                processor.shutdown()
            except Exception as e:
                LOGGER.error(
                    f"Error shutting down processor {processor.__class__.__name__}: {e}"
                )

    def force_flush(self) -> None:
        """Force flush all processors."""
        for processor in self._processors:
            try:
                processor.force_flush()
            except Exception as e:
                LOGGER.error(
                    f"Error flushing processor {processor.__class__.__name__}: {e}"
                )
