import datetime
from typing import Any

import braintrust
import structlog
from braintrust.logger import NOOP_SPAN

from agentlane.tracing import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    Span,
    Trace,
    TracingProcessor,
    get_collector,
    peek_collector,
)

LOGGER = structlog.get_logger(log_tag="tracing.braintrust_processor")


def _span_type(span: Span[Any]) -> braintrust.SpanTypeAttribute:
    """Map our span types to Braintrust span types."""
    if span.span_data.type in ["task", "agent", "custom"]:
        return braintrust.SpanTypeAttribute.TASK
    if span.span_data.type in ["function"]:
        return braintrust.SpanTypeAttribute.TOOL
    if span.span_data.type in ["generation"]:
        return braintrust.SpanTypeAttribute.LLM
    return braintrust.SpanTypeAttribute.TASK


def _span_name(span: Span[Any]) -> str:
    """Get the display name for a span."""
    if isinstance(span.span_data, (AgentSpanData, FunctionSpanData, CustomSpanData)):
        return span.span_data.name
    if isinstance(span.span_data, GenerationSpanData):
        return "Generation"
    return "Unknown"


def _timestamp_from_maybe_iso(timestamp: str | None = None) -> float | None:
    """Convert ISO timestamp string to float timestamp."""
    if timestamp is None:
        return None
    return datetime.datetime.fromisoformat(timestamp).timestamp()


def _maybe_timestamp_elapsed(
    end: str | None = None, start: str | None = None
) -> float | None:
    """Calculate elapsed time between two ISO timestamps."""
    if start is None or end is None:
        return None
    return (
        datetime.datetime.fromisoformat(end) - datetime.datetime.fromisoformat(start)
    ).total_seconds()


class BraintrustProcessor(TracingProcessor):
    """Braintrust processor using native SDK for rich integration.

    This processor logs traces and spans directly to Braintrust using
    the native SDK, providing full feature integration including experiments,
    scoring, and rich span metadata.

    Args:
        api_key: Braintrust API key
        project_id: Braintrust project ID
        logger: Optional pre-configured Braintrust logger/experiment
    """

    def __init__(
        self,
        api_key: str | None = None,
        project_id: str | None = None,
        logger: (
            braintrust.Span | braintrust.Experiment | braintrust.Logger | None
        ) = None,
        environment: str | None = None,
    ):
        """Initialize the Braintrust processor.

        Args:
            api_key: Braintrust API key
            project_id: Braintrust project ID to log to
            logger: Optional logger/experiment to use. If not provided,
                will create one. If provided, api_key and project_id are
                ignored.
            environment: Environment tag for traces (e.g. production, staging, dev).
        """
        # Initialize logger if not provided
        if logger is None:
            if api_key is None or project_id is None:
                raise ValueError(
                    "api_key and project_id are required when no logger is provided"
                )

            self._logger = braintrust.init_logger(
                api_key=api_key,
                project_id=project_id,
                set_current=False,
                force_login=True,
            )
        else:
            self._logger = logger

        self._environment = environment

        # Track active spans
        self._spans: dict[str, braintrust.Span] = {}

        # Track first input and last output for trace summary
        self._first_input: Any | None = None
        self._last_output: Any | None = None

        # Storage for custom metrics from MetricsProcessor
        self._pending_custom_metrics: dict[str, dict[str, Any]] = {}

    def _build_propagated_event(
        self, trace_metadata: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Build propagated_event for environment + metadata propagation to child spans."""
        metadata: dict[str, Any] = {}
        tags: list[str] = []

        if trace_metadata:
            metadata.update(trace_metadata)

        if self._environment is not None:
            tags.append(self._environment)
            metadata["environment"] = self._environment

        if not metadata and not tags:
            return None

        event: dict[str, Any] = {}
        if tags:
            event["tags"] = tags
        if metadata:
            event["metadata"] = metadata
        return event

    def on_trace_start(self, trace: Trace) -> None:
        """Handle trace start by creating a root Braintrust span."""
        current_context = braintrust.current_span()
        propagated = self._build_propagated_event(trace.metadata)
        env_tags: dict[str, Any] = (
            {"tags": [self._environment]} if self._environment is not None else {}
        )

        if current_context != NOOP_SPAN:
            # If there's an existing Braintrust context, nest under it
            self._spans[trace.trace_id] = current_context.start_span(
                name=trace.name,
                span_attributes={"type": "task", "name": trace.name},
                propagated_event=propagated,
                metadata=trace.metadata,
                **env_tags,
            )
        else:
            self._spans[trace.trace_id] = self._logger.start_span(
                name=trace.name,
                span_attributes={"type": "task", "name": trace.name},
                span_id=trace.trace_id,
                root_span_id=trace.trace_id,
                start_time=_timestamp_from_maybe_iso(trace.started_at),
                propagated_event=propagated,
                metadata=trace.metadata,
                **env_tags,
            )

    def set_trace_metrics(
        self,
        trace_id: str,
        metrics: dict[str, Any],
    ) -> None:
        """Set custom metrics to include when trace ends.

        Called by MetricsProcessor callback to provide aggregated metrics.

        Args:
            trace_id: The trace ID.
            metrics: Dictionary of metric name -> value.
        """
        self._pending_custom_metrics[trace_id] = metrics

    def on_trace_end(self, trace: Trace) -> None:
        """Handle trace end by closing the root Braintrust span."""
        if trace.trace_id not in self._spans:
            return

        span = self._spans.pop(trace.trace_id)

        # Build log data
        if trace.first_input is not None or trace.last_output is not None:
            log_data = self._log_trace_data(trace)
        else:
            log_data = {"input": self._first_input, "output": self._last_output}

        # Get aggregated metrics directly from collector
        # This avoids processor order dependency (MetricsProcessor may not have run yet)
        self._add_trace_metrics(log_data, trace)

        # Include custom metrics if available (backward compatibility)
        if trace.trace_id in self._pending_custom_metrics:
            custom_metrics = self._pending_custom_metrics.pop(trace.trace_id)
            existing_metrics = log_data.get("metrics")
            if existing_metrics is not None:
                existing_metrics.update(custom_metrics)
            else:
                log_data["metrics"] = custom_metrics

        # Inject environment tag and metadata (copy to avoid mutating trace)
        if self._environment is not None:
            log_data["tags"] = [self._environment]
            metadata = dict(log_data.get("metadata") or {})
            metadata["environment"] = self._environment
            log_data["metadata"] = metadata

        # Log the data
        span.log(**log_data)

        # End the span
        span.end(_timestamp_from_maybe_iso(trace.ended_at))

        # Reset accumulators
        self._first_input = None
        self._last_output = None

    def _add_trace_metrics(self, log_data: dict[str, Any], trace: Trace) -> None:
        """Add aggregated trace-level metrics to log data.

        Args:
            log_data: The log data dictionary to update.
            trace: The trace being logged.
        """
        trace_id = trace.trace_id
        if trace_id == "no-op":
            return

        try:
            collector = peek_collector(trace_id)
            if collector is None:
                return

            aggregated = collector.aggregate_as_dict()
            if not aggregated:
                return

            existing = log_data.get("metrics")
            if existing is not None:
                existing.update(aggregated)
            else:
                log_data["metrics"] = aggregated
        except Exception:  # noqa: BLE001
            LOGGER.debug("Failed to retrieve trace metrics", exc_info=True)

    def _log_trace_data(self, trace: Trace) -> dict[str, Any]:
        """Extract log data from a trace."""
        if trace.first_input is None and trace.last_output is None:
            return {}

        return {
            "input": trace.first_input,
            "output": trace.last_output,
            "metadata": trace.metadata,
        }

    def _agent_log_data(self, span: Span[AgentSpanData]) -> dict[str, Any]:
        """Extract log data from an agent span."""
        return {
            "metadata": {
                "tools": span.span_data.tools,
                "handoffs": span.span_data.handoffs,
                "output_type": span.span_data.output_type,
            }
        }

    def _function_log_data(self, span: Span[FunctionSpanData]) -> dict[str, Any]:
        """Extract log data from a function span."""
        return {
            "input": span.span_data.input,
            "output": span.span_data.output,
        }

    def _generation_log_data(self, span: Span[GenerationSpanData]) -> dict[str, Any]:
        """Extract log data from a generation span."""
        metrics = {}
        ttft = _maybe_timestamp_elapsed(span.ended_at, span.started_at)

        if ttft is not None:
            metrics["time_to_first_token"] = ttft

        usage = span.span_data.usage or {}

        # Handle different token count field names
        if "prompt_tokens" in usage:
            metrics["prompt_tokens"] = usage["prompt_tokens"]
        elif "input_tokens" in usage:
            metrics["prompt_tokens"] = usage["input_tokens"]

        if "completion_tokens" in usage:
            metrics["completion_tokens"] = usage["completion_tokens"]
        elif "output_tokens" in usage:
            metrics["completion_tokens"] = usage["output_tokens"]

        if "total_tokens" in usage:
            metrics["tokens"] = usage["total_tokens"]
        elif "input_tokens" in usage and "output_tokens" in usage:
            metrics["tokens"] = usage["input_tokens"] + usage["output_tokens"]

        if "total_cost" in usage:
            metrics["total_cost"] = usage["total_cost"]

        # Extract additional metrics (retry, cache) by direct key copy
        for key in (
            "attempts",
            "retry_wait",
            "backend_latency",
            "cached_tokens",
            "cache_hit_rate_pct",
        ):
            if key in usage:
                metrics[key] = usage[key]

        return {
            "input": span.span_data.input,
            "output": span.span_data.output,
            "metadata": {
                "model": span.span_data.model,
                "model_config": span.span_data.model_config,
            },
            "metrics": metrics,
        }

    def _custom_log_data(self, span: Span[CustomSpanData]) -> dict[str, Any]:
        """Extract log data from a custom span."""
        return span.span_data.data or {}

    def _log_data(self, span: Span[Any]) -> dict[str, Any]:
        """Route span to appropriate data extraction method."""
        if isinstance(span.span_data, AgentSpanData):
            return self._agent_log_data(span)
        if isinstance(span.span_data, FunctionSpanData):
            return self._function_log_data(span)
        if isinstance(span.span_data, GenerationSpanData):
            return self._generation_log_data(span)
        if isinstance(span.span_data, CustomSpanData):
            return self._custom_log_data(span)
        return {}

    def on_span_start(self, span: Span[Any]) -> None:
        """Handle span start by creating a nested Braintrust span."""
        # Find parent span
        if span.parent_id is not None and span.parent_id in self._spans:
            parent = self._spans[span.parent_id]
        elif span.trace_id in self._spans:
            parent = self._spans[span.trace_id]
        else:
            # No parent found, skip
            LOGGER.warning(f"No parent found for span {span.span_id}")
            return

        # Create nested span
        created_span = parent.start_span(
            id=span.span_id,
            name=_span_name(span),
            type=_span_type(span),
            start_time=_timestamp_from_maybe_iso(span.started_at),
        )

        self._spans[span.span_id] = created_span

        # Set as current for nested operations
        created_span.set_current()

    def _add_span_metrics(self, event: dict[str, Any], span: Span[Any]) -> None:
        """Add custom metrics emitted by this span to the event."""
        trace_id = span.trace_id
        if trace_id == "no-op":
            return

        try:
            collector = get_collector(trace_id)
            records = collector.get_records_for_span(span.span_id)
            if not records:
                return

            # Convert records to simple dict
            span_metrics = {r.name: r.value for r in records}

            # Merge with existing metrics (if any from _generation_log_data)
            existing = event.get("metrics")
            if existing is not None:
                existing.update(span_metrics)
            else:
                event["metrics"] = span_metrics
        except Exception:  # noqa: BLE001
            # Don't fail span logging if metrics retrieval fails
            LOGGER.debug("Failed to retrieve span metrics", exc_info=True)

    def on_span_end(self, span: Span[Any]) -> None:
        """Handle span end by logging data and closing the Braintrust span."""
        if span.span_id not in self._spans:
            return

        bt_span = self._spans.pop(span.span_id)

        # Prepare event data
        event = {"error": span.error, **self._log_data(span)}

        # Include span-level custom metrics if available
        self._add_span_metrics(event, span)

        # Log the event
        bt_span.log(**event)

        # Unset as current
        bt_span.unset_current()

        # End the span
        bt_span.end(_timestamp_from_maybe_iso(span.ended_at))

        # Track first input and last output for trace summary
        input_ = event.get("input")
        output = event.get("output")

        if self._first_input is None and input_ is not None:
            self._first_input = input_

        if output is not None:
            self._last_output = output

    def shutdown(self) -> None:
        """Shutdown the processor and flush any pending data."""
        self._logger.flush()

    def force_flush(self) -> None:
        """Force flush any buffered data to Braintrust."""
        self._logger.flush()
