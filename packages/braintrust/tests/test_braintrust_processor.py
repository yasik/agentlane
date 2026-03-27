"""Tests for Braintrust processor environment propagation."""

from typing import Any
from unittest.mock import MagicMock

from agentlane_braintrust import BraintrustProcessor


def _make_mock_trace(
    trace_id: str = "trace-1",
    name: str = "test-trace",
    first_input: dict[str, Any] | None = None,
    last_output: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
) -> Any:
    """Create a mock trace-like object for processor unit tests."""
    mock_trace: Any = MagicMock()
    mock_trace.trace_id = trace_id
    mock_trace.name = name
    mock_trace.first_input = first_input
    mock_trace.last_output = last_output
    mock_trace.metadata = metadata
    mock_trace.started_at = started_at
    mock_trace.ended_at = ended_at
    return mock_trace


class TestBraintrustProcessorEnvironment:
    """Tests for environment tagging in BraintrustProcessor."""

    def _create_processor(
        self, environment: str | None = None
    ) -> tuple[BraintrustProcessor, Any]:
        """Create a processor with a mock logger."""
        mock_logger: Any = MagicMock()
        mock_span: Any = MagicMock()
        mock_logger.start_span.return_value = mock_span
        processor = BraintrustProcessor(logger=mock_logger, environment=environment)
        return processor, mock_span

    def test_environment_none_no_tags_no_metadata(self) -> None:
        """No environment should produce no environment tags."""
        processor, mock_span = self._create_processor(environment=None)
        trace = _make_mock_trace(
            first_input={"question": "hello"},
            last_output={"answer": "world"},
        )

        processor.on_trace_start(trace)
        processor.on_trace_end(trace)

        assert "tags" not in mock_span.log.call_args.kwargs

    def test_environment_set_adds_tags_and_metadata(self) -> None:
        """An environment should be added as both tag and metadata."""
        processor, mock_span = self._create_processor(environment="production")
        trace = _make_mock_trace(
            first_input={"question": "hello"},
            last_output={"answer": "world"},
        )

        processor.on_trace_start(trace)
        processor.on_trace_end(trace)

        log_data = mock_span.log.call_args.kwargs
        assert log_data["tags"] == ["production"]
        assert log_data["metadata"]["environment"] == "production"

    def test_environment_merges_with_existing_metadata(self) -> None:
        """Existing metadata should be preserved when environment is added."""
        processor, mock_span = self._create_processor(environment="staging")
        trace = _make_mock_trace(
            first_input={"question": "hello"},
            last_output={"answer": "world"},
            metadata={"user_id": "u-123", "request_id": "r-456"},
        )

        processor.on_trace_start(trace)
        processor.on_trace_end(trace)

        log_data = mock_span.log.call_args.kwargs
        assert log_data["tags"] == ["staging"]
        assert log_data["metadata"] == {
            "user_id": "u-123",
            "request_id": "r-456",
            "environment": "staging",
        }

    def test_environment_works_when_trace_has_no_metadata(self) -> None:
        """Environment tagging should work with empty trace metadata."""
        processor, mock_span = self._create_processor(environment="dev")
        trace = _make_mock_trace(
            first_input={"question": "hello"},
            last_output={"answer": "world"},
            metadata=None,
        )

        processor.on_trace_start(trace)
        processor.on_trace_end(trace)

        log_data = mock_span.log.call_args.kwargs
        assert log_data["tags"] == ["dev"]
        assert log_data["metadata"]["environment"] == "dev"

    def test_environment_works_with_empty_trace(self) -> None:
        """Environment tagging should work even with empty trace I/O."""
        processor, mock_span = self._create_processor(environment="production")
        trace = _make_mock_trace(first_input=None, last_output=None, metadata=None)

        processor.on_trace_start(trace)
        processor.on_trace_end(trace)

        log_data = mock_span.log.call_args.kwargs
        assert log_data["tags"] == ["production"]
        assert log_data["metadata"]["environment"] == "production"

    def test_start_span_called_with_propagated_event_when_environment_set(
        self,
    ) -> None:
        """Root span creation should propagate environment metadata to children."""
        mock_logger: Any = MagicMock()
        mock_span: Any = MagicMock()
        mock_logger.start_span.return_value = mock_span
        processor = BraintrustProcessor(logger=mock_logger, environment="production")

        trace = _make_mock_trace()
        processor.on_trace_start(trace)

        call_kwargs = mock_logger.start_span.call_args.kwargs
        assert call_kwargs["propagated_event"] == {
            "tags": ["production"],
            "metadata": {"environment": "production"},
        }
        assert call_kwargs["tags"] == ["production"]

    def test_start_span_no_propagated_event_when_no_environment_no_metadata(
        self,
    ) -> None:
        """Root spans should omit propagated_event when there is nothing to propagate."""
        mock_logger: Any = MagicMock()
        mock_span: Any = MagicMock()
        mock_logger.start_span.return_value = mock_span
        processor = BraintrustProcessor(logger=mock_logger, environment=None)

        trace = _make_mock_trace()
        processor.on_trace_start(trace)

        call_kwargs = mock_logger.start_span.call_args.kwargs
        assert call_kwargs["propagated_event"] is None
        assert "tags" not in call_kwargs

    def test_start_span_propagates_trace_metadata(self) -> None:
        """Trace metadata should be copied into propagated_event."""
        mock_logger: Any = MagicMock()
        mock_span: Any = MagicMock()
        mock_logger.start_span.return_value = mock_span
        processor = BraintrustProcessor(logger=mock_logger, environment=None)

        trace = _make_mock_trace(metadata={"report_id": "r-789"})
        processor.on_trace_start(trace)

        call_kwargs = mock_logger.start_span.call_args.kwargs
        assert call_kwargs["propagated_event"] == {
            "metadata": {"report_id": "r-789"},
        }
        assert call_kwargs["metadata"] == {"report_id": "r-789"}

    def test_start_span_merges_environment_and_trace_metadata(self) -> None:
        """Environment and trace metadata should be merged for propagation."""
        mock_logger: Any = MagicMock()
        mock_span: Any = MagicMock()
        mock_logger.start_span.return_value = mock_span
        processor = BraintrustProcessor(logger=mock_logger, environment="staging")

        trace = _make_mock_trace(metadata={"report_id": "r-789", "user_id": "u-1"})
        processor.on_trace_start(trace)

        call_kwargs = mock_logger.start_span.call_args.kwargs
        assert call_kwargs["propagated_event"] == {
            "tags": ["staging"],
            "metadata": {
                "report_id": "r-789",
                "user_id": "u-1",
                "environment": "staging",
            },
        }
        assert call_kwargs["metadata"] == {"report_id": "r-789", "user_id": "u-1"}
        assert call_kwargs["tags"] == ["staging"]

    def test_child_span_inherits_environment_via_propagated_event(self) -> None:
        """Child spans should inherit root propagated state from Braintrust."""
        mock_logger: Any = MagicMock()
        mock_root_span: Any = MagicMock()
        mock_child_span: Any = MagicMock()
        mock_logger.start_span.return_value = mock_root_span
        mock_root_span.start_span.return_value = mock_child_span

        processor = BraintrustProcessor(logger=mock_logger, environment="staging")
        trace = _make_mock_trace()
        processor.on_trace_start(trace)

        child_span: Any = MagicMock()
        child_span.span_id = "child-1"
        child_span.parent_id = None
        child_span.trace_id = "trace-1"
        child_span.span_data = MagicMock()
        child_span.span_data.type = "task"
        child_span.span_data.name = "child-task"
        child_span.started_at = None

        processor.on_span_start(child_span)

        assert mock_logger.start_span.call_args.kwargs["propagated_event"] == {
            "tags": ["staging"],
            "metadata": {"environment": "staging"},
        }
        mock_root_span.start_span.assert_called_once()

    def test_environment_does_not_mutate_trace_metadata(self) -> None:
        """Injecting environment should not mutate the original trace metadata dict."""
        processor, _ = self._create_processor(environment="production")
        original_metadata = {"user_id": "u-123"}
        trace = _make_mock_trace(
            first_input={"question": "hello"},
            last_output={"answer": "world"},
            metadata=original_metadata,
        )

        processor.on_trace_start(trace)
        processor.on_trace_end(trace)

        assert original_metadata == {"user_id": "u-123"}
