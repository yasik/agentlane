"""Harness run-stream handle for live agent execution."""

from collections.abc import Callable

from agentlane.models import ModelStreamEvent

from ._stream_base import BaseRunStream

_STREAM_END = object()


class RunStream(BaseRunStream[ModelStreamEvent]):
    """Async stream handle for one harness run.

    `RunStream` exposes the live per-event model stream while keeping the final
    harness `RunResult` available separately via `result()`.
    """

    def __init__(
        self,
        *,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """Initialize one run stream handle."""
        super().__init__(end_sentinel=_STREAM_END, on_close=on_close)
