from typing import Any

from ... import tracing


class TraceCtxManager:
    """Creates a trace only if there is no current trace, and
    manages the trace lifecycle."""

    def __init__(
        self,
        workflow_name: str,
        trace_id: str | None = None,
        group_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        disabled: bool = False,
    ):
        self.trace: tracing.Trace | None = None
        self.workflow_name = workflow_name
        self.trace_id = trace_id
        self.group_id = group_id
        self.metadata = metadata
        self.disabled = disabled

    def __enter__(self) -> tracing.Trace:
        current_trace = tracing.get_current_trace()
        if not current_trace:
            self.trace = tracing.trace(
                workflow_name=self.workflow_name,
                trace_id=self.trace_id,
                group_id=self.group_id,
                metadata=self.metadata,
                disabled=self.disabled,
            )
            self.trace.start(mark_as_current=True)
            return self.trace

        return current_trace

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        if self.trace:
            self.trace.finish(reset_current=True)
