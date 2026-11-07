"""Model-facing output for tools that execute native processes."""

from agentlane.models import ToolError, ToolErrorKind, ToolFailure

from ._output import PROCESS_MAX_BYTES, PROCESS_MAX_LINES
from ._process_executor import ProcessExecutionResult, ProcessStopReason


def format_process_output(result: ProcessExecutionResult) -> str | ToolFailure:
    """Render a native result without mixing display text into captured bytes."""
    output = result.output.text.rstrip("\n") or "(no output)"
    notices: list[str] = []

    # Preview truncation and log completeness describe separate output boundaries.
    if result.output.truncated and result.log_path is not None:
        label = "Full output" if result.log_complete else "Partial output"
        notices.append(
            "Showing last "
            f"{PROCESS_MAX_LINES} lines or {PROCESS_MAX_BYTES} bytes. "
            f"{label}: {result.log_path}"
        )

    error = _process_failure(result)
    if result.stop_reason == ProcessStopReason.TIMEOUT:
        if result.timeout_seconds is None:
            notices.append("Command timed out")
        else:
            notices.append(
                f"Command timed out after {_format_seconds(result.timeout_seconds)} seconds"
            )
    elif error is not None:
        notices.append(error.message)

    text = (
        output
        if not notices
        else output + "\n\n" + "\n".join(f"[{notice}]" for notice in notices)
    )
    if error is None:
        return text

    return ToolFailure(text=text, error=error)


def _process_failure(result: ProcessExecutionResult) -> ToolError | None:
    if result.stop_reason == ProcessStopReason.TIMEOUT:
        return ToolError(message="Command timed out", kind=ToolErrorKind.TIMEOUT.value)
    if result.stop_reason == ProcessStopReason.CANCELLED:
        return ToolError(
            message="Command cancelled", kind=ToolErrorKind.CANCELLED.value
        )
    if result.stop_reason == ProcessStopReason.STDOUT_LIMIT:
        return ToolError(
            message="Command exceeded stdout capture limit",
            kind=ToolErrorKind.ERROR.value,
        )
    if result.exit_code != 0:
        return ToolError(
            message=f"Command exited with code {result.exit_code}",
            kind=ToolErrorKind.NONZERO_EXIT.value,
        )

    return None


def _format_seconds(seconds: float) -> str:
    if seconds.is_integer():
        return str(int(seconds))

    return str(seconds)
