"""Shared model-facing output for tools that run commands."""

from agentlane.models import ToolError, ToolErrorKind, ToolFailure

from ._bash_executor import BashExecutionResult
from ._output import BASH_MAX_BYTES, BASH_MAX_LINES


def format_command_output(result: BashExecutionResult) -> str | ToolFailure:
    """Render command output and notices, with typed failure metadata.

    `ToolFailure` keeps the same model-facing text as a plain string while
    exposing timeout, cancellation, and exit-code failures to the runner.
    """
    output = result.output.text.rstrip("\n") or "(no output)"
    notices: list[str] = []

    if result.output_truncated and result.full_output_path is not None:
        notices.append(
            "Showing last "
            f"{BASH_MAX_LINES} lines or {BASH_MAX_BYTES} bytes. "
            f"Full output: {result.full_output_path}"
        )

    error = _command_failure(result)
    if result.timed_out:
        if result.timeout_seconds is None:
            notices.append("Command timed out")
        else:
            notices.append(
                f"Command timed out after {_format_seconds(result.timeout_seconds)} seconds"
            )
    elif result.cancelled:
        notices.append("Command cancelled")
    elif result.exit_code is not None and result.exit_code != 0:
        notices.append(f"Command exited with code {result.exit_code}")

    text = (
        output
        if not notices
        else output + "\n\n" + "\n".join(f"[{notice}]" for notice in notices)
    )
    if error is None:
        return text
    return ToolFailure(text=text, error=error)


def _command_failure(result: BashExecutionResult) -> ToolError | None:
    """Map a command execution result to a typed failure, or None on success."""
    if result.timed_out:
        return ToolError(message="Command timed out", kind=ToolErrorKind.TIMEOUT.value)
    if result.cancelled:
        return ToolError(
            message="Command cancelled", kind=ToolErrorKind.CANCELLED.value
        )
    if result.exit_code is not None and result.exit_code != 0:
        return ToolError(
            message=f"Command exited with code {result.exit_code}",
            kind=ToolErrorKind.NONZERO_EXIT.value,
        )
    return None


def _format_seconds(seconds: float) -> str:
    if seconds.is_integer():
        return str(int(seconds))
    return str(seconds)
