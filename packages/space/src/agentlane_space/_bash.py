"""Run the first-party bash tool through a SPACE command session."""

from pathlib import Path
from uuid import uuid4

from agentlane.harness.tools import (
    BASH_MAX_BYTES,
    BASH_MAX_LINES,
    BashExecutionRequest,
    BashExecutionResult,
    TruncatedOutput,
    truncate_output,
)
from agentlane.runtime import CancellationToken

from ._client import SpaceSandbox
from ._errors import SpaceError, SpaceErrorKind

_MAX_COMMAND_BYTES = 48 * 1024
CAPTURE_ERROR = b"agentlane: failed to capture bash output\n"
# The wrapper does not create an extra process group; SPACE can terminate its
# foreground pipeline. User commands can still detach children. Command stderr
# goes to stdout; only capture errors use the wrapper's stderr.
# Exclusive creation rejects existing log paths. PIPESTATUS preserves
# the command's exit code while still treating a failed tee as capture failure.
CAPTURE_SCRIPT = """\
if ! (umask 077; set -o noclobber; : > "$1") 2>/dev/null; then
    printf 'agentlane: failed to capture bash output\\n' >&2
    exit 125
fi
/bin/bash -lc "$2" 2>&1 | command tee -- "$1"
statuses=("${PIPESTATUS[@]}")
if (( statuses[1] != 0 )); then
    printf 'agentlane: failed to capture bash output\\n' >&2
    exit 125
fi
exit "${statuses[0]}"
"""


class SpaceBashExecutor:
    """Execute bash in a checked SPACE sandbox.

    Each command writes a private log in the guest's `/tmp` directory. The
    application owns log retention. Truncated results include the guest log
    path. Environment entries override the guest environment.

    Args:
        sandbox: Checked sandbox that owns native command sessions.
    """

    def __init__(self, sandbox: SpaceSandbox) -> None:
        self._sandbox = sandbox

    async def run(
        self,
        request: BashExecutionRequest,
        cancellation_token: CancellationToken,
    ) -> BashExecutionResult:
        """Run a command and retain its complete output in the guest."""
        if len(request.command.encode("utf-8")) > _MAX_COMMAND_BYTES:
            raise SpaceError(
                "SPACE bash commands must not exceed 48 KiB of UTF-8.",
                kind=SpaceErrorKind.INPUT_LIMIT,
            )

        cwd = str(request.cwd)
        self._sandbox.require_checked(cwd)
        timeout = (
            request.timeout_seconds
            if request.timeout_seconds is not None
            else self._sandbox.default_timeout
        )
        # This is a guest path, created exclusively with mode 0600 above.
        log_path = Path(f"/tmp/agentlane-bash-{uuid4().hex}.log")  # noqa: S108
        result = await self._sandbox.execute(
            (
                "/bin/bash",
                "-c",
                CAPTURE_SCRIPT,
                "agentlane-bash",
                str(log_path),
                request.command,
            ),
            cwd=cwd,
            cancellation_token=cancellation_token,
            timeout=timeout,
            env=request.env,
            # Keep enough extra bytes to detect a full buffer and decode a
            # UTF-8 sequence that crosses the final visible output boundary.
            output_limit=BASH_MAX_BYTES + 4,
            tail=True,
            require_checked=True,
        )
        # Exit code 125 alone is not a capture failure: the user's command can
        # return it. Only the wrapper's dedicated stderr marker identifies one.
        if CAPTURE_ERROR in result.stderr:
            raise SpaceError(
                "Failed to capture SPACE bash output.",
                kind=SpaceErrorKind.PROTOCOL,
            )

        output = truncate_output(
            result.combined.decode("utf-8", errors="replace"),
            max_lines=BASH_MAX_LINES,
            max_bytes=BASH_MAX_BYTES,
            tail=True,
            include_marker=False,
        )
        if result.truncated and not output.truncated:
            # The transport may already have dropped bytes before the visible
            # output fits these limits. Keep the full-log reference in that case.
            output = TruncatedOutput(text=output.text, truncated=True)

        return BashExecutionResult(
            command=request.command,
            cwd=request.cwd,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            cancelled=False,
            timeout_seconds=timeout,
            output=output,
            full_output_path=log_path if output.truncated else None,
        )
