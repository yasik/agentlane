"""Local bash command execution primitives for harness tools."""

import asyncio
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Protocol

from agentlane.runtime import CancellationToken

from ._output import TruncatedOutput
from ._process_executor import (
    ProcessExecutionRequest,
    ProcessStopReason,
    run_local_process,
)


@dataclass(frozen=True, slots=True)
class BashShellConfig:
    """Resolved bash-compatible shell invocation."""

    executable: str
    args: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BashExecutionRequest:
    """Executor-facing request for one bash command."""

    command: str
    cwd: Path
    timeout_seconds: float | None = None
    # Executor-only override for host callers; no LLM-side path sets it today.
    env: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class BashExecutionResult:
    """Captured result for one bash command."""

    command: str
    cwd: Path
    exit_code: int | None
    timed_out: bool
    cancelled: bool
    timeout_seconds: float | None
    output: TruncatedOutput
    full_output_path: Path | None

    @property
    def output_truncated(self) -> bool:
        """Return whether visible output was truncated."""
        return self.output.truncated


class BashExecutor(Protocol):
    """Protocol for reusable bash command executors."""

    async def run(
        self,
        request: BashExecutionRequest,
        cancellation_token: CancellationToken,
    ) -> BashExecutionResult:
        """Execute one bash command."""
        ...


class LocalBashExecutor:
    """Adapt the accepted Bash command API to local native process execution."""

    def __init__(
        self,
        *,
        default_timeout: float | None = None,
        shell_config: BashShellConfig | None = None,
    ) -> None:
        self._default_timeout = default_timeout
        self._shell_config = shell_config

    async def run(
        self,
        request: BashExecutionRequest,
        cancellation_token: CancellationToken,
    ) -> BashExecutionResult:
        """Run the configured shell while retaining the existing Bash result API."""
        effective_timeout = (
            request.timeout_seconds
            if request.timeout_seconds is not None
            else self._default_timeout
        )
        if cancellation_token.is_cancelled:
            return _cancelled_result(request, timeout_seconds=effective_timeout)

        # Shell discovery and invocation belong only to this Bash adapter.
        shell = self._shell_config or resolve_bash_shell()
        try:
            result = await run_local_process(
                ProcessExecutionRequest(
                    argv=(shell.executable, *shell.args, request.command),
                    cwd=str(request.cwd),
                    stdin=None,
                    timeout_seconds=effective_timeout,
                    env=request.env,
                ),
                cancellation_token,
                log_header=f"Command: {request.command}\nWorking directory: {request.cwd}\n",
                log_prefix="agentlane-bash-",
            )
        except asyncio.CancelledError:
            # Preserve Bash's cooperative cancellation result without swallowing
            # actual task cancellation, including simultaneous cancellation paths.
            task = asyncio.current_task()
            if (
                cancellation_token.is_cancelled
                and task is not None
                and not task.cancelling()
            ):
                return _cancelled_result(request, timeout_seconds=effective_timeout)
            raise

        return BashExecutionResult(
            command=request.command,
            cwd=request.cwd,
            exit_code=result.exit_code,
            timed_out=result.stop_reason == ProcessStopReason.TIMEOUT,
            cancelled=result.stop_reason == ProcessStopReason.CANCELLED,
            timeout_seconds=result.timeout_seconds,
            output=result.output,
            full_output_path=(
                Path(result.log_path) if result.log_path is not None else None
            ),
        )


def _cancelled_result(
    request: BashExecutionRequest, *, timeout_seconds: float | None
) -> BashExecutionResult:
    return BashExecutionResult(
        command=request.command,
        cwd=request.cwd,
        exit_code=None,
        timed_out=False,
        cancelled=True,
        timeout_seconds=timeout_seconds,
        output=TruncatedOutput(text="", truncated=False),
        full_output_path=None,
    )


@cache
def resolve_bash_shell() -> BashShellConfig:
    """Resolve a bash-compatible executable without falling back to `sh`."""
    executable = _find_bash_executable()
    return BashShellConfig(executable=executable, args=("-lc",))


def _find_bash_executable() -> str:
    if os.name == "nt":
        for candidate in (
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Git"
            / "bin"
            / "bash.exe",
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
            / "Git"
            / "bin"
            / "bash.exe",
        ):
            if candidate.exists():
                return str(candidate)

        for command in ("bash.exe", "bash"):
            path = shutil.which(command)
            if path is not None:
                return path

        return "bash"

    default_bash = Path("/bin/bash")
    if default_bash.exists():
        return str(default_bash)

    return shutil.which("bash") or "bash"
