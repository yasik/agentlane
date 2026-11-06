"""Local bash command execution primitives for harness tools."""

import asyncio
import os
import shutil
import signal
import subprocess
import tempfile
from collections import deque
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from agentlane.runtime import CancellationToken

from ._output import BASH_MAX_BYTES, BASH_MAX_LINES, TruncatedOutput, truncate_output

_STREAM_READ_SIZE = 8192
_COMBINED_LOG_THRESHOLD_BYTES = BASH_MAX_BYTES
_COMBINED_LOG_BUFFER_BYTES = max(BASH_MAX_BYTES * 4, 256 * 1024)
_TERMINATE_GRACE_SECONDS = 2.0
_EXIT_STDIO_GRACE_SECONDS = 0.1
_PROCESS_EXIT_POLL_SECONDS = 0.01


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
    stdin: bytes | None = None
    """Bytes to send before closing stdin; None preserves inherited stdin."""
    stdout_capture_limit: int | None = None
    """Capture up to this many raw stdout bytes; None disables raw capture."""
    shell_startup: bool = True
    """Allow startup files and inherited shell functions/options when true."""


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
    stdout: bytes | None = None
    """Raw stdout when capture was requested, without stderr or log headers."""
    stdout_truncated: bool = False
    """Whether captured stdout exceeded its limit or did not reach EOF."""

    @property
    def output_truncated(self) -> bool:
        """Return whether visible output was truncated."""
        return self.output.truncated


class BashExecutor(Protocol):
    """Protocol for reusable bash command executors.

    Executors that accept `BashExecutionRequest.stdin` also implement
    `StdinBashExecutor`. Callers check that optional capability before supplying
    input. Raw stdout capture requires `StdoutBashExecutor`. Executors must
    reject input or capture requests they cannot support. When `shell_startup`
    is false, executors must disable startup files and inherited shell
    functions/options, or reject the request before execution.
    """

    async def run(
        self,
        request: BashExecutionRequest,
        cancellation_token: CancellationToken,
    ) -> BashExecutionResult:
        """Execute one bash command."""
        ...


@runtime_checkable
class StdinBashExecutor(BashExecutor, Protocol):
    """Optional command-executor capability for binary standard input.

    When supported, `run` sends all supplied bytes and closes the input stream.
    Input transfer participates in the command's timeout and cancellation.
    """

    supports_stdin: bool
    """Whether this executor can send `BashExecutionRequest.stdin` bytes."""


@runtime_checkable
class StdoutBashExecutor(BashExecutor, Protocol):
    """Optional command-executor capability for bounded raw stdout capture.

    Capture preserves exact bytes separately from the display output. Exceeding
    the limit stops execution. Missing EOF marks the capture as truncated even
    when the command's leader exits successfully.
    """

    supports_stdout_capture: bool
    """Whether this executor supports `BashExecutionRequest.stdout_capture_limit`."""


class LocalBashExecutor:
    """Execute bash commands as local child processes."""

    supports_stdin: bool = True
    """Whether requests can supply bytes for the child process stdin."""
    supports_stdout_capture: bool = True
    """Whether requests can capture bounded raw stdout bytes."""

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
        """Run a command and return captured combined-output metadata."""
        if (
            request.stdout_capture_limit is not None
            and request.stdout_capture_limit < 1
        ):
            raise ValueError("stdout_capture_limit must be greater than zero.")

        effective_timeout = (
            request.timeout_seconds
            if request.timeout_seconds is not None
            else self._default_timeout
        )
        shell_config = self._shell_config or resolve_bash_shell()

        # Keep startup output and imported functions out of requests that opt out.
        # Preserve host shell options, including a final option terminator.
        shell_args = shell_config.args
        if not request.shell_startup:
            if shell_args[-1:] == ("--",):
                shell_args = (*shell_args[:-1], "-p", "--")
            else:
                shell_args = (*shell_args, "-p")
            shell_args = ("--noprofile", "--norc", *shell_args)

        combined_output = _CombinedOutputRecorder(
            command=request.command,
            cwd=request.cwd,
        )
        stdout_capture = (
            _StdoutCapture(request.stdout_capture_limit)
            if request.stdout_capture_limit is not None
            else None
        )

        process = await asyncio.create_subprocess_exec(
            shell_config.executable,
            *shell_args,
            request.command,
            cwd=str(request.cwd),
            env=dict(request.env) if request.env is not None else None,
            stdin=asyncio.subprocess.PIPE if request.stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **_process_group_kwargs(),
        )
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Expected subprocess pipes to be available.")

        stdout_task = asyncio.create_task(
            _read_stream(
                stream=process.stdout,
                combined_output=combined_output,
                stdout_capture=stdout_capture,
            )
        )
        stderr_task = asyncio.create_task(
            _read_stream(
                stream=process.stderr,
                combined_output=combined_output,
            )
        )
        stream_tasks: tuple[asyncio.Task[None], ...] = (stdout_task, stderr_task)
        if request.stdin is not None:
            if process.stdin is None:
                raise RuntimeError("Expected subprocess stdin pipe to be available.")
            stream_tasks += (
                asyncio.create_task(_write_stdin(process.stdin, request.stdin)),
            )
        process_wait_task = asyncio.create_task(process.wait())
        process_exit_task = asyncio.create_task(
            _watch_process_exit(
                process=process,
                process_wait_task=process_wait_task,
            )
        )
        cancellation_task = asyncio.create_task(cancellation_token.wait_cancelled())
        timeout_task = (
            asyncio.create_task(asyncio.sleep(effective_timeout))
            if effective_timeout is not None
            else None
        )
        capture_limit_task = (
            asyncio.create_task(stdout_capture.overflow.wait())
            if stdout_capture is not None
            else None
        )

        timed_out = False
        cancelled = False
        try:
            wait_tasks: set[asyncio.Task[Any]] = {
                process_exit_task,
                cancellation_task,
            }
            if timeout_task is not None:
                wait_tasks.add(timeout_task)
            if capture_limit_task is not None:
                wait_tasks.add(capture_limit_task)

            done, pending = await asyncio.wait(
                wait_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if process_exit_task in done:
                await process_exit_task
                await _wait_for_stream_tasks_after_exit(
                    process=process,
                    stream_tasks=stream_tasks,
                )
            else:
                timed_out = timeout_task in done
                cancelled = cancellation_task in done
                await _terminate_process_group(
                    process=process,
                    process_exit_task=process_exit_task,
                    stream_tasks=stream_tasks,
                )

            for task in pending:
                task.cancel()
            await _discard_cancelled_tasks(pending)
        except asyncio.CancelledError:
            await _terminate_process_group(
                process=process,
                process_exit_task=process_exit_task,
                stream_tasks=stream_tasks,
            )
            raise
        finally:
            cleanup_tasks: set[asyncio.Task[Any]] = set()
            if not cancellation_task.done():
                cancellation_task.cancel()
                cleanup_tasks.add(cancellation_task)
            if timeout_task is not None and not timeout_task.done():
                timeout_task.cancel()
                cleanup_tasks.add(timeout_task)
            if capture_limit_task is not None and not capture_limit_task.done():
                capture_limit_task.cancel()
                cleanup_tasks.add(capture_limit_task)
            if not process_exit_task.done():
                process_exit_task.cancel()
                cleanup_tasks.add(process_exit_task)
            if not process_wait_task.done():
                process_wait_task.cancel()
                cleanup_tasks.add(process_wait_task)
            for task in stream_tasks:
                if not task.done():
                    task.cancel()
                    cleanup_tasks.add(task)
            await _discard_cancelled_tasks(cleanup_tasks)
            combined_output.close()

        output = _truncate_combined_output(combined_output)
        full_output_path = combined_output.path
        if output.truncated and full_output_path is None:
            full_output_path = combined_output.materialize()
        if not output.truncated:
            combined_output.discard()
            full_output_path = None

        return BashExecutionResult(
            command=request.command,
            cwd=request.cwd,
            exit_code=process.returncode,
            timed_out=timed_out,
            cancelled=cancelled,
            timeout_seconds=effective_timeout,
            output=output,
            full_output_path=full_output_path,
            stdout=(
                bytes(stdout_capture.content) if stdout_capture is not None else None
            ),
            stdout_truncated=(
                stdout_capture is not None
                and (stdout_capture.overflow.is_set() or not stdout_capture.eof)
            ),
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


class _CombinedOutputRecorder:
    """Lazy temp-file sink for full combined command output."""

    def __init__(self, *, command: str, cwd: Path) -> None:
        self._command = command
        self._cwd = cwd
        self._chunks: deque[_RecordedChunk] = deque()
        self._buffer_bytes = 0
        self._total_bytes = 0
        self._dropped = False
        self._handle: Any | None = None
        self._path: Path | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def path(self) -> Path | None:
        """Return the temp log path when the recorder opened one."""
        return self._path

    @property
    def dropped(self) -> bool:
        """Return whether bytes were dropped from the in-memory tail."""
        return self._dropped

    async def write(self, chunk: bytes) -> None:
        if not chunk:
            return

        async with self._lock:
            self._total_bytes += len(chunk)
            recorded_chunk = _RecordedChunk(chunk=chunk)
            self._chunks.append(recorded_chunk)
            self._buffer_bytes += len(chunk)
            self._trim_buffer()

            if self._handle is None:
                if self._total_bytes <= _COMBINED_LOG_THRESHOLD_BYTES:
                    return
                self._open()
                for buffered_chunk in self._chunks:
                    self._write_chunk(buffered_chunk)
                return

            self._write_chunk(recorded_chunk)

    def close(self) -> None:
        if self._closed:
            return
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
        self._closed = True

    def discard(self) -> None:
        self.close()
        if self._path is None:
            return
        with suppress(FileNotFoundError):
            self._path.unlink()
        self._path = None

    def _open(self) -> None:
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            prefix="agentlane-bash-",
            suffix=".log",
        )
        self._handle = handle
        self._path = Path(handle.name)
        self._write_header(handle)

    def materialize(self) -> Path:
        """Write the retained full output tail to a temp log.

        Safe to call after close(); this opens a separate file handle.
        """
        if self._path is not None:
            return self._path

        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            prefix="agentlane-bash-",
            suffix=".log",
        )
        path = Path(handle.name)
        with handle:
            self._write_header(handle)
            for recorded_chunk in self._chunks:
                handle.write(recorded_chunk.chunk)
        self._path = path
        return path

    def _write_chunk(self, recorded_chunk: "_RecordedChunk") -> None:
        if self._handle is None:
            return
        self._handle.write(recorded_chunk.chunk)

    def _write_header(self, handle: Any) -> None:
        handle.write(
            f"Command: {self._command}\nWorking directory: {self._cwd}\n".encode(
                "utf-8",
                errors="replace",
            )
        )

    def _trim_buffer(self) -> None:
        while self._buffer_bytes > _COMBINED_LOG_BUFFER_BYTES and self._chunks:
            dropped = self._chunks.popleft()
            self._buffer_bytes -= len(dropped.chunk)
            self._dropped = True

    def text(self) -> str:
        """Decode the retained chronological output tail."""
        output = b"".join(chunk.chunk for chunk in self._chunks)
        return output.decode("utf-8", errors="replace")


@dataclass(frozen=True, slots=True)
class _RecordedChunk:
    """One chronological output chunk."""

    chunk: bytes


class _StdoutCapture:
    """Keep a bounded stdout prefix and its completion state."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self.content = bytearray()
        self.eof = False
        self.overflow = asyncio.Event()

    def append(self, chunk: bytes) -> None:
        remaining = self._limit - len(self.content)
        self.content.extend(chunk[:remaining])
        if len(chunk) > remaining:
            self.overflow.set()


async def _read_stream(
    *,
    stream: asyncio.StreamReader,
    combined_output: _CombinedOutputRecorder,
    stdout_capture: _StdoutCapture | None = None,
) -> None:
    while chunk := await stream.read(_STREAM_READ_SIZE):
        if stdout_capture is not None:
            if stdout_capture.overflow.is_set():
                # Keep draining until termination closes the pipe, without
                # retaining or logging further bytes after capture overflow.
                continue
            stdout_capture.append(chunk)
        await combined_output.write(chunk)

    if stdout_capture is not None:
        stdout_capture.eof = True


async def _write_stdin(stream: asyncio.StreamWriter, data: bytes) -> None:
    """Send input while output is drained, then close the child's stdin."""
    close_task: asyncio.Task[None] | None = None
    try:
        try:
            for offset in range(0, len(data), _STREAM_READ_SIZE):
                stream.write(data[offset : offset + _STREAM_READ_SIZE])
                await stream.drain()
        finally:
            stream.close()

        # The subprocess protocol owns the close waiter. Keep cancellation from
        # cancelling that waiter before its connection-lost callback completes.
        close_task = asyncio.create_task(stream.wait_closed())
        await asyncio.shield(close_task)
    except asyncio.CancelledError:
        # Discard buffered input: a detached child can keep the pipe open
        # without reading, even after the command's process group exits.
        stream.transport.abort()
        if close_task is None:
            close_task = asyncio.create_task(stream.wait_closed())
        with suppress(BrokenPipeError, ConnectionResetError):
            await asyncio.shield(close_task)
        raise
    except (BrokenPipeError, ConnectionResetError):
        # A command can finish successfully before it consumes all its input.
        pass


async def _watch_process_exit(
    *,
    process: asyncio.subprocess.Process,
    process_wait_task: asyncio.Task[int],
) -> int:
    while process.returncode is None:
        if process_wait_task.done():
            return await process_wait_task
        # asyncio's wait task can remain pending until inherited pipe handles
        # close, even after the child process returncode is known.
        await asyncio.sleep(_PROCESS_EXIT_POLL_SECONDS)
    return process.returncode


async def _wait_for_stream_tasks_after_exit(
    *,
    process: asyncio.subprocess.Process,
    stream_tasks: tuple[asyncio.Task[None], ...],
) -> None:
    done, pending = await asyncio.wait(
        set(stream_tasks),
        timeout=_EXIT_STDIO_GRACE_SECONDS,
    )
    if not pending:
        await _propagate_stream_task_errors(done)
        return

    _kill_process_group(process, include_exited_group=True)
    for task in pending:
        task.cancel()
    await _discard_cancelled_tasks(pending)
    await _propagate_stream_task_errors(done)


async def _terminate_process_group(
    *,
    process: asyncio.subprocess.Process,
    process_exit_task: asyncio.Task[int],
    stream_tasks: tuple[asyncio.Task[None], ...],
) -> None:
    if process.returncode is None:
        _send_process_signal(process, signal.SIGTERM)

    with suppress(TimeoutError):
        await asyncio.wait_for(
            asyncio.shield(process_exit_task),
            _TERMINATE_GRACE_SECONDS,
        )
        await _wait_for_stream_tasks_after_exit(
            process=process,
            stream_tasks=stream_tasks,
        )
        return

    _kill_process_group(process, include_exited_group=True)
    with suppress(TimeoutError):
        await asyncio.wait_for(asyncio.shield(process_exit_task), 1.0)
    await _wait_for_stream_tasks_after_exit(
        process=process,
        stream_tasks=stream_tasks,
    )


def _send_process_signal(
    process: asyncio.subprocess.Process,
    sig: signal.Signals,
    *,
    include_exited_group: bool = False,
) -> None:
    if process.returncode is not None and not include_exited_group:
        return

    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        elif process.returncode is None:
            _send_windows_graceful_signal(process)
    except ProcessLookupError:
        return


def _kill_process_group(
    process: asyncio.subprocess.Process,
    *,
    include_exited_group: bool = False,
) -> None:
    if process.returncode is not None and not include_exited_group:
        return

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif os.name == "nt":
            _kill_windows_process_tree(process.pid)
    except ProcessLookupError:
        return


def _send_windows_graceful_signal(process: asyncio.subprocess.Process) -> None:
    ctrl_break_event = getattr(signal, "CTRL_BREAK_EVENT", None)
    if ctrl_break_event is not None:
        try:
            process.send_signal(ctrl_break_event)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass

    # Fallback is leader-only, used when CTRL_BREAK_EVENT is unavailable or
    # cannot be delivered to the process group.
    process.terminate()


def _kill_windows_process_tree(pid: int) -> None:
    taskkill_path = (
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
    )
    try:
        subprocess.run(  # noqa: S603
            [str(taskkill_path), "/F", "/T", "/PID", str(pid)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return


def _process_group_kwargs() -> dict[str, Any]:
    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {}


async def _discard_cancelled_tasks(tasks: set[asyncio.Task[Any]]) -> None:
    for task in tasks:
        with suppress(asyncio.CancelledError):
            await task


async def _propagate_stream_task_errors(tasks: set[asyncio.Task[None]]) -> None:
    for task in tasks:
        await task


def _truncate_combined_output(recorder: _CombinedOutputRecorder) -> TruncatedOutput:
    output = truncate_output(
        recorder.text(),
        max_lines=BASH_MAX_LINES,
        max_bytes=BASH_MAX_BYTES,
        tail=True,
        include_marker=False,
    )
    if not recorder.dropped and not output.truncated:
        return output

    return TruncatedOutput(text=output.text, truncated=True)
