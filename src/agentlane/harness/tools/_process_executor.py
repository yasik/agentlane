"""Native process execution and byte streams for harness tools."""

import asyncio
import json
import os
import signal
import subprocess
import tempfile
from collections import deque
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import Any, Protocol

from agentlane.runtime import CancellationToken

from ._output import (
    PROCESS_MAX_BYTES,
    PROCESS_MAX_LINES,
    TruncatedOutput,
    truncate_output,
)

_STREAM_READ_SIZE = 8192
_COMBINED_LOG_THRESHOLD_BYTES = PROCESS_MAX_BYTES
_COMBINED_LOG_BUFFER_BYTES = max(PROCESS_MAX_BYTES * 4, 256 * 1024)
_TERMINATE_GRACE_SECONDS = 2.0
_EXIT_STDIO_GRACE_SECONDS = 0.1
_PROCESS_EXIT_POLL_SECONDS = 0.01


class ProcessStopReason(StrEnum):
    """Reason a started process finished execution."""

    EXITED = "exited"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    STDOUT_LIMIT = "stdout_limit"


@dataclass(frozen=True, slots=True)
class ProcessExecutionRequest:
    """Execute an argument vector in the selected backend's namespace."""

    argv: tuple[str, ...]
    cwd: str
    stdin: bytes | None = b""
    """Input bytes followed by EOF; None inherits the backend's input."""
    timeout_seconds: float | None = None
    """Positive execution timeout; None uses the executor's default."""
    env: Mapping[str, str] | None = None
    """Replacement environment; None uses the executor's default."""
    stdout_capture_limit: int | None = None
    """Positive raw stdout byte bound; None keeps only preview and log output."""


@dataclass(frozen=True, slots=True)
class ProcessExecutionResult:
    """Terminal result after the child and owned stream tasks have settled."""

    exit_code: int
    stop_reason: ProcessStopReason
    timeout_seconds: float | None
    stdout: bytes | None
    stdout_complete: bool
    output: TruncatedOutput
    log_path: str | None
    log_complete: bool


class ProcessExecutor(Protocol):
    """Backend contract for native argv, binary input, and bounded raw output.

    Backends own executable, cwd, and environment interpretation. They must
    support the requested byte streams without falling back to another backend.
    Startup and transport errors raise; task cancellation settles cleanup and
    propagates. A returned result always has a known child exit code.
    """

    async def run(
        self,
        request: ProcessExecutionRequest,
        cancellation_token: CancellationToken,
    ) -> ProcessExecutionResult:
        """Execute a native process without interpreting arguments as shell text."""
        ...


class LocalProcessExecutor:
    """Execute native argument vectors as local child processes."""

    def __init__(
        self,
        *,
        default_timeout: float | None = None,
        default_env: Mapping[str, str] | None = None,
    ) -> None:
        _validate_timeout(default_timeout, name="default_timeout")
        self._default_timeout = default_timeout
        self._default_env = dict(default_env) if default_env is not None else None

    async def run(
        self,
        request: ProcessExecutionRequest,
        cancellation_token: CancellationToken,
    ) -> ProcessExecutionResult:
        """Run argv directly and return bounded output and completion metadata."""
        configured = replace(
            request,
            timeout_seconds=(
                request.timeout_seconds
                if request.timeout_seconds is not None
                else self._default_timeout
            ),
            env=request.env if request.env is not None else self._default_env,
        )
        return await run_local_process(configured, cancellation_token)


async def run_local_process(
    request: ProcessExecutionRequest,
    cancellation_token: CancellationToken,
    *,
    log_header: str | None = None,
    log_prefix: str = "agentlane-process-",
) -> ProcessExecutionResult:
    """Own execution; adapters can preserve their established log presentation."""
    _validate_request(request)
    if cancellation_token.is_cancelled:
        raise asyncio.CancelledError

    # Snapshot caller-owned configuration before the first asynchronous step.
    effective_timeout = request.timeout_seconds
    env = dict(request.env) if request.env is not None else None
    recorder = _CombinedOutputRecorder(
        header=(
            log_header
            if log_header is not None
            else f"Arguments: {json.dumps(request.argv)}\n"
            f"Working directory: {request.cwd}\n"
        ),
        prefix=log_prefix,
    )

    # Keep ownership if task cancellation arrives while the OS starts a child.
    startup = asyncio.create_task(
        _start_process(
            request=request,
            env=env,
            cancellation_token=cancellation_token,
        )
    )
    startup_cancelled = await _settle_task(startup)
    try:
        process = startup.result()
    except BaseException:
        recorder.close()
        if startup_cancelled:
            raise asyncio.CancelledError from None
        raise

    return await _run_started_process(
        process=process,
        request=request,
        cancellation_token=cancellation_token,
        effective_timeout=effective_timeout,
        recorder=recorder,
        startup_cancelled=startup_cancelled,
    )


async def _start_process(
    *,
    request: ProcessExecutionRequest,
    env: dict[str, str] | None,
    cancellation_token: CancellationToken,
) -> asyncio.subprocess.Process:
    # Cancellation can arrive after startup was queued but before it begins.
    if cancellation_token.is_cancelled:
        raise asyncio.CancelledError

    return await asyncio.create_subprocess_exec(
        *request.argv,
        cwd=request.cwd,
        env=env,
        stdin=asyncio.subprocess.PIPE if request.stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **_process_group_kwargs(),
    )


def _validate_timeout(timeout: float | None, *, name: str) -> None:
    if timeout is not None and (not isfinite(timeout) or timeout <= 0):
        raise ValueError(f"{name} must be finite and greater than zero.")


def _validate_request(request: ProcessExecutionRequest) -> None:
    if not request.argv or not request.argv[0]:
        raise ValueError("argv must contain a nonempty executable.")
    if any("\x00" in argument for argument in request.argv):
        raise ValueError("argv must not contain null bytes.")
    if not request.cwd or "\x00" in request.cwd:
        raise ValueError("cwd must be nonempty and must not contain null bytes.")
    _validate_timeout(request.timeout_seconds, name="timeout_seconds")
    if request.stdout_capture_limit is not None and request.stdout_capture_limit < 1:
        raise ValueError("stdout_capture_limit must be greater than zero.")


@dataclass(slots=True)
class _StreamState:
    """Completion and log omissions for one output stream."""

    eof: bool = False
    log_omitted: bool = False


async def _run_started_process(
    *,
    process: asyncio.subprocess.Process,
    request: ProcessExecutionRequest,
    cancellation_token: CancellationToken,
    effective_timeout: float | None,
    recorder: "_CombinedOutputRecorder",
    startup_cancelled: bool,
) -> ProcessExecutionResult:
    stdout_capture = (
        _StdoutCapture(request.stdout_capture_limit)
        if request.stdout_capture_limit is not None
        else None
    )
    stdout_state = _StreamState()
    stderr_state = _StreamState()
    stream_tasks: list[asyncio.Task[None]] = []
    for stream, state, capture in (
        (process.stdout, stdout_state, stdout_capture),
        (process.stderr, stderr_state, None),
    ):
        if stream is not None:
            stream_tasks.append(
                asyncio.create_task(
                    _read_stream(
                        stream=stream,
                        state=state,
                        combined_output=recorder,
                        stdout_capture=capture,
                    )
                )
            )
    if request.stdin is not None and process.stdin is not None:
        stream_tasks.append(
            asyncio.create_task(_write_stdin(process.stdin, request.stdin))
        )

    process_wait_task = asyncio.create_task(process.wait())
    process_exit_task = asyncio.create_task(
        _watch_process_exit(process=process, process_wait_task=process_wait_task)
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
    auxiliary_tasks: set[asyncio.Task[Any]] = {cancellation_task}
    if timeout_task is not None:
        auxiliary_tasks.add(timeout_task)
    if capture_limit_task is not None:
        auxiliary_tasks.add(capture_limit_task)

    # Supervise each pipe as well as the leader, so a failed sink cannot deadlock it.
    failure: BaseException | None = None
    reason = ProcessStopReason.EXITED
    try:
        if startup_cancelled:
            raise asyncio.CancelledError
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Expected subprocess output pipes to be available.")
        if request.stdin is not None and process.stdin is None:
            raise RuntimeError("Expected subprocess stdin pipe to be available.")
        reason = await _supervise_process(
            process_exit_task=process_exit_task,
            stream_tasks=stream_tasks,
            cancellation_task=cancellation_task,
            timeout_task=timeout_task,
            capture_limit_task=capture_limit_task,
        )
    except BaseException as error:
        failure = error

    # Cleanup has one task owner. Repeated caller cancellation cannot abandon it.
    cleanup = asyncio.create_task(
        _cleanup_process(
            process=process,
            process_exit_task=process_exit_task,
            process_wait_task=process_wait_task,
            stream_tasks=stream_tasks,
            auxiliary_tasks=auxiliary_tasks,
        )
    )
    cleanup_cancelled = await _settle_task(cleanup)
    try:
        cleanup.result()
    except BaseException as error:
        if failure is None:
            failure = error
    try:
        recorder.close()
    except Exception as error:
        if failure is None:
            failure = error
    if cleanup_cancelled:
        raise asyncio.CancelledError
    if failure is not None:
        raise failure

    # Normal exit is provisional until buffered output and cancellation settle.
    if reason == ProcessStopReason.EXITED:
        if cancellation_token.is_cancelled:
            reason = ProcessStopReason.CANCELLED
        elif stdout_capture is not None and stdout_capture.overflow.is_set():
            reason = ProcessStopReason.STDOUT_LIMIT
    if process.returncode is None:
        raise RuntimeError("Process cleanup did not produce an exit code.")
    output = _truncate_combined_output(recorder)
    log_path = recorder.path
    if output.truncated and log_path is None:
        log_path = recorder.materialize()
    if not output.truncated:
        recorder.discard()
        log_path = None

    return ProcessExecutionResult(
        exit_code=process.returncode,
        stop_reason=reason,
        timeout_seconds=effective_timeout,
        stdout=bytes(stdout_capture.content) if stdout_capture is not None else None,
        stdout_complete=(
            stdout_capture is not None
            and stdout_state.eof
            and not stdout_capture.overflow.is_set()
        ),
        output=output,
        log_path=str(log_path) if log_path is not None else None,
        log_complete=(
            stdout_state.eof
            and stderr_state.eof
            and not stdout_state.log_omitted
            and not stderr_state.log_omitted
        ),
    )


async def _supervise_process(
    *,
    process_exit_task: asyncio.Task[int],
    stream_tasks: list[asyncio.Task[None]],
    cancellation_task: asyncio.Task[None],
    timeout_task: asyncio.Task[None] | None,
    capture_limit_task: asyncio.Task[bool] | None,
) -> ProcessStopReason:
    pending_streams = set(stream_tasks)
    exit_deadline: float | None = None
    loop = asyncio.get_running_loop()
    while True:
        wait_tasks: set[asyncio.Task[Any]] = {cancellation_task, *pending_streams}
        if not process_exit_task.done():
            wait_tasks.add(process_exit_task)
        if timeout_task is not None:
            wait_tasks.add(timeout_task)
        if capture_limit_task is not None:
            wait_tasks.add(capture_limit_task)
        await asyncio.wait(
            wait_tasks,
            timeout=(max(0.0, exit_deadline - loop.time()) if exit_deadline else None),
            return_when=asyncio.FIRST_COMPLETED,
        )

        # A successful EOF or stdin close is not a process stop condition.
        for task in tuple(pending_streams):
            if task.done():
                pending_streams.remove(task)
                task.result()
        if cancellation_task.done():
            return ProcessStopReason.CANCELLED
        if capture_limit_task is not None and capture_limit_task.done():
            return ProcessStopReason.STDOUT_LIMIT
        if timeout_task is not None and timeout_task.done():
            return ProcessStopReason.TIMEOUT
        if process_exit_task.done():
            process_exit_task.result()
            if not pending_streams:
                return ProcessStopReason.EXITED
            if exit_deadline is None:
                exit_deadline = loop.time() + _EXIT_STDIO_GRACE_SECONDS
            elif loop.time() >= exit_deadline:
                return ProcessStopReason.EXITED


async def _settle_task(task: asyncio.Task[Any]) -> bool:
    """Join an owned task without letting caller cancellation abandon it."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break

    return cancelled


async def _cleanup_process(
    *,
    process: asyncio.subprocess.Process,
    process_exit_task: asyncio.Task[int],
    process_wait_task: asyncio.Task[int],
    stream_tasks: list[asyncio.Task[None]],
    auxiliary_tasks: set[asyncio.Task[Any]],
) -> None:
    if process.returncode is None:
        _send_process_signal(process, signal.SIGTERM)
        try:
            await asyncio.wait_for(
                asyncio.shield(process_exit_task), _TERMINATE_GRACE_SECONDS
            )
        except TimeoutError:
            _kill_process_group(process, include_exited_group=True)
            await process_exit_task

    # A descendant can retain pipe handles after the leader exits. Stop the
    # owned group and close our pipe transports even if a descendant detached.
    if any(not task.done() for task in stream_tasks):
        _kill_process_group(process, include_exited_group=True)
    for task in stream_tasks:
        if not task.done():
            task.cancel()

    # The writer owns aborting buffered stdin. Join it before closing the whole
    # transport; aborting an already-closed pipe is not safe on every platform.
    stream_results = await asyncio.gather(*stream_tasks, return_exceptions=True)
    transport = getattr(process, "_transport", None)
    if transport is not None:
        transport.close()
    for task in auxiliary_tasks:
        if not task.done():
            task.cancel()

    results = await asyncio.gather(
        *auxiliary_tasks,
        process_exit_task,
        process_wait_task,
        return_exceptions=True,
    )
    for result in (*stream_results, *results):
        if isinstance(result, Exception):
            raise result


class _CombinedOutputRecorder:
    """Lazy temp-file sink for full combined command output."""

    def __init__(self, *, header: str, prefix: str) -> None:
        self._header = header
        self._prefix = prefix
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
        self._closed = True
        if self._handle is not None:
            try:
                self._handle.flush()
            finally:
                self._handle.close()

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
            prefix=self._prefix,
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
            prefix=self._prefix,
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
        handle.write(self._header.encode("utf-8", errors="replace"))

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
        self.overflow = asyncio.Event()

    def append(self, chunk: bytes) -> None:
        remaining = self._limit - len(self.content)
        self.content.extend(chunk[:remaining])
        if len(chunk) > remaining:
            self.overflow.set()


async def _read_stream(
    *,
    stream: asyncio.StreamReader,
    state: _StreamState,
    combined_output: _CombinedOutputRecorder,
    stdout_capture: _StdoutCapture | None = None,
) -> None:
    while chunk := await stream.read(_STREAM_READ_SIZE):
        if stdout_capture is not None:
            if stdout_capture.overflow.is_set():
                # Keep draining until termination closes the pipe, without
                # retaining or logging further bytes after capture overflow.
                state.log_omitted = True
                continue
            stdout_capture.append(chunk)
        await combined_output.write(chunk)

    state.eof = True


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


def _truncate_combined_output(recorder: _CombinedOutputRecorder) -> TruncatedOutput:
    output = truncate_output(
        recorder.text(),
        max_lines=PROCESS_MAX_LINES,
        max_bytes=PROCESS_MAX_BYTES,
        tail=True,
        include_marker=False,
    )
    if not recorder.dropped and not output.truncated:
        return output

    return TruncatedOutput(text=output.text, truncated=True)
