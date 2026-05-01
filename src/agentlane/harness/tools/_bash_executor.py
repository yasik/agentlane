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
from typing import Any, Protocol

from agentlane.runtime import CancellationToken

from ._output import BASH_MAX_BYTES, BASH_MAX_LINES, TruncatedOutput, truncate_output

_STREAM_READ_SIZE = 8192
_STREAM_TAIL_BYTES = max(BASH_MAX_BYTES * 4, 256 * 1024)
_COMBINED_LOG_THRESHOLD_BYTES = BASH_MAX_BYTES
_COMBINED_LOG_BUFFER_BYTES = max(BASH_MAX_BYTES * 4, 256 * 1024)
_TERMINATE_GRACE_SECONDS = 2.0
_EXIT_STDIO_GRACE_SECONDS = 0.1


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
    env: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class BashExecutionResult:
    """Captured result for one bash command."""

    command: str
    cwd: Path
    exit_code: int | None
    timed_out: bool
    cancelled: bool
    stdout: TruncatedOutput
    stderr: TruncatedOutput
    full_output_path: Path | None

    @property
    def output_truncated(self) -> bool:
        """Return whether either visible output stream was truncated."""
        return self.stdout.truncated or self.stderr.truncated


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
    """Execute bash commands as local child processes."""

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
        """Run a command and return captured stdout/stderr metadata."""
        effective_timeout = (
            request.timeout_seconds
            if request.timeout_seconds is not None
            else self._default_timeout
        )
        shell_config = self._shell_config or resolve_bash_shell()
        stdout_capture = _StreamCapture()
        stderr_capture = _StreamCapture()
        combined_output = _CombinedOutputRecorder(
            command=request.command,
            cwd=request.cwd,
        )

        process = await asyncio.create_subprocess_exec(
            shell_config.executable,
            *shell_config.args,
            request.command,
            cwd=str(request.cwd),
            env=dict(request.env) if request.env is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **_process_group_kwargs(),
        )
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Expected subprocess pipes to be available.")

        stdout_task = asyncio.create_task(
            _read_stream(
                stream=process.stdout,
                stream_name="stdout",
                capture=stdout_capture,
                combined_output=combined_output,
            )
        )
        stderr_task = asyncio.create_task(
            _read_stream(
                stream=process.stderr,
                stream_name="stderr",
                capture=stderr_capture,
                combined_output=combined_output,
            )
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

        timed_out = False
        cancelled = False
        try:
            wait_tasks: set[asyncio.Task[Any]] = {
                process_exit_task,
                cancellation_task,
            }
            if timeout_task is not None:
                wait_tasks.add(timeout_task)

            done, pending = await asyncio.wait(
                wait_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if process_exit_task in done:
                await process_exit_task
                await _wait_for_stream_tasks_after_exit(
                    process=process,
                    stream_tasks=(stdout_task, stderr_task),
                )
            else:
                timed_out = timeout_task in done
                cancelled = cancellation_task in done
                await _terminate_process_group(
                    process=process,
                    process_exit_task=process_exit_task,
                    stream_tasks=(stdout_task, stderr_task),
                )

            for task in pending:
                task.cancel()
            await _discard_cancelled_tasks(pending)
        except asyncio.CancelledError:
            await _terminate_process_group(
                process=process,
                process_exit_task=process_exit_task,
                stream_tasks=(stdout_task, stderr_task),
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
            if not process_exit_task.done():
                process_exit_task.cancel()
                cleanup_tasks.add(process_exit_task)
            if not process_wait_task.done():
                process_wait_task.cancel()
                cleanup_tasks.add(process_wait_task)
            await _discard_cancelled_tasks(cleanup_tasks)
            combined_output.close()

        stdout = _truncate_capture(stdout_capture)
        stderr = _truncate_capture(stderr_capture)
        full_output_path = combined_output.path
        if not stdout.truncated and not stderr.truncated:
            combined_output.discard()
            full_output_path = None

        return BashExecutionResult(
            command=request.command,
            cwd=request.cwd,
            exit_code=process.returncode,
            timed_out=timed_out,
            cancelled=cancelled,
            stdout=stdout,
            stderr=stderr,
            full_output_path=full_output_path,
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
        self._handle: Any | None = None
        self._path: Path | None = None
        self._current_stream: str | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def path(self) -> Path | None:
        """Return the temp log path when the recorder opened one."""
        return self._path

    async def write(self, *, stream_name: str, chunk: bytes) -> None:
        if not chunk:
            return

        async with self._lock:
            self._total_bytes += len(chunk)
            recorded_chunk = _RecordedChunk(stream_name=stream_name, chunk=chunk)
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
        handle.write(
            f"Command: {self._command}\nWorking directory: {self._cwd}\n".encode(
                "utf-8",
                errors="replace",
            )
        )

    def _write_chunk(self, recorded_chunk: "_RecordedChunk") -> None:
        if self._handle is None:
            return
        if self._current_stream != recorded_chunk.stream_name:
            self._handle.write(f"\n[{recorded_chunk.stream_name}]\n".encode())
            self._current_stream = recorded_chunk.stream_name
        self._handle.write(recorded_chunk.chunk)

    def _trim_buffer(self) -> None:
        while self._buffer_bytes > _COMBINED_LOG_BUFFER_BYTES and self._chunks:
            dropped = self._chunks.popleft()
            self._buffer_bytes -= len(dropped.chunk)


@dataclass(frozen=True, slots=True)
class _RecordedChunk:
    """One chronological output chunk."""

    stream_name: str
    chunk: bytes


class _StreamCapture:
    """Keep a bounded tail of one output stream for model-visible output."""

    def __init__(self) -> None:
        self._tail = bytearray()
        self._dropped = False

    @property
    def dropped(self) -> bool:
        """Return whether bytes were dropped from the in-memory tail."""
        return self._dropped

    def append(self, chunk: bytes) -> None:
        """Append bytes and retain only the bounded tail."""
        self._tail.extend(chunk)
        excess = len(self._tail) - _STREAM_TAIL_BYTES
        if excess > 0:
            del self._tail[:excess]
            self._dropped = True

    def text(self) -> str:
        """Decode the retained bytes as replacement-tolerant text."""
        return bytes(self._tail).decode("utf-8", errors="replace")


async def _read_stream(
    *,
    stream: asyncio.StreamReader,
    stream_name: str,
    capture: _StreamCapture,
    combined_output: _CombinedOutputRecorder,
) -> None:
    while chunk := await stream.read(_STREAM_READ_SIZE):
        capture.append(chunk)
        await combined_output.write(stream_name=stream_name, chunk=chunk)


async def _watch_process_exit(
    *,
    process: asyncio.subprocess.Process,
    process_wait_task: asyncio.Task[int],
) -> int:
    while process.returncode is None:
        if process_wait_task.done():
            return await process_wait_task
        await asyncio.sleep(0.01)
    return process.returncode


async def _wait_for_stream_tasks_after_exit(
    *,
    process: asyncio.subprocess.Process,
    stream_tasks: tuple[asyncio.Task[None], asyncio.Task[None]],
) -> None:
    done, pending = await asyncio.wait(
        set(stream_tasks),
        timeout=_EXIT_STDIO_GRACE_SECONDS,
    )
    if not pending:
        await _propagate_stream_task_errors(done)
        return

    _send_process_signal(process, signal.SIGKILL, include_exited_group=True)
    for task in pending:
        task.cancel()
    await _discard_cancelled_tasks(pending)
    await _propagate_stream_task_errors(done)


async def _terminate_process_group(
    *,
    process: asyncio.subprocess.Process,
    process_exit_task: asyncio.Task[int],
    stream_tasks: tuple[asyncio.Task[None], asyncio.Task[None]],
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

    _send_process_signal(process, signal.SIGKILL, include_exited_group=True)
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
        elif sig == signal.SIGKILL:
            _kill_windows_process_tree(process.pid)
        elif process.returncode is None:
            process.terminate()
    except ProcessLookupError:
        return


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


def _truncate_capture(capture: _StreamCapture) -> TruncatedOutput:
    output = truncate_output(
        capture.text(),
        max_lines=BASH_MAX_LINES,
        max_bytes=BASH_MAX_BYTES,
        tail=True,
    )
    if not capture.dropped or output.truncated:
        return output

    return TruncatedOutput(
        text=(
            f"[output truncated: showing last {BASH_MAX_LINES} lines or "
            f"{BASH_MAX_BYTES} bytes]\n{output.text}"
        ),
        truncated=True,
    )
