"""Real-process tests for native argv, byte streams, and process cleanup."""

import asyncio
import importlib
import os
import signal
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import pytest

from agentlane.harness.tools import (
    LocalProcessExecutor,
    ProcessExecutionRequest,
    ProcessStopReason,
)
from agentlane.harness.tools._command_output import format_process_output
from agentlane.runtime import CancellationToken


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [b"", b"first\x00line\n\xc3\xa9\n"])
async def test_local_executor_sends_stdin_bytes_and_closes_pipe(
    tmp_path: Path, data: bytes
) -> None:
    argv = (sys.executable, "-c", "import sys; print(sys.stdin.buffer.read().hex())")
    result = await LocalProcessExecutor().run(
        ProcessExecutionRequest(
            argv=argv, cwd=str(tmp_path), stdin=data, timeout_seconds=2
        ),
        CancellationToken(),
    )

    assert result.exit_code == 0
    assert result.stop_reason != ProcessStopReason.TIMEOUT
    assert result.output.text == f"{data.hex()}\n"


@pytest.mark.asyncio
async def test_local_executor_drains_output_while_sending_large_stdin(
    tmp_path: Path,
) -> None:
    data = b"input-value\n" * 100_000
    code = (
        "import sys; "
        "sys.stdout.write('x' * 131072); sys.stdout.flush(); "
        "data = sys.stdin.buffer.read(); "
        "print('\\nreceived=' + str(len(data)) + ',valid=' + "
        "str(data == b'input-value\\n' * 100000))"
    )
    result = await LocalProcessExecutor().run(
        ProcessExecutionRequest(
            argv=(sys.executable, "-c", code),
            cwd=str(tmp_path),
            stdin=data,
            timeout_seconds=3,
        ),
        CancellationToken(),
    )

    try:
        assert result.exit_code == 0
        assert result.stop_reason != ProcessStopReason.TIMEOUT
        assert result.output.text.endswith(f"\nreceived={len(data)},valid=True\n")
        assert result.output.truncated
        assert result.log_path is not None
        assert "x" * 131072 in Path(result.log_path).read_text(encoding="utf-8")
    finally:
        if result.log_path is not None:
            Path(result.log_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_local_executor_accepts_exit_before_stdin_is_consumed(
    tmp_path: Path,
) -> None:
    result = await LocalProcessExecutor().run(
        ProcessExecutionRequest(
            argv=(sys.executable, "-c", "print('done', end='')"),
            cwd=str(tmp_path),
            stdin=b"unused" * 200_000,
            timeout_seconds=2,
        ),
        CancellationToken(),
    )

    assert result.exit_code == 0
    assert result.output.text == "done"
    assert result.stop_reason != ProcessStopReason.TIMEOUT


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["timeout", "token", "task"])
async def test_local_executor_stops_while_stdin_is_blocked(
    tmp_path: Path, stop: str
) -> None:
    initial_tasks = asyncio.all_tasks()
    token = CancellationToken()
    task = asyncio.create_task(
        LocalProcessExecutor().run(
            ProcessExecutionRequest(
                argv=(sys.executable, "-c", "import time; time.sleep(5)"),
                cwd=str(tmp_path),
                stdin=b"unused" * 200_000,
                timeout_seconds=0.1 if stop == "timeout" else None,
            ),
            token,
        )
    )
    if stop != "timeout":
        await asyncio.sleep(0.1)
        if stop == "token":
            token.cancel()
        else:
            task.cancel()

    if stop == "task":
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)
    else:
        result = await asyncio.wait_for(task, timeout=3)
        assert (result.stop_reason == ProcessStopReason.TIMEOUT) == (stop == "timeout")
        assert (result.stop_reason == ProcessStopReason.CANCELLED) == (stop == "token")
        assert result.exit_code is not None

    assert asyncio.all_tasks() == initial_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["exit", "timeout", "token", "task"])
@pytest.mark.parametrize("input_bytes", [32 * 1024, 96 * 1024, 2 * 1024 * 1024])
async def test_local_executor_closes_stdin_held_by_detached_child(
    tmp_path: Path, stop: str, input_bytes: int
) -> None:
    if os.name != "posix":
        pytest.skip("requires POSIX process sessions")

    # The detached child keeps stdin open without reading it, even after the
    # command's process group has exited.
    pid_file = tmp_path / "child.pid"
    code = (
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(30)'],\n"
        "    start_new_session=True,\n"
        "    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,\n"
        ")\n"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))\n"
        "time.sleep(0 if sys.argv[2] == 'exit' else 30)\n"
    )

    # Pipe capacities vary by platform. Exercise both pending writes and the
    # final flush, and capture callback failures from cancelled close waiters.
    initial_tasks = asyncio.all_tasks()
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    token = CancellationToken()
    task = asyncio.create_task(
        LocalProcessExecutor().run(
            ProcessExecutionRequest(
                argv=(sys.executable, "-c", code, str(pid_file), stop),
                cwd=str(tmp_path),
                stdin=b"x" * input_bytes,
                timeout_seconds=1 if stop == "timeout" else None,
            ),
            token,
        )
    )
    try:
        # Wait for the inherited handle before testing either cancellation path.
        async with asyncio.timeout(3):
            while not pid_file.exists() or not pid_file.read_text():
                await asyncio.sleep(0.01)

        if stop == "token":
            token.cancel()
        elif stop == "task":
            task.cancel()

        # Observe completion without cancelling a hung executor: wait_for would
        # itself wait indefinitely if cancellation cleanup regresses.
        done, _ = await asyncio.wait({task}, timeout=3)
        assert task in done
        if stop == "task":
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert (result.stop_reason == ProcessStopReason.TIMEOUT) == (
                stop == "timeout"
            )
            assert (result.stop_reason == ProcessStopReason.CANCELLED) == (
                stop == "token"
            )
            assert result.exit_code is not None
            if stop == "exit":
                assert result.exit_code == 0

        assert asyncio.all_tasks() == initial_tasks
        assert unhandled == []
    finally:
        loop.set_exception_handler(previous_handler)

        # The detached process is outside executor ownership. Close its inherited
        # pipe even on assertion failure, so the test cannot leave cleanup hung.
        if pid_file.exists():
            with suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
        if not task.done():
            token.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [b"", b"raw\x00\xff\n" * 20_000])
@pytest.mark.parametrize("exit_code", [0, 7])
async def test_local_executor_captures_exact_stdout_separately_from_stderr(
    tmp_path: Path, data: bytes, exit_code: int
) -> None:
    executor = LocalProcessExecutor()
    code = (
        "import sys; "
        "sys.stdout.buffer.write(sys.stdin.buffer.read()); "
        "sys.stderr.write('diagnostic\\n'); "
        f"sys.exit({exit_code})"
    )
    result = await executor.run(
        ProcessExecutionRequest(
            argv=(sys.executable, "-c", code),
            cwd=str(tmp_path),
            stdin=data,
            stdout_capture_limit=max(1, len(data)),
            timeout_seconds=3,
        ),
        CancellationToken(),
    )

    try:
        assert result.stdout == data
        assert result.stdout_complete
        assert result.exit_code == exit_code
        assert result.stop_reason != ProcessStopReason.TIMEOUT
        if data:
            assert result.output.truncated
            assert result.log_path is not None
            assert b"diagnostic\n" in Path(result.log_path).read_bytes()
            assert f"Full output: {result.log_path}" in format_process_output(result)
        else:
            assert result.output.text == "diagnostic\n"
    finally:
        if result.log_path is not None:
            Path(result.log_path).unlink(missing_ok=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1])
async def test_local_executor_rejects_invalid_stdout_capture_limit_before_startup(
    tmp_path: Path, limit: int
) -> None:
    marker = tmp_path / "must-not-exist"
    with pytest.raises(
        ValueError, match="stdout_capture_limit must be greater than zero"
    ):
        await LocalProcessExecutor().run(
            ProcessExecutionRequest(
                argv=(
                    sys.executable,
                    "-c",
                    "import pathlib, sys; pathlib.Path(sys.argv[1]).touch()",
                    str(marker),
                ),
                cwd=str(tmp_path),
                stdout_capture_limit=limit,
            ),
            CancellationToken(),
        )

    assert not marker.exists()


@pytest.mark.asyncio
async def test_local_executor_stops_at_capture_limit_and_marks_log_partial(
    tmp_path: Path,
) -> None:
    initial_tasks = asyncio.all_tasks()
    code = "import os\nwhile True:\n    os.write(1, b'x' * 8192)\n"
    result = await asyncio.wait_for(
        LocalProcessExecutor().run(
            ProcessExecutionRequest(
                argv=(sys.executable, "-c", code),
                cwd=str(tmp_path),
                stdout_capture_limit=100_000,
            ),
            CancellationToken(),
        ),
        timeout=3,
    )

    try:
        assert result.stdout == b"x" * 100_000
        assert not result.stdout_complete
        assert result.exit_code is not None
        assert result.stop_reason != ProcessStopReason.TIMEOUT
        assert result.stop_reason != ProcessStopReason.CANCELLED
        assert asyncio.all_tasks() == initial_tasks
        assert result.log_path is not None
        assert f"Partial output: {result.log_path}" in format_process_output(result)
    finally:
        if result.log_path is not None:
            Path(result.log_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_local_executor_marks_capture_incomplete_without_stdout_eof(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("requires POSIX fork semantics")

    # The leader succeeds, but its descendant retains stdout. Cleanup must not
    # mistake the retained prefix for a complete result that can replace a file.
    code = (
        "import os, time\n"
        "if os.fork() == 0:\n"
        "    time.sleep(30)\n"
        "else:\n"
        "    print('prefix', flush=True)\n"
        "    os._exit(0)\n"
    )
    result = await asyncio.wait_for(
        LocalProcessExecutor().run(
            ProcessExecutionRequest(
                argv=(sys.executable, "-c", code),
                cwd=str(tmp_path),
                stdout_capture_limit=1024,
            ),
            CancellationToken(),
        ),
        timeout=3,
    )

    assert result.exit_code == 0
    assert result.stdout == b"prefix\n"
    assert not result.stdout_complete


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["timeout", "token", "task"])
async def test_local_executor_cleans_up_stdout_capture_when_stopped(
    tmp_path: Path, stop: str
) -> None:
    initial_tasks = asyncio.all_tasks()
    token = CancellationToken()
    task = asyncio.create_task(
        LocalProcessExecutor().run(
            ProcessExecutionRequest(
                argv=(
                    sys.executable,
                    "-c",
                    "import time; print('prefix', end='', flush=True); time.sleep(5)",
                ),
                cwd=str(tmp_path),
                stdout_capture_limit=1024,
                timeout_seconds=0.1 if stop == "timeout" else None,
            ),
            token,
        )
    )
    if stop != "timeout":
        await asyncio.sleep(0.1)
        if stop == "token":
            token.cancel()
        else:
            task.cancel()

    if stop == "task":
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)
    else:
        result = await asyncio.wait_for(task, timeout=3)
        assert result.stdout == b"prefix"
        assert (result.stop_reason == ProcessStopReason.TIMEOUT) == (stop == "timeout")
        assert (result.stop_reason == ProcessStopReason.CANCELLED) == (stop == "token")

    assert asyncio.all_tasks() == initial_tasks


def test_native_process_windows_graceful_signal_uses_process_group_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWindowsProcess:
        pid = 123
        returncode: int | None = None

        def __init__(self) -> None:
            self.signals: list[object] = []
            self.terminated = False

        def send_signal(self, sig: object) -> None:
            self.signals.append(sig)

        def terminate(self) -> None:
            self.terminated = True

    ctrl_break_event = object()
    process = FakeWindowsProcess()

    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(signal, "CTRL_BREAK_EVENT", ctrl_break_event, raising=False)

    process_executor = importlib.import_module(
        "agentlane.harness.tools._process_executor"
    )
    helper_name = "_send_process_signal"
    send_process_signal = getattr(process_executor, helper_name)
    send_process_signal(cast(Any, process), signal.SIGTERM)

    assert process.signals == [ctrl_break_event]
    assert process.terminated is False


@pytest.mark.asyncio
async def test_native_process_waits_for_child_after_output_streams_close(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "finished"
    result = await LocalProcessExecutor().run(
        ProcessExecutionRequest(
            argv=(
                sys.executable,
                "-c",
                "import os, pathlib, sys, time; os.close(1); os.close(2); "
                "time.sleep(0.1); pathlib.Path(sys.argv[1]).write_text('done')",
                str(marker),
            ),
            cwd=str(tmp_path),
            stdout_capture_limit=1024,
            timeout_seconds=2,
        ),
        CancellationToken(),
    )

    assert result.stop_reason == ProcessStopReason.EXITED
    assert result.exit_code == 0
    assert result.stdout == b""
    assert result.stdout_complete
    assert result.log_complete
    assert marker.read_text() == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [3, 4])
async def test_native_process_distinguishes_exact_capture_limit_from_one_byte_over(
    tmp_path: Path, size: int
) -> None:
    result = await LocalProcessExecutor().run(
        ProcessExecutionRequest(
            argv=(sys.executable, "-c", f"import os; os.write(1, b'x' * {size})"),
            cwd=str(tmp_path),
            stdout_capture_limit=3,
        ),
        CancellationToken(),
    )

    assert result.stdout == b"xxx"
    assert result.stdout_complete == (size == 3)
    assert result.stop_reason == (
        ProcessStopReason.EXITED if size == 3 else ProcessStopReason.STDOUT_LIMIT
    )
    # A short overflowing stream can still be logged completely.
    assert result.log_complete


@pytest.mark.asyncio
@pytest.mark.parametrize("request_env", [None, {}, {"AGENTLANE_VALUE": "request"}])
async def test_native_process_environment_defaults_are_snapshotted_and_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request_env: dict[str, str] | None
) -> None:
    monkeypatch.setenv("AGENTLANE_VALUE", "inherited")
    monkeypatch.setenv("AGENTLANE_HOST_ONLY", "private")
    configured = {"AGENTLANE_VALUE": "configured"}
    executor = LocalProcessExecutor(default_env=configured)
    configured["AGENTLANE_VALUE"] = "mutated"
    result = await executor.run(
        ProcessExecutionRequest(
            argv=(
                sys.executable,
                "-c",
                "import os; print(os.getenv('AGENTLANE_VALUE')); "
                "print(os.getenv('AGENTLANE_HOST_ONLY'))",
            ),
            cwd=str(tmp_path),
            env=request_env,
        ),
        CancellationToken(),
    )

    expected = (
        "configured" if request_env is None else request_env.get("AGENTLANE_VALUE")
    )
    assert result.output.text == f"{expected}\nNone\n"
    assert result.stdout is None
    assert not result.stdout_complete


@pytest.mark.asyncio
async def test_native_process_pre_cancelled_token_does_not_start_child(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "must-not-exist"
    token = CancellationToken()
    token.cancel()
    with pytest.raises(asyncio.CancelledError):
        await LocalProcessExecutor().run(
            ProcessExecutionRequest(
                argv=(
                    sys.executable,
                    "-c",
                    "import pathlib, sys; pathlib.Path(sys.argv[1]).touch()",
                    str(marker),
                ),
                cwd=str(tmp_path),
            ),
            token,
        )

    assert not marker.exists()


@pytest.mark.asyncio
async def test_native_process_repeated_cancellation_settles_child_and_transports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != "posix":
        pytest.skip("requires POSIX signal handlers")

    actual_start = asyncio.create_subprocess_exec
    processes: list[asyncio.subprocess.Process] = []

    async def capture_process(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await actual_start(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_process)
    initial_tasks = asyncio.all_tasks()
    marker = tmp_path / "ready"
    task = asyncio.create_task(
        LocalProcessExecutor().run(
            ProcessExecutionRequest(
                argv=(
                    sys.executable,
                    "-c",
                    "import pathlib, signal, sys, time; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "pathlib.Path(sys.argv[1]).touch(); time.sleep(30)",
                    str(marker),
                ),
                cwd=str(tmp_path),
                stdin=b"x" * 1_000_000,
            ),
            CancellationToken(),
        )
    )
    try:
        async with asyncio.timeout(3):
            while not marker.exists():
                await asyncio.sleep(0.01)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0.03)

        done, _ = await asyncio.wait({task}, timeout=3)
        assert task in done
        with pytest.raises(asyncio.CancelledError):
            await task
        assert processes[0].returncode == -signal.SIGKILL
        assert cast(Any, processes[0])._transport.is_closing()
        assert asyncio.all_tasks() == initial_tasks
    finally:
        if processes and processes[0].returncode is None:
            processes[0].kill()
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_native_process_cancellation_during_startup_retains_child_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual_start = asyncio.create_subprocess_exec
    started = asyncio.Event()
    release = asyncio.Event()
    processes: list[asyncio.subprocess.Process] = []

    async def delayed_start(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await actual_start(*args, **kwargs)
        processes.append(process)
        started.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_start)
    initial_tasks = asyncio.all_tasks()
    task = asyncio.create_task(
        LocalProcessExecutor().run(
            ProcessExecutionRequest(
                argv=(sys.executable, "-c", "import time; time.sleep(30)"),
                cwd=str(tmp_path),
            ),
            CancellationToken(),
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=3)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)

        assert processes[0].returncode is not None
        assert cast(Any, processes[0])._transport.is_closing()
        assert asyncio.all_tasks() == initial_tasks
    finally:
        release.set()
        if processes and processes[0].returncode is None:
            processes[0].kill()
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_native_process_stream_sink_failure_stops_child_and_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual_start = asyncio.create_subprocess_exec
    processes: list[asyncio.subprocess.Process] = []

    async def capture_process(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await actual_start(*args, **kwargs)
        processes.append(process)
        return process

    async def fail_write(self: object, chunk: bytes) -> None:
        del self, chunk
        raise OSError("output sink failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_process)
    monkeypatch.setattr(
        "agentlane.harness.tools._process_executor._CombinedOutputRecorder.write",
        fail_write,
    )
    initial_tasks = asyncio.all_tasks()
    with pytest.raises(OSError, match="output sink failed"):
        await asyncio.wait_for(
            LocalProcessExecutor().run(
                ProcessExecutionRequest(
                    argv=(
                        sys.executable,
                        "-c",
                        "import os\nwhile True: os.write(1, b'x' * 8192)",
                    ),
                    cwd=str(tmp_path),
                ),
                CancellationToken(),
            ),
            timeout=3,
        )

    assert processes[0].returncode is not None
    assert cast(Any, processes[0])._transport.is_closing()
    assert asyncio.all_tasks() == initial_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("argv", [(), ("",), (sys.executable, "bad\x00argument")])
async def test_native_process_rejects_invalid_argv_before_startup(
    tmp_path: Path, argv: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unexpected_start(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        del args, kwargs
        raise AssertionError("must validate before startup")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected_start)
    with pytest.raises(ValueError, match="argv"):
        await LocalProcessExecutor().run(
            ProcessExecutionRequest(argv=argv, cwd=str(tmp_path)), CancellationToken()
        )


@pytest.mark.asyncio
async def test_native_process_queued_token_cancellation_prevents_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = False

    async def unexpected_start(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        nonlocal started
        del args, kwargs
        started = True
        raise AssertionError("cancelled startup must not create a child")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected_start)
    initial_tasks = asyncio.all_tasks()
    token = CancellationToken()
    task = asyncio.create_task(
        LocalProcessExecutor().run(
            ProcessExecutionRequest(
                argv=(sys.executable, "-c", "pass"), cwd=str(tmp_path)
            ),
            token,
        )
    )
    asyncio.get_running_loop().call_soon(token.cancel)
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not started
    assert asyncio.all_tasks() == initial_tasks
