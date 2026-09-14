"""Transport failures must end the session without a second terminal outcome."""

import asyncio
import os
import subprocess
import sys
import threading
from io import StringIO
from pathlib import Path

import pytest
from agentlane_process_bridge import (
    BridgeBackend,
    BridgeEventType,
    EventWriter,
    PromptCommand,
    ResetCommand,
    serve_stdio,
)
from structlog.testing import capture_logs

from agentlane.harness import RunResult

from .helpers import FakeAgent, emitted_events, wait_for_stream


@pytest.mark.parametrize("reset", [False, True])
def test_completion_owns_terminal_during_flush(reset: bool) -> None:
    class GatedOutput(StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()
            self.block = False

        def write(self, value: str) -> int:
            if '"type":"run_complete"' in value:
                self.block = True
            return super().write(value)

        def flush(self) -> None:
            if self.block:
                self.entered.set()
                self.release.wait(2)
                self.block = False
            super().flush()

    async def scenario() -> None:
        output = GatedOutput()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))
        await backend.handle_command(PromptCommand(text="go"))
        stream = await wait_for_stream(agent)
        stream.finish(RunResult(final_output="done", responses=[], turn_count=1))
        while not output.entered.is_set():
            await asyncio.sleep(0.001)
        try:
            if reset:
                command = asyncio.create_task(backend.handle_command(ResetCommand()))
            else:
                backend.request_active_run_cancel(emit_terminal=True)
                command = None
            await asyncio.sleep(0.01)
        finally:
            output.release.set()
        if command is not None:
            await command
        while backend.has_active_run():
            await asyncio.sleep(0.001)
        terminal = [
            event["type"]
            for event in emitted_events(output)
            if event["type"] in {"run_complete", "run_cancelled", "error"}
        ]
        assert terminal == ["run_complete"]
        await backend.close()

    asyncio.run(scenario())


_CHILD_SCRIPT = """
import asyncio
import os
from pathlib import Path
from unittest.mock import patch
from agentlane_process_bridge import AgentBackend, EventWriter
from agentlane.harness import RunModelStreamEvent
from agentlane.models import ModelStreamEvent, ModelStreamEventKind
from packages.process_bridge.tests.helpers import FakeAgent

class Agent(FakeAgent):
    async def run_events(self, *args, **kwargs):
        stream = await super().run_events(*args, **kwargs)
        original_close = stream.aclose
        async def close():
            await original_close()
            assert self.cancellation_tokens[0].is_cancelled
            Path(os.environ["BRIDGE_TEST_MARKER"]).write_text("closed")
        stream.aclose = close
        async def produce():
            await asyncio.sleep(0.1)
            size = 1 if os.environ["BRIDGE_TEST_PIPE"] == "broken" else 4000000
            stream.emit(RunModelStreamEvent(event=ModelStreamEvent(kind=ModelStreamEventKind.TEXT_DELTA, text="x" * size)))
        self.producer = asyncio.create_task(produce())
        return stream

def create_backend():
    patch("agentlane_process_bridge._stdio.EventWriter", side_effect=lambda output: EventWriter(output, write_timeout_seconds=0.1)).start()
    return AgentBackend(agent=Agent())
"""


@pytest.mark.parametrize("broken", [False, True])
def test_subprocess_exits_after_writer_failure_with_stdin_open(
    broken: bool, tmp_path: Path
) -> None:
    app = tmp_path / "writer_failure_app.py"
    app.write_text(_CHILD_SCRIPT)
    marker = tmp_path / "closed"
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(tmp_path), str(Path.cwd())]),
        "BRIDGE_TEST_MARKER": str(marker),
        "BRIDGE_TEST_PIPE": "broken" if broken else "slow",
    }
    # The interpreter and generated app are controlled by this test.
    process = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "agentlane_process_bridge",
            "--app",
            "writer_failure_app:create_backend",
        ],
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        assert b'"type":"ready"' in process.stdout.readline()
        process.stdin.write(b'{"protocol_version":"1.0","type":"prompt","text":"go"}\n')
        process.stdin.flush()
        assert b'"type":"run_start"' in process.stdout.readline()
        if broken:
            process.stdout.close()
        # Keep stdin open. The child must end without a further command or EOF.
        assert process.wait(timeout=3) == 0
        assert marker.read_text() == "closed"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        process.stdin.close()
        process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


@pytest.mark.parametrize("invalid_unicode", [False, True])
def test_completion_write_failure_marks_writer_and_never_succeeds(
    invalid_unicode: bool,
) -> None:
    class InvalidOutput(StringIO):
        def write(self, value: str) -> int:
            if '"type":"run_complete"' in value:
                if invalid_unicode:
                    value.encode("utf-8")
                raise ValueError("closed borrowed stream")
            return super().write(value)

    async def scenario() -> None:
        output = InvalidOutput()
        agent = FakeAgent()
        writer = EventWriter(output)
        backend = BridgeBackend(agent=agent, events=writer)
        await backend.handle_command(PromptCommand(text="go"))
        stream = await wait_for_stream(agent)
        stream.finish(
            RunResult(
                final_output="\ud800" if invalid_unicode else "done",
                responses=[],
                turn_count=1,
            )
        )
        await asyncio.wait_for(writer.wait_failed(), 1)
        while backend.has_active_run():
            await asyncio.sleep(0.001)
        assert [event["type"] for event in emitted_events(output)] == ["run_start"]
        assert not writer.is_writable
        assert stream.aclose_calls == 1
        assert agent.cancellation_tokens[0] is not None
        assert agent.cancellation_tokens[0].is_cancelled
        with pytest.raises((ValueError, UnicodeEncodeError)):
            await backend.close()

    asyncio.run(scenario())


def test_timed_out_batch_does_not_send_remaining_completion() -> None:
    class GatedOutput(StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()
            self.returned = threading.Event()
            self.flush_count = 0

        def write(self, value: str) -> int:
            if not self.entered.is_set():
                self.entered.set()
                self.release.wait(2)
            return super().write(value)

        def flush(self) -> None:
            self.flush_count += 1
            super().flush()

    async def scenario() -> None:
        output = GatedOutput()

        class TrackingWriter(EventWriter):
            def _write_lines(self, lines: list[str]) -> None:
                try:
                    assert len(lines) == 2
                    super()._write_lines(lines)
                finally:
                    output.returned.set()

        writer = TrackingWriter(output, write_timeout_seconds=0.01)
        first = asyncio.create_task(
            writer.emit(
                BridgeEventType.RUN_EVENT,
                verbatim_payload={
                    "event": {
                        "type": "model_stream",
                        "payload": {
                            "kind": "model_stream",
                            "event": {"kind": "text_delta", "text": "first"},
                        },
                    }
                },
            )
        )
        completion = asyncio.create_task(
            writer.emit(BridgeEventType.RUN_COMPLETE, final_output="done")
        )
        try:
            await first
            with pytest.raises((TimeoutError, BrokenPipeError)):
                await completion
            await writer.wait_failed()
        finally:
            output.release.set()
        for _ in range(100):
            if output.returned.is_set():
                break
            await asyncio.sleep(0.001)
        assert output.returned.is_set()
        assert [event["type"] for event in emitted_events(output)] == ["run_event"]
        assert output.flush_count == 0
        with pytest.raises(BrokenPipeError):
            await writer.aclose()

    asyncio.run(scenario())


def test_dead_client_close_logs_error_type_without_rendering_payload() -> None:
    class FailedOutput(StringIO):
        def write(self, value: str) -> int:
            raise BrokenPipeError("reader disconnected")

    async def scenario() -> None:
        backend = BridgeBackend(agent=FakeAgent(), events=EventWriter(FailedOutput()))
        with pytest.raises(BrokenPipeError):
            await backend.start()
        with capture_logs() as logs:
            await serve_stdio(backend, readline=lambda _size: "")
        assert logs == [
            {
                "event": "bridge_close_after_dead_client_failed",
                "log_level": "error",
                "error_type": "BrokenPipeError",
            }
        ]

    asyncio.run(scenario())
