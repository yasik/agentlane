import asyncio
import time
from collections.abc import AsyncIterator, Callable
from io import StringIO
from pathlib import Path
from typing import TextIO, cast

import pytest
from agentlane_process_bridge import BridgeBackend, EventWriter, run_stdio, serve_stdio

from agentlane.harness.tools import (
    ToolApprovalEvent,
    ToolOperation,
    ToolPermissionDecision,
    ToolPermissionRequest,
)
from agentlane.runtime import CancellationToken

from .helpers import FakeAgent, FakeRunEventStream, emitted_events, wait_for_stream


class _PrintingAgent(FakeAgent):
    async def run_events(
        self,
        input: str,
        /,
        *,
        approval_events: AsyncIterator[ToolApprovalEvent],
        cancellation_token: CancellationToken | None = None,
    ) -> FakeRunEventStream:
        print("diagnostic from agent")
        return await super().run_events(
            input,
            approval_events=approval_events,
            cancellation_token=cancellation_token,
        )


def _read_lines(lines: list[str]) -> Callable[[int], str]:
    input_lines = iter(lines)
    return lambda _limit: next(input_lines, "")


def test_serve_stdio_reports_bad_command_and_survives() -> None:
    async def scenario() -> None:
        output = StringIO()
        backend = BridgeBackend(agent=FakeAgent(), events=EventWriter(output))

        await serve_stdio(
            backend,
            readline=_read_lines(
                ["not-json\n", '{"protocol_version":"1.0","type":"shutdown"}\n']
            ),
        )

        events = emitted_events(output)
        assert [event["type"] for event in events] == ["error", "shutdown"]
        assert events[0]["scope"] == "command"

    asyncio.run(scenario())


def test_serve_stdio_rejects_invalid_command_shapes() -> None:
    async def scenario() -> None:
        output = StringIO()
        backend = BridgeBackend(agent=FakeAgent(), events=EventWriter(output))

        await serve_stdio(
            backend,
            readline=_read_lines(
                [
                    "[]\n",
                    "{}\n",
                    '{"type":"prompt","text":"missing version"}\n',
                    (
                        '{"protocol_version":"2.0","type":"prompt",'
                        '"text":"bad version"}\n'
                    ),
                    '{"protocol_version":"1.0","type":"shutdown"}\n',
                ]
            ),
        )

        events = emitted_events(output)
        assert [event["type"] for event in events] == [
            "error",
            "error",
            "error",
            "error",
            "shutdown",
        ]
        assert all(event["scope"] == "command" for event in events[:-1])

    asyncio.run(scenario())


def test_serve_stdio_denies_non_boolean_approval_values() -> None:
    async def scenario() -> None:
        output = StringIO()
        backend = BridgeBackend(agent=FakeAgent(), events=EventWriter(output))
        approval_task = asyncio.create_task(
            backend.approvals.callback(
                ToolPermissionRequest(
                    tool_name="write",
                    operation=ToolOperation.CREATE_FILE,
                    cwd=Path("/workspace"),
                ),
                ToolPermissionDecision.require_approval("needs review"),
            )
        )
        for _ in range(100):
            pending = backend.approvals.pending()
            if pending:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("Expected pending approval.")

        await serve_stdio(
            backend,
            readline=_read_lines(
                [
                    (
                        '{"protocol_version":"1.0","type":"approve",'
                        f'"id":"{pending[0].request_id}","allowed":"true"}}\n'
                    ),
                    '{"protocol_version":"1.0","type":"shutdown"}\n',
                ]
            ),
        )
        decision = await approval_task

        assert not decision.allowed

    asyncio.run(scenario())


def test_serve_stdio_reports_unknown_command_and_survives() -> None:
    async def scenario() -> None:
        output = StringIO()
        backend = BridgeBackend(agent=FakeAgent(), events=EventWriter(output))

        await serve_stdio(
            backend,
            readline=_read_lines(
                [
                    '{"protocol_version":"1.0","type":"future_command"}\n',
                    '{"protocol_version":"1.0","type":"shutdown"}\n',
                ]
            ),
        )

        events = emitted_events(output)
        assert [event["type"] for event in events] == ["error", "shutdown"]
        assert events[0]["scope"] == "command"
        assert events[0]["message"] == "Unknown command: future_command"

    asyncio.run(scenario())


def test_serve_stdio_rejects_oversized_command_line() -> None:
    async def scenario() -> None:
        output = StringIO()
        backend = BridgeBackend(agent=FakeAgent(), events=EventWriter(output))

        await serve_stdio(
            backend,
            readline=_read_lines(
                [
                    '{"protocol_version":"1.0","type":"prompt","text":"'
                    + ("x" * 64)
                    + '"}\n',
                    '{"protocol_version":"1.0","type":"shutdown"}\n',
                ]
            ),
            max_command_line_chars=48,
        )

        events = emitted_events(output)
        assert [event["type"] for event in events] == ["error", "shutdown"]
        assert events[0]["message"] == "Command line exceeds bridge size limit."

    asyncio.run(scenario())


def test_serve_stdio_discards_oversized_unterminated_command_line() -> None:
    async def scenario() -> None:
        output = StringIO()
        backend = BridgeBackend(agent=FakeAgent(), events=EventWriter(output))

        await serve_stdio(
            backend,
            readline=_read_lines(
                [
                    "x" * 96,
                    "tail\n",
                    '{"protocol_version":"1.0","type":"shutdown"}\n',
                ]
            ),
            max_command_line_chars=48,
        )

        events = emitted_events(output)
        assert [event["type"] for event in events] == ["error", "shutdown"]
        assert events[0]["message"] == "Command line exceeds bridge size limit."

    asyncio.run(scenario())


def test_serve_stdio_broken_stdout_during_error_report_closes_cleanly() -> None:
    class BrokenOutput(StringIO):
        def write(self, value: str) -> int:
            del value
            raise BrokenPipeError("stdout closed")

    async def scenario() -> None:
        backend = BridgeBackend(agent=FakeAgent(), events=EventWriter(BrokenOutput()))

        await serve_stdio(
            backend,
            readline=_read_lines(["not-json\n"]),
        )

    asyncio.run(scenario())


def test_serve_stdio_eof_closes_active_run_without_shutdown_event() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))
        sent_prompt = False

        def readline(_limit: int) -> str:
            nonlocal sent_prompt
            if not sent_prompt:
                sent_prompt = True
                return '{"protocol_version":"1.0","type":"prompt","text":"go"}\n'
            for _ in range(100):
                if agent.streams:
                    return ""
                time.sleep(0.01)
            return ""

        await serve_stdio(backend, readline=readline)
        stream = await wait_for_stream(agent)

        events = emitted_events(output)
        assert [event["type"] for event in events] == ["run_start"]
        assert stream.aclose_calls == 1
        assert stream.result_awaits == 1

    asyncio.run(scenario())


def test_serve_stdio_read_error_closes_active_run() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))
        sent_prompt = False

        def readline(_limit: int) -> str:
            nonlocal sent_prompt
            if not sent_prompt:
                sent_prompt = True
                return '{"protocol_version":"1.0","type":"prompt","text":"go"}\n'
            for _ in range(100):
                if agent.streams:
                    raise OSError("stdin closed")
                time.sleep(0.01)
            raise OSError("stdin closed")

        await serve_stdio(backend, readline=readline)
        stream = await wait_for_stream(agent)

        events = emitted_events(output)
        assert [event["type"] for event in events] == ["run_start"]
        assert stream.aclose_calls == 1
        assert stream.result_awaits == 1

    asyncio.run(scenario())


def test_run_stdio_routes_python_prints_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        protocol = StringIO()
        diagnostics = StringIO()
        agent = _PrintingAgent()
        monkeypatch.setattr("sys.stdout", protocol)
        monkeypatch.setattr("sys.stderr", diagnostics)

        class PromptThenEof:
            sent_prompt = False

            def readline(self, _limit: int) -> str:
                if not self.sent_prompt:
                    self.sent_prompt = True
                    return '{"protocol_version":"1.0","type":"prompt","text":"go"}\n'
                for _ in range(100):
                    if agent.streams:
                        return ""
                    time.sleep(0.01)
                return ""

        await run_stdio(agent=agent, stdin=cast(TextIO, PromptThenEof()))

        events = emitted_events(protocol)
        assert [event["type"] for event in events] == ["ready", "run_start"]
        assert diagnostics.getvalue() == "diagnostic from agent\n"

    asyncio.run(scenario())
