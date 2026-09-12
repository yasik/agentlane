"""Strict delivery failures must settle one run and release its resources."""

import asyncio
from io import StringIO
from pathlib import Path

import pytest
from agentlane_process_bridge import (
    BridgeBackend,
    CancelCommand,
    EventWriter,
    PromptCommand,
    ResetCommand,
)

from agentlane.harness import RunAgentEndEvent, RunModelStreamEvent, RunResult, RunState
from agentlane.harness.tools import (
    ToolOperation,
    ToolPermissionDecision,
    ToolPermissionRequest,
)
from agentlane.models import ModelResponse, ModelStreamEvent, ModelStreamEventKind

from .helpers import FakeAgent, emitted_events, wait_for_event_count, wait_for_stream


@pytest.mark.parametrize("location", ["event", "output", "state", "response"])
@pytest.mark.parametrize("failure", ["unsupported", "cycle", "nonfinite", "key"])
def test_strict_delivery_failures_cancel_close_and_never_complete(
    location: str, failure: str
) -> None:
    value: object
    if failure == "cycle":
        items: list[object] = []
        items.append(items)
        value = items
    else:
        value = {"unsupported": object(), "nonfinite": float("inf"), "key": {1: "bad"}}[
            failure
        ]

    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))
        await backend.handle_command(PromptCommand(text="go"))
        stream = await wait_for_stream(agent)
        result = RunResult(final_output="ok", responses=[], turn_count=1)
        if location == "event":
            stream.emit(
                RunModelStreamEvent(
                    event=ModelStreamEvent(
                        kind=ModelStreamEventKind.PROVIDER, raw=value
                    )
                )
            )
        elif location == "output":
            result.final_output = value
        elif location == "state":
            state = RunState(instructions=None, history=[], responses=[])
            state.shim_state["invalid"] = value
            result.run_state = state
        else:
            # Exercise strict conversion in a response field the completion
            # envelope does not expose, without changing model validation.
            response = ModelResponse.model_validate(
                {
                    "id": "r",
                    "created": 0,
                    "model": "test",
                    "object": "chat.completion",
                    "choices": [],
                    "invalid": value,
                }
            )
            result.responses.append(response)
        stream.finish(result)
        events = await wait_for_event_count(output, 2)
        assert [event["type"] for event in events] == ["run_start", "error"]
        assert events[-1]["scope"] == "run"
        assert stream.aclose_calls == 1
        token = agent.cancellation_tokens[0]
        assert token is not None and token.is_cancelled
        await backend.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("command", [CancelCommand(), ResetCommand()])
def test_failure_cleanup_denies_approvals_before_close_and_survives_commands(
    monkeypatch: pytest.MonkeyPatch, command: CancelCommand | ResetCommand
) -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))
        await backend.handle_command(PromptCommand(text="go"))
        stream = await wait_for_stream(agent)
        waiter = asyncio.create_task(
            backend.approvals.callback(
                ToolPermissionRequest(
                    tool_name="write",
                    operation=ToolOperation.CREATE_FILE,
                    cwd=Path("/workspace"),
                ),
                ToolPermissionDecision.require_approval(),
            )
        )
        while not backend.approvals.pending():
            await asyncio.sleep(0)
        closing = asyncio.Event()
        release = asyncio.Event()
        original_close = stream.aclose

        async def close() -> None:
            decision = await waiter
            assert not decision.allowed
            closing.set()
            await release.wait()
            await original_close()

        monkeypatch.setattr(stream, "aclose", close)
        stream.emit(
            RunModelStreamEvent(
                event=ModelStreamEvent(kind=ModelStreamEventKind.PROVIDER, raw=object())
            )
        )
        await asyncio.wait_for(closing.wait(), 1)
        command_task = asyncio.create_task(backend.handle_command(command))
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(command_task, 1)
        events = await wait_for_event_count(output, 3)
        terminal = [
            event
            for event in events
            if event["type"] in {"error", "run_complete", "run_cancelled"}
        ]
        assert len(terminal) == 1 and terminal[0]["type"] == "error"
        assert backend.approvals.pending() == ()
        await backend.close()

    asyncio.run(scenario())


def test_error_after_root_agent_end_uses_authoritative_stream_result() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))
        await backend.handle_command(PromptCommand(text="go"))
        stream = await wait_for_stream(agent)
        stream.emit(RunAgentEndEvent(task_name="root", task_id="root", result=None))
        stream.fail(RuntimeError("authoritative failure"))
        events = await wait_for_event_count(output, 3)
        assert [event["type"] for event in events] == [
            "run_start",
            "run_event",
            "error",
        ]
        assert events[-1]["message"] == "authoritative failure"
        assert stream.result_awaits == 1
        await backend.close()

    asyncio.run(scenario())


def test_result_without_state_keeps_legacy_runtime_metadata_fallback() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        state = RunState(instructions=None, history=[], responses=[])
        state.shim_state["nonfinite"] = float("nan")
        agent.run_state = state
        backend = BridgeBackend(agent=agent, events=EventWriter(output))
        await backend.handle_command(PromptCommand(text="go"))
        stream = await wait_for_stream(agent)
        stream.finish(RunResult(final_output="done", responses=[], turn_count=1))
        await wait_for_event_count(output, 2)
        assert emitted_events(output)[-1]["shim_state"] == {"nonfinite": "nan"}
        await backend.close()

    asyncio.run(scenario())


def test_failed_close_does_not_wait_for_unresolved_result() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))
        await backend.handle_command(PromptCommand(text="go"))
        stream = await wait_for_stream(agent)
        stream.close_error = RuntimeError("close failed before result settled")
        stream.emit(
            RunModelStreamEvent(
                event=ModelStreamEvent(kind=ModelStreamEventKind.PROVIDER, raw=object())
            )
        )
        events = await asyncio.wait_for(wait_for_event_count(output, 2), 1)
        assert events[-1]["type"] == "error"
        assert "Unsupported event value" in events[-1]["message"]
        assert (
            "cleanup failed: close failed before result settled"
            in events[-1]["message"]
        )
        assert stream.result_awaits == 1
        await backend.close()

    asyncio.run(scenario())
