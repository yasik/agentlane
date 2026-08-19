"""Tests for the addressed Claude Agent SDK task."""

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import agentlane_claude_agent_sdk._agent as agent_module
import pytest
from agentlane_claude_agent_sdk import ClaudeAgent
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKError, ResultMessage

from agentlane.harness import Task
from agentlane.messaging import AgentId, DeliveryOutcome, DeliveryStatus
from agentlane.runtime import SingleThreadedRuntimeEngine


def _result(
    result: Any,
    *,
    subtype: str = "success",
    is_error: bool = False,
) -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id="session-1",
        result=result,
    )


async def _deliver(
    agent_id: AgentId,
    payload: str,
    *,
    options: ClaudeAgentOptions | None = None,
) -> DeliveryOutcome:
    runtime = SingleThreadedRuntimeEngine()
    ClaudeAgent.bind(runtime, agent_id, options=options)
    outcome = await runtime.send_message(payload, recipient=agent_id)
    await runtime.stop_when_idle()
    return outcome


def _install_query(
    monkeypatch: pytest.MonkeyPatch,
    stream_factory: Callable[[], AsyncIterator[object]],
) -> list[tuple[str, ClaudeAgentOptions]]:
    calls: list[tuple[str, ClaudeAgentOptions]] = []

    def fake_query(
        *,
        prompt: str,
        options: ClaudeAgentOptions,
    ) -> AsyncIterator[object]:
        calls.append((prompt, options))
        return stream_factory()

    monkeypatch.setattr(agent_module, "query", fake_query)
    return calls


def test_claude_agent_is_one_concrete_task() -> None:
    assert issubclass(ClaudeAgent, Task)


def test_claude_agent_returns_last_success_result_after_full_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_exhausted = False

    async def stream() -> AsyncIterator[object]:
        nonlocal stream_exhausted
        yield object()
        yield _result("ignored", subtype="error_during_execution", is_error=True)
        yield _result("Claude result")
        yield object()
        stream_exhausted = True

    calls = _install_query(monkeypatch, stream)
    agent_id = AgentId.from_values("claude-agent", "coworker")

    outcome = asyncio.run(_deliver(agent_id, "Do the subtask."))

    assert outcome.status == DeliveryStatus.DELIVERED
    assert outcome.response_payload == "Claude result"
    assert stream_exhausted
    assert [prompt for prompt, _options in calls] == ["Do the subtask."]


def test_claude_agent_starts_one_fresh_query_for_each_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stream() -> AsyncIterator[object]:
        yield _result("done")

    calls = _install_query(monkeypatch, stream)

    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        agent_id = AgentId.from_values("claude-agent", "coworker")
        ClaudeAgent.bind(runtime, agent_id)

        first = await runtime.send_message("one", recipient=agent_id)
        second = await runtime.send_message("two", recipient=agent_id)
        await runtime.stop_when_idle()

        assert first.status == DeliveryStatus.DELIVERED
        assert second.status == DeliveryStatus.DELIVERED

    asyncio.run(scenario())

    assert [prompt for prompt, _options in calls] == ["one", "two"]
    assert len(calls) == 2
    assert all(not options.continue_conversation for _prompt, options in calls)
    assert all(options.resume is None for _prompt, options in calls)


def test_claude_agent_passes_explicit_options_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stream() -> AsyncIterator[object]:
        yield _result("done")

    calls = _install_query(monkeypatch, stream)
    options = ClaudeAgentOptions(model="claude-sonnet-4-5", max_turns=2)
    agent_id = AgentId.from_values("claude-agent", "coworker")

    outcome = asyncio.run(_deliver(agent_id, "hello", options=options))

    assert outcome.status == DeliveryStatus.DELIVERED
    assert calls == [("hello", options)]
    assert calls[0][1] is options


def test_claude_agent_uses_isolated_tool_free_default_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stream() -> AsyncIterator[object]:
        yield _result("done")

    calls = _install_query(monkeypatch, stream)
    agent_id = AgentId.from_values("claude-agent", "coworker")

    outcome = asyncio.run(_deliver(agent_id, "hello"))

    assert outcome.status == DeliveryStatus.DELIVERED
    [(_prompt, options)] = calls
    assert options.tools == []
    assert options.setting_sources == []
    assert options.skills == []
    assert options.mcp_servers == {}
    assert options.strict_mcp_config is True
    assert options.max_turns == 1


@pytest.mark.parametrize(
    "options",
    [
        ClaudeAgentOptions(continue_conversation=True),
        ClaudeAgentOptions(resume="session-1"),
        ClaudeAgentOptions(fork_session=True),
        ClaudeAgentOptions(resume_session_at="message-1"),
        ClaudeAgentOptions(resume_drops_turn="prompt-1"),
        ClaudeAgentOptions(extra_args={"continue": None}),
        ClaudeAgentOptions(extra_args={"resume": "session-1"}),
        ClaudeAgentOptions(extra_args={"fork-session": None}),
        ClaudeAgentOptions(extra_args={"resume-session-at": "message-1"}),
        ClaudeAgentOptions(extra_args={"resume-drops-turn": "prompt-1"}),
    ],
    ids=[
        "continue-conversation",
        "resume",
        "fork-session",
        "resume-session-at",
        "resume-drops-turn",
        "extra-continue",
        "extra-resume",
        "extra-fork-session",
        "extra-resume-session-at",
        "extra-resume-drops-turn",
    ],
)
def test_claude_agent_rejects_continuity_options_before_query(
    monkeypatch: pytest.MonkeyPatch,
    options: ClaudeAgentOptions,
) -> None:
    async def stream() -> AsyncIterator[object]:
        raise AssertionError("query must not start")
        yield

    calls = _install_query(monkeypatch, stream)
    agent_id = AgentId.from_values("claude-agent", "coworker")

    outcome = asyncio.run(_deliver(agent_id, "hello", options=options))

    assert outcome.status == DeliveryStatus.HANDLER_ERROR
    assert outcome.error is not None
    assert outcome.error.message == "ClaudeAgent options must start a fresh session."
    assert calls == []


def test_claude_agent_preserves_sdk_exception_as_handler_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stream() -> AsyncIterator[object]:
        raise ClaudeSDKError("SDK exploded")
        yield

    _install_query(monkeypatch, stream)
    agent_id = AgentId.from_values("claude-agent", "coworker")

    outcome = asyncio.run(_deliver(agent_id, "hello"))

    assert outcome.status == DeliveryStatus.HANDLER_ERROR
    assert outcome.error is not None
    assert outcome.error.message == "SDK exploded"


@pytest.mark.parametrize(
    "messages",
    [
        [_result("partial", subtype="error_during_execution", is_error=True)],
        [object()],
        [_result(None)],
        [_result({"answer": "not text"})],
    ],
    ids=["error-terminal", "no-result-message", "none-result", "non-string-result"],
)
def test_claude_agent_rejects_stream_without_valid_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[object],
) -> None:
    async def stream() -> AsyncIterator[object]:
        for message in messages:
            yield message

    _install_query(monkeypatch, stream)
    agent_id = AgentId.from_values("claude-agent", "coworker")

    outcome = asyncio.run(_deliver(agent_id, "hello"))

    assert outcome.status == DeliveryStatus.HANDLER_ERROR
    assert outcome.error is not None
    assert outcome.error.message == "Claude query did not return a successful string."


def test_claude_agent_accepts_empty_string_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stream() -> AsyncIterator[object]:
        yield _result("")

    _install_query(monkeypatch, stream)
    agent_id = AgentId.from_values("claude-agent", "coworker")

    outcome = asyncio.run(_deliver(agent_id, "hello"))

    assert outcome.status == DeliveryStatus.DELIVERED
    assert outcome.response_payload == ""
