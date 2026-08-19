"""Tests for the native AgentLane to Claude coworker proof."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import agentlane_claude_agent_sdk._agent as agent_module
import pytest
from agentlane_claude_agent_sdk import ClaudeAgent
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage
from pydantic import BaseModel

from agentlane.harness import AgentDescriptor, Runner, Task
from agentlane.harness.agents import DefaultAgent
from agentlane.messaging import AgentId, DeliveryStatus
from agentlane.models import (
    MessageDict,
    Model,
    ModelResponse,
    Tool,
    ToolCall,
    Tools,
    as_tool,
)
from agentlane.runtime import (
    CancellationToken,
    Engine,
    MessageContext,
    RuntimeEngine,
    SingleThreadedRuntimeEngine,
    on_message,
)
from agentlane.tracing import Span

NATIVE_ID = AgentId.from_values("agentlane-native", "researcher")
CLAUDE_ID = AgentId.from_values("claude-sdk", "analyst")


def _build_ask_claude_tool(
    *,
    runtime: RuntimeEngine,
    native_id: AgentId,
    claude_id: AgentId,
) -> Tool[BaseModel, Any]:
    @as_tool
    async def ask_claude(
        task: str,
        cancellation_token: CancellationToken,
    ) -> str:
        """Ask the addressed Claude coworker to complete one text task."""
        outcome = await runtime.send_message(
            task,
            sender=native_id,
            recipient=claude_id,
            cancellation_token=cancellation_token,
        )
        if outcome.status != DeliveryStatus.DELIVERED:
            message = (
                outcome.error.message
                if outcome.error is not None
                else f"Claude delivery failed with status {outcome.status.value}."
            )
            raise RuntimeError(message)
        if not isinstance(outcome.response_payload, str):
            raise RuntimeError("Claude delivery returned a non-string response.")
        return outcome.response_payload

    return ask_claude


def _assistant_response(
    content: str | None,
    *,
    tool_calls: list[ToolCall] | None = None,
) -> ModelResponse:
    return ModelResponse.model_validate(
        {
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
                }
            ],
        }
    )


def _tool_call(*, task: str) -> ToolCall:
    return ToolCall.model_validate(
        {
            "id": "call_claude",
            "type": "function",
            "function": {
                "name": "ask_claude",
                "arguments": f'{{"task": "{task}"}}',
            },
        }
    )


class _SequenceModel(Model[ModelResponse]):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = responses
        self.calls: list[list[MessageDict]] = []

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: Tools | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        del extra_call_args, schema, tools, cancellation_token, parent_span, kwargs
        self.calls.append([dict(message) for message in messages])
        return self._responses.pop(0)


class _RecordingClaudeAgent(ClaudeAgent):
    def __init__(
        self,
        engine: Engine,
        options: ClaudeAgentOptions,
        observed_senders: list[AgentId | None],
    ) -> None:
        super().__init__(engine, options)
        self._observed_senders = observed_senders

    @on_message
    async def handle(self, payload: str, context: MessageContext) -> str:
        self._observed_senders.append(context.sender)
        return await super().handle(payload, context)


class _FailingTask(Task):
    @on_message
    async def handle(self, payload: str, context: MessageContext) -> str:
        del payload, context
        raise RuntimeError("Claude delivery failed.")


class _NonStringTask(Task):
    @on_message
    async def handle(self, payload: str, context: MessageContext) -> dict[str, str]:
        del payload, context
        return {"result": "not text"}


def test_native_agent_delegates_to_addressed_claude_and_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = "Return the test sentinel."
    claude_result = "CLAUDE_SENTINEL_42"
    final_answer = "AgentLane completed the work with Claude's result."
    stream_exhausted = False
    query_calls: list[tuple[str, ClaudeAgentOptions]] = []

    async def stream() -> AsyncIterator[object]:
        nonlocal stream_exhausted
        yield object()
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="session-1",
            result=claude_result,
        )
        yield object()
        stream_exhausted = True

    def fake_query(
        *,
        prompt: str,
        options: ClaudeAgentOptions,
    ) -> AsyncIterator[object]:
        query_calls.append((prompt, options))
        return stream()

    monkeypatch.setattr(agent_module, "query", fake_query)

    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine(worker_count=2)
        options = ClaudeAgentOptions(
            tools=[],
            setting_sources=[],
            skills=[],
            mcp_servers={},
            strict_mcp_config=True,
            max_turns=1,
        )
        observed_senders: list[AgentId | None] = []
        _RecordingClaudeAgent.bind(
            runtime,
            CLAUDE_ID,
            options=options,
            observed_senders=observed_senders,
        )
        model = _SequenceModel(
            [
                _assistant_response(content=None, tool_calls=[_tool_call(task=task)]),
                _assistant_response(content=final_answer),
            ]
        )
        native = DefaultAgent(
            runtime=runtime,
            runner=Runner(),
            agent_id=NATIVE_ID,
            descriptor=AgentDescriptor(
                name="Native researcher",
                model=model,
                instructions="Ask Claude once, then complete the work.",
                tools=Tools(
                    tools=[
                        _build_ask_claude_tool(
                            runtime=runtime,
                            native_id=NATIVE_ID,
                            claude_id=CLAUDE_ID,
                        )
                    ],
                    tool_choice="required",
                    tool_call_limits={"ask_claude": 1},
                ),
            ),
        )

        result = await native.run("Complete the research task.")

        assert result.final_output == final_answer
        assert observed_senders == [NATIVE_ID]
        assert stream_exhausted
        assert query_calls == [(task, options)]
        assert options.tools == []
        assert options.setting_sources == []
        assert options.skills == []
        assert options.mcp_servers == {}
        assert options.strict_mcp_config is True
        assert options.max_turns == 1
        tool_results = [
            message["content"]
            for message in model.calls[1]
            if message.get("role") == "tool"
        ]
        assert tool_results == [claude_result]

    asyncio.run(scenario())


def test_relay_tool_raises_delivery_error() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine(worker_count=2)
        _FailingTask.bind(runtime, CLAUDE_ID)
        tool = _build_ask_claude_tool(
            runtime=runtime,
            native_id=NATIVE_ID,
            claude_id=CLAUDE_ID,
        )

        with pytest.raises(RuntimeError, match="Claude delivery failed\\."):
            await tool.run(
                tool.args_type()(task="Do work."),
                CancellationToken(),
            )

        await runtime.stop_when_idle()

    asyncio.run(scenario())


def test_relay_tool_raises_for_non_string_payload() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine(worker_count=2)
        _NonStringTask.bind(runtime, CLAUDE_ID)
        tool = _build_ask_claude_tool(
            runtime=runtime,
            native_id=NATIVE_ID,
            claude_id=CLAUDE_ID,
        )

        with pytest.raises(RuntimeError, match="non-string response"):
            await tool.run(
                tool.args_type()(task="Do work."),
                CancellationToken(),
            )

        await runtime.stop_when_idle()

    asyncio.run(scenario())
