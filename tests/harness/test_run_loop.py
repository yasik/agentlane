"""Loop execution through model and tool contracts, without a real Runner."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Self
from unittest.mock import Mock, create_autospec

import pytest

from agentlane.harness import RunnerHooks, RunState, Task
from agentlane.harness._lifecycle import LifecycleRunner
from agentlane.harness._run_loop import AgentDelegationState, RunLoop
from agentlane.messaging import AgentId
from agentlane.models import MessageDict, Model, ModelResponse, ToolExecutor, Tools


def response(message: MessageDict) -> ModelResponse:
    return ModelResponse.model_validate(
        {
            "id": "test",
            "object": "chat.completion",
            "created": 1,
            "model": "test",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        }
    )


class ToolThenAnswer(Model[ModelResponse]):
    def __init__(self, failure: Exception | None) -> None:
        self.failure = failure
        self.requests: list[list[MessageDict]] = []
        self.closed = False

    @asynccontextmanager
    async def open_run(self) -> AsyncGenerator[Self, None]:
        try:
            yield self
        finally:
            self.closed = True

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: object = None,
        schema: object = None,
        tools: object = None,
        cancellation_token: object = None,
        parent_span: object = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.requests.append(messages)

        # Request one tool round, then complete or fail on the next model call.
        if len(self.requests) == 1:
            return response(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"text":"hello"}',
                            },
                        }
                    ],
                }
            )
        if self.failure is not None:
            raise self.failure

        return response({"role": "assistant", "content": "done"})


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_loop_executes_tool_turn_and_closes_model(fails: bool) -> None:
    calls: list[str] = []

    async def echo(text: str) -> str:
        """Echo the supplied text."""
        calls.append(text)

        return text.upper()

    failure = ValueError("model failed") if fails else None
    model = ToolThenAnswer(failure)
    task_id = AgentId.from_values("test", "loop")
    agent = Mock(
        spec=Task,
        model=model,
        model_args=None,
        tools=Tools(tools=[echo]),
        base_tools=None,
        schema=None,
        instructions=None,
        shims=None,
        task_id=task_id,
    )
    agent.name = "test"
    delegate = create_autospec(LifecycleRunner, instance=True)
    state = RunState(instructions=None, history=["hello"], responses=[])
    loop = RunLoop(
        agent=agent,
        runner_task=agent,
        model=model,
        state=state,
        delegate_runner=delegate,
        agent_delegation=AgentDelegationState(max_depth=4, max_threads=16),
        tool_executor=ToolExecutor(),
        hooks=RunnerHooks(),
    )

    # Exercise the loop directly without invoking a real Runner.
    if fails:
        with pytest.raises(ValueError) as caught:
            await loop.run()

        assert caught.value is failure
    else:
        result = await loop.run()

        assert result.final_output == "done"
        assert result.turn_count == 2
        assert len(result.responses) == 2

    # Tool delivery and model closure must hold on both success and failure.
    assert calls == ["hello"]
    assert model.requests[1][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "echo",
        "content": "HELLO",
    }
    assert model.closed
    assert delegate.mock_calls == []
