import asyncio
from typing import Any, cast

import pytest
from pydantic import BaseModel

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    Runner,
    RunnerHooks,
    RunResult,
    RunState,
    Task,
)
from agentlane.messaging import AgentId, DeliveryStatus
from agentlane.models import (
    MessageDict,
    Model,
    ModelBehaviorError,
    ModelResponse,
    OutputSchema,
    PromptSpec,
    PromptTemplate,
    ToolCall,
    Tools,
    get_content_or_none,
)
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine


def _message(role: str, content: object) -> MessageDict:
    """Build one expected canonical model message for assertions."""
    return {
        "role": role,
        "content": content,
    }


def _copy_messages(messages: list[MessageDict]) -> list[MessageDict]:
    """Return a shallow copy of message dictionaries for assertions."""
    return [dict(message) for message in messages]


def make_assistant_response(
    content: str | None,
    *,
    tool_calls: list[ToolCall] | None = None,
) -> ModelResponse:
    """Build one canonical chat-completions response for tests."""
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


class _RetryableModelError(Exception):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _SequenceModel(Model[ModelResponse]):
    def __init__(
        self,
        outcomes: list[object],
        *,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self._outcomes = list(outcomes)
        self._started = started
        self._release = release
        self.calls: list[list[MessageDict]] = []
        self.call_options: list[dict[str, object]] = []

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: object | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        del cancellation_token

        self.calls.append(_copy_messages(messages))
        self.call_options.append(
            {
                "extra_call_args": extra_call_args,
                "schema": schema,
                "tools": tools,
                "kwargs": dict(kwargs),
            }
        )
        if self._started is not None:
            self._started.set()
        if self._release is not None:
            await self._release.wait()

        if not self._outcomes:
            raise AssertionError("Expected one queued model outcome for the test.")

        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if not isinstance(outcome, ModelResponse):
            raise AssertionError("Expected ModelResponse outcomes in runner tests.")
        return outcome


class _RecordingHooks(RunnerHooks):
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    async def on_agent_start(
        self,
        task: Task,
        state: RunState,
    ) -> None:
        self.events.append(
            (
                "agent_start",
                (_task_name(task), state.original_input, state.turn_count),
            )
        )

    async def on_agent_end(
        self,
        task: Task,
        result: RunResult | None,
    ) -> None:
        final_output = None if result is None else result.final_output
        self.events.append(("agent_end", (_task_name(task), final_output)))

    async def on_llm_start(
        self,
        task: Task,
        messages: list[MessageDict],
    ) -> None:
        self.events.append(("llm_start", (_task_name(task), _copy_messages(messages))))

    async def on_llm_end(
        self,
        task: Task,
        response: ModelResponse,
    ) -> None:
        self.events.append(
            ("llm_end", (_task_name(task), get_content_or_none(response)))
        )


def _task_name(task: Task) -> str:
    """Return the agent name when present for hook assertions."""
    if isinstance(task, Agent):
        return task.name
    return type(task).__name__


class _StructuredResponse(BaseModel):
    value: str


class _MockOutputSchema:
    def response_format(self) -> dict[str, Any] | None:
        return None


def test_runner_returns_run_result_and_updates_run_state() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        model = _SequenceModel([make_assistant_response(content="hello back")])
        agent = Agent(
            runtime,
            runner,
            descriptor=AgentDescriptor(
                name="Support",
                model=model,
                instructions="You are helpful.",
            ),
        )
        state = RunState(
            original_input="hello",
            continuation_history=[],
            responses=[],
        )

        result = await runner.run(agent, state)

        assert result.final_output == "hello back"
        assert result.turn_count == 1
        assert [get_content_or_none(response) for response in result.responses] == [
            "hello back"
        ]
        assert model.calls == [
            [
                _message("system", "You are helpful."),
                _message("user", "hello"),
            ]
        ]
        assert state.turn_count == 1
        assert [get_content_or_none(response) for response in state.responses] == [
            "hello back"
        ]
        assert len(state.continuation_history) == 1
        assert isinstance(state.continuation_history[0], ModelResponse)

    asyncio.run(scenario())


def test_runner_retries_retryable_model_failures() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner(max_attempts=2)
        model = _SequenceModel(
            [
                _RetryableModelError("rate limited", status_code=429),
                make_assistant_response(content="retried response"),
            ]
        )
        agent = Agent(
            runtime,
            runner,
            descriptor=AgentDescriptor(name="Retryer", model=model),
        )
        state = RunState(
            original_input="hello",
            continuation_history=[],
            responses=[],
        )

        result = await runner.run(agent, state)

        assert result.final_output == "retried response"
        assert result.turn_count == 1
        assert model.calls == [
            [_message("user", "hello")],
            [_message("user", "hello")],
        ]

    asyncio.run(scenario())


def test_runner_invokes_hooks_in_order() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        model = _SequenceModel([make_assistant_response(content="observed")])
        hooks = _RecordingHooks()
        agent = Agent(
            runtime,
            runner,
            descriptor=AgentDescriptor(name="Observer", model=model),
        )
        state = RunState(
            original_input="inspect",
            continuation_history=[],
            responses=[],
        )

        result = await runner.run(agent, state, hooks=hooks)

        assert result.final_output == "observed"
        assert hooks.events == [
            ("agent_start", ("Observer", "inspect", 0)),
            ("llm_start", ("Observer", [_message("user", "inspect")])),
            ("llm_end", ("Observer", "observed")),
            ("agent_end", ("Observer", "observed")),
        ]

    asyncio.run(scenario())


def test_runner_forwards_native_model_call_options() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        model = _SequenceModel([make_assistant_response(content="configured")])
        schema = _StructuredResponse
        tools = Tools(tools=[])
        agent = Agent(
            runtime,
            runner,
            descriptor=AgentDescriptor(
                name="Configurer",
                model=model,
                model_args={
                    "temperature": 0.2,
                    "reasoning_effort": "medium",
                    "prompt_cache_retention": "24h",
                },
                schema=schema,
                tools=tools,
            ),
        )
        state = RunState(
            original_input="configure",
            continuation_history=[],
            responses=[],
        )

        await runner.run(agent, state)

        assert model.call_options == [
            {
                "extra_call_args": {
                    "temperature": 0.2,
                    "reasoning_effort": "medium",
                    "prompt_cache_retention": "24h",
                },
                "schema": schema,
                "tools": tools,
                "kwargs": {},
            }
        ]

    asyncio.run(scenario())


def test_runner_builds_request_from_prompt_instructions_and_history_items() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        model = _SequenceModel([make_assistant_response(content="done")])
        prior_response = make_assistant_response(content="first reply")
        instruction_template = PromptTemplate[dict[str, object], list[str]](
            system_template="You support {{ team }}.",
            user_template=None,
            output_schema=cast(OutputSchema[list[str]], _MockOutputSchema()),
        )
        user_template = PromptTemplate[dict[str, object], list[str]](
            system_template=None,
            user_template="follow up for {{ team }}",
            output_schema=cast(OutputSchema[list[str]], _MockOutputSchema()),
        )
        agent = Agent(
            runtime,
            runner,
            descriptor=AgentDescriptor(
                name="Support",
                model=model,
                instructions=PromptSpec(
                    template=instruction_template,
                    values={"team": "ops"},
                ),
            ),
        )
        state = RunState(
            original_input=[
                "first question",
                prior_response,
                PromptSpec(template=user_template, values={"team": "ops"}),
            ],
            continuation_history=[],
            responses=[],
        )

        result = await runner.run(agent, state)

        assert result.final_output == "done"
        assert model.calls == [
            [
                _message("system", "You support ops."),
                _message("user", "first question"),
                _message("assistant", "first reply"),
                _message("user", "follow up for ops"),
            ]
        ]

    asyncio.run(scenario())


def test_runner_shared_instance_serves_multiple_agents_safely() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine(worker_count=2)
        runner = Runner()
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        release = asyncio.Event()

        first_model = _SequenceModel(
            [make_assistant_response(content="first done")],
            started=first_started,
            release=release,
        )
        second_model = _SequenceModel(
            [make_assistant_response(content="second done")],
            started=second_started,
            release=release,
        )

        first_id = AgentId.from_values("assistant-agent", "runner-a")
        second_id = AgentId.from_values("assistant-agent", "runner-b")

        first_agent = Agent.bind(
            runtime,
            first_id,
            runner=runner,
            descriptor=AgentDescriptor(
                name="First",
                model=first_model,
                instructions="You answer the first queue.",
            ),
        )
        second_agent = Agent.bind(
            runtime,
            second_id,
            runner=runner,
            descriptor=AgentDescriptor(
                name="Second",
                model=second_model,
                instructions="You answer the second queue.",
            ),
        )

        first_task = asyncio.create_task(
            runtime.send_message("one", recipient=first_id)
        )
        second_task = asyncio.create_task(
            runtime.send_message("two", recipient=second_id)
        )

        await asyncio.wait_for(
            asyncio.gather(first_started.wait(), second_started.wait()),
            timeout=1.0,
        )
        release.set()

        first_outcome, second_outcome = await asyncio.gather(first_task, second_task)
        await runtime.stop_when_idle()

        assert first_outcome.status == DeliveryStatus.DELIVERED
        assert second_outcome.status == DeliveryStatus.DELIVERED
        assert isinstance(first_outcome.response_payload, RunResult)
        assert isinstance(second_outcome.response_payload, RunResult)
        assert first_outcome.response_payload.final_output == "first done"
        assert second_outcome.response_payload.final_output == "second done"
        assert first_agent.run_state is not None
        assert second_agent.run_state is not None
        assert first_agent.run_state.original_input == "one"
        assert second_agent.run_state.original_input == "two"
        assert first_agent.run_state.turn_count == 1
        assert second_agent.run_state.turn_count == 1

    asyncio.run(scenario())


def test_runner_raises_for_tool_calls_before_phase_five() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        model = _SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        ToolCall.model_validate(
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query":"docs"}',
                                },
                            }
                        )
                    ],
                )
            ]
        )
        agent = Agent(
            runtime,
            runner,
            descriptor=AgentDescriptor(name="ToolBoundary", model=model),
        )
        state = RunState(
            original_input="search docs",
            continuation_history=[],
            responses=[],
        )

        with pytest.raises(
            ModelBehaviorError,
            match="Runner does not execute tool calls yet",
        ):
            await runner.run(agent, state)

    asyncio.run(scenario())
