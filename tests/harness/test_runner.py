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
    Tool,
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

    async def on_tool_call_start(
        self,
        task: Task,
        tool_call: ToolCall,
    ) -> None:
        self.events.append(
            ("tool_start", (_task_name(task), tool_call.function.name, tool_call.id))
        )

    async def on_tool_call_end(
        self,
        task: Task,
        tool_call: ToolCall,
        result: object,
    ) -> None:
        self.events.append(
            (
                "tool_end",
                (_task_name(task), tool_call.function.name, tool_call.id, result),
            )
        )


def _task_name(task: Task) -> str:
    """Return the agent name when present for hook assertions."""
    if isinstance(task, Agent):
        return task.name
    return type(task).__name__


class _StructuredResponse(BaseModel):
    value: str


class _EchoArgs(BaseModel):
    text: str


class _MockOutputSchema:
    def response_format(self) -> dict[str, Any] | None:
        return None


def _make_tool_call(
    *,
    tool_id: str,
    arguments: str,
    name: str = "echo",
) -> ToolCall:
    """Build one canonical tool call payload for harness tests."""
    return ToolCall.model_validate(
        {
            "id": tool_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }
    )


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


def test_runner_executes_tool_calls_and_continues_loop() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        executed: list[str] = []

        async def echo_handler(
            args: _EchoArgs,
            cancellation_token: CancellationToken,
        ) -> str:
            del cancellation_token
            executed.append(args.text)
            return f"tool:{args.text}"

        tool = Tool(
            name="echo",
            description="Echo text",
            args_model=_EchoArgs,
            handler=echo_handler,
        )
        model = _SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        _make_tool_call(
                            tool_id="call_1",
                            arguments='{"text":"docs"}',
                        )
                    ],
                ),
                make_assistant_response(content="docs are ready"),
            ]
        )
        agent = Agent(
            runtime,
            runner,
            descriptor=AgentDescriptor(
                name="ToolRunner",
                model=model,
                tools=Tools(tools=[tool]),
            ),
        )
        state = RunState(
            original_input="search docs",
            continuation_history=[],
            responses=[],
        )

        result = await runner.run(agent, state)

        assert result.final_output == "docs are ready"
        assert executed == ["docs"]
        assert len(result.responses) == 2
        assert model.calls == [
            [_message("user", "search docs")],
            [
                _message("user", "search docs"),
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"text":"docs"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "echo",
                    "content": "tool:docs",
                },
            ],
        ]
        assert len(state.continuation_history) == 3
        assert isinstance(state.continuation_history[0], ModelResponse)
        assert state.continuation_history[1] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "echo",
            "content": "tool:docs",
        }
        assert isinstance(state.continuation_history[2], ModelResponse)

    asyncio.run(scenario())


def test_runner_executes_parallel_tool_calls_and_invokes_hooks() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        hooks = _RecordingHooks()
        started_calls: list[str] = []
        both_started = asyncio.Event()
        release = asyncio.Event()

        async def echo_handler(
            args: _EchoArgs,
            cancellation_token: CancellationToken,
        ) -> str:
            del cancellation_token
            started_calls.append(args.text)
            if len(started_calls) == 2:
                both_started.set()
            await release.wait()
            return f"done:{args.text}"

        tool = Tool(
            name="echo",
            description="Echo text",
            args_model=_EchoArgs,
            handler=echo_handler,
        )
        model = _SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        _make_tool_call(
                            tool_id="call_1",
                            arguments='{"text":"one"}',
                        ),
                        _make_tool_call(
                            tool_id="call_2",
                            arguments='{"text":"two"}',
                        ),
                    ],
                ),
                make_assistant_response(content="complete"),
            ]
        )
        agent = Agent(
            runtime,
            runner,
            descriptor=AgentDescriptor(
                name="ParallelTools",
                model=model,
                tools=Tools(
                    tools=[tool],
                    parallel_tool_calls=True,
                ),
            ),
        )
        state = RunState(
            original_input="run tools",
            continuation_history=[],
            responses=[],
        )

        run_task = asyncio.create_task(runner.run(agent, state, hooks=hooks))
        await asyncio.wait_for(both_started.wait(), timeout=1.0)
        release.set()

        result = await run_task

        assert result.final_output == "complete"
        assert set(started_calls) == {"one", "two"}
        assert ("tool_start", ("ParallelTools", "echo", "call_1")) in hooks.events
        assert ("tool_start", ("ParallelTools", "echo", "call_2")) in hooks.events
        assert (
            "tool_end",
            ("ParallelTools", "echo", "call_1", "done:one"),
        ) in hooks.events
        assert (
            "tool_end",
            ("ParallelTools", "echo", "call_2", "done:two"),
        ) in hooks.events

    asyncio.run(scenario())


def test_runner_filters_exhausted_tools_on_later_turns() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()

        async def echo_handler(
            args: _EchoArgs,
            cancellation_token: CancellationToken,
        ) -> str:
            del cancellation_token
            return f"echo:{args.text}"

        async def lookup_handler(
            args: _EchoArgs,
            cancellation_token: CancellationToken,
        ) -> str:
            del cancellation_token
            return f"lookup:{args.text}"

        echo_tool = Tool(
            name="echo",
            description="Echo text",
            args_model=_EchoArgs,
            handler=echo_handler,
        )
        lookup_tool = Tool(
            name="lookup",
            description="Lookup text",
            args_model=_EchoArgs,
            handler=lookup_handler,
        )
        tools = Tools(
            tools=[echo_tool, lookup_tool],
            tool_call_limits={"echo": 1},
        )
        model = _SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        _make_tool_call(
                            tool_id="call_1",
                            arguments='{"text":"docs"}',
                        )
                    ],
                ),
                make_assistant_response(content="done"),
            ]
        )
        agent = Agent(
            runtime,
            runner,
            descriptor=AgentDescriptor(
                name="LimitedTools",
                model=model,
                tools=tools,
            ),
        )
        state = RunState(
            original_input="search docs",
            continuation_history=[],
            responses=[],
        )

        result = await runner.run(agent, state)

        assert result.final_output == "done"
        assert model.call_options[0]["tools"] == tools
        second_turn_tools = cast(Tools, model.call_options[1]["tools"])
        assert [tool.name for tool in second_turn_tools.tools] == ["lookup"]

    asyncio.run(scenario())


def test_runner_disables_tools_after_max_round_trips() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()

        async def echo_handler(
            args: _EchoArgs,
            cancellation_token: CancellationToken,
        ) -> str:
            del cancellation_token
            return f"echo:{args.text}"

        tool = Tool(
            name="echo",
            description="Echo text",
            args_model=_EchoArgs,
            handler=echo_handler,
        )
        tools = Tools(
            tools=[tool],
            max_tool_round_trips=1,
        )
        model = _SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        _make_tool_call(
                            tool_id="call_1",
                            arguments='{"text":"docs"}',
                        )
                    ],
                ),
                make_assistant_response(content="done"),
            ]
        )
        agent = Agent(
            runtime,
            runner,
            descriptor=AgentDescriptor(
                name="RoundTripLimited",
                model=model,
                tools=tools,
            ),
        )
        state = RunState(
            original_input="search docs",
            continuation_history=[],
            responses=[],
        )

        result = await runner.run(agent, state)

        assert result.final_output == "done"
        assert model.call_options[0]["tools"] == tools
        assert model.call_options[1]["tools"] is None

    asyncio.run(scenario())


def test_agent_tool_resolution_supports_inherit_override_and_disable() -> None:
    runtime = SingleThreadedRuntimeEngine()
    runner = Runner()
    parent_tools = Tools(tools=[])

    inherited = Agent(
        runtime,
        runner,
        descriptor=AgentDescriptor(name="Inherited"),
        parent_tools=parent_tools,
    )
    overridden_tools = Tools(tools=[])
    overridden = Agent(
        runtime,
        runner,
        descriptor=AgentDescriptor(
            name="Overridden",
            tools=overridden_tools,
        ),
        parent_tools=parent_tools,
    )
    disabled = Agent(
        runtime,
        runner,
        descriptor=AgentDescriptor(
            name="Disabled",
            tools=None,
        ),
        parent_tools=parent_tools,
    )

    assert inherited.tools is parent_tools
    assert overridden.tools is overridden_tools
    assert disabled.tools is None


def test_runner_raises_when_model_returns_tool_calls_without_tools() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        model = _SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        _make_tool_call(
                            tool_id="call_1",
                            arguments='{"text":"docs"}',
                        )
                    ],
                )
            ]
        )
        agent = Agent(
            runtime,
            runner,
            descriptor=AgentDescriptor(
                name="ToolBoundary",
                model=model,
                tools=None,
            ),
        )
        state = RunState(
            original_input="search docs",
            continuation_history=[],
            responses=[],
        )

        with pytest.raises(
            ModelBehaviorError,
            match="agent exposes no tools",
        ):
            await runner.run(agent, state)

    asyncio.run(scenario())
