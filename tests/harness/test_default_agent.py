import asyncio
from collections.abc import Sequence

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    Runner,
    RunnerHooks,
    RunResult,
    RunState,
    RunStream,
    Task,
)
from agentlane.harness._run import copy_run_state
from agentlane.harness.agents import AgentBase, DefaultAgent
from agentlane.models import (
    MessageDict,
    Model,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
)
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine


def make_assistant_response(content: str) -> ModelResponse:
    """Build one canonical assistant response for default-agent tests."""
    return ModelResponse.model_validate(
        {
            "id": "chatcmpl_default_agent",
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
                    },
                }
            ],
        }
    )


class _SequenceModel(Model[ModelResponse]):
    def __init__(self, outcomes: list[ModelResponse]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[list[MessageDict]] = []

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: object | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        del extra_call_args
        del schema
        del tools
        del cancellation_token
        del kwargs

        self.calls.append([dict(message) for message in messages])
        if not self._outcomes:
            raise AssertionError("Expected one queued model response.")
        return self._outcomes.pop(0)


class _StreamingSequenceModel(Model[ModelResponse]):
    def __init__(
        self,
        outcomes: list[ModelResponse],
        *,
        wait_for_cancel: bool = False,
    ) -> None:
        self._outcomes = list(outcomes)
        self._wait_for_cancel = wait_for_cancel
        self.calls: list[list[MessageDict]] = []

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: object | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        del messages, extra_call_args, schema, tools, cancellation_token, kwargs
        raise AssertionError("Streaming tests should call stream_response directly.")

    def stream_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: object | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: object,
    ):
        del extra_call_args, schema, tools, kwargs

        async def _stream():
            self.calls.append([dict(message) for message in messages])

            if self._wait_for_cancel:
                if cancellation_token is None:
                    raise AssertionError("Expected cancellation token for stream test.")
                await cancellation_token.wait_cancelled()
                raise asyncio.CancelledError()

            if not self._outcomes:
                raise AssertionError("Expected one queued model response.")

            response = self._outcomes.pop(0)
            content = response.choices[0].message.content or ""
            yield ModelStreamEvent(
                kind=ModelStreamEventKind.TEXT_DELTA,
                text=content,
            )
            yield ModelStreamEvent(
                kind=ModelStreamEventKind.COMPLETED,
                response=response,
            )

        return _stream()


async def _collect_run_stream(stream: RunStream) -> list[ModelStreamEvent]:
    """Collect all streamed events from one harness run."""
    events: list[ModelStreamEvent] = []
    async for event in stream:
        events.append(event)
    return events


def _last_user_input(run_state: RunState) -> object:
    """Return the latest user-side item from one run state."""
    for item in reversed(run_state.history):
        if isinstance(item, ModelResponse):
            continue
        return item
    return None


class _RecordingRunner(Runner):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[RunState] = []

    async def run(
        self,
        agent: Task,
        state: RunState,
        *,
        hooks: RunnerHooks | Sequence[RunnerHooks] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        del hooks
        del cancellation_token

        copied = copy_run_state(state)
        if copied is None:
            raise AssertionError("Expected copied run state.")
        self.calls.append(copied)

        if not isinstance(agent, Agent):
            raise AssertionError("Expected runtime-facing harness Agent.")

        state.turn_count += 1
        reply_text = f"{agent.name}:{_last_user_input(state)}"
        response = make_assistant_response(reply_text)
        state.responses.append(response)
        state.history.append(response)
        return RunResult(
            final_output=reply_text,
            responses=list(state.responses),
            turn_count=state.turn_count,
        )


class _NamedHooks(RunnerHooks):
    def __init__(self, name: str, events: list[tuple[str, str]]) -> None:
        self._name = name
        self._events = events

    async def on_agent_start(
        self,
        task: Task,
        state: RunState,
    ) -> None:
        del task, state
        self._events.append((self._name, "agent_start"))

    async def on_agent_end(
        self,
        task: Task,
        result: RunResult | None,
    ) -> None:
        del task, result
        self._events.append((self._name, "agent_end"))


def test_default_agent_auto_manages_runtime_and_runner() -> None:
    async def scenario() -> None:
        model = _SequenceModel([make_assistant_response("Thanks, I can help.")])
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Support",
                model=model,
                instructions="You are a support agent.",
            )
        )

        result = await agent.run("My order arrived damaged.")

        assert result.final_output == "Thanks, I can help."
        assert result.turn_count == 1
        assert agent.run_state is not None
        assert agent.run_state.turn_count == 1
        assert model.calls == [
            [
                {"role": "system", "content": "You are a support agent."},
                {"role": "user", "content": "My order arrived damaged."},
            ]
        ]

    asyncio.run(scenario())


def test_default_agent_accepts_hook_sequences() -> None:
    async def scenario() -> None:
        model = _SequenceModel([make_assistant_response("Thanks, I can help.")])
        events: list[tuple[str, str]] = []
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Support",
                model=model,
                instructions="You are a support agent.",
            ),
            hooks=(
                _NamedHooks("first", events),
                _NamedHooks("second", events),
            ),
        )

        result = await agent.run("My order arrived damaged.")

        assert result.final_output == "Thanks, I can help."
        assert events == [
            ("first", "agent_start"),
            ("second", "agent_start"),
            ("first", "agent_end"),
            ("second", "agent_end"),
        ]

    asyncio.run(scenario())


def test_default_agent_reuses_persisted_run_state_between_calls() -> None:
    async def scenario() -> None:
        model = _SequenceModel(
            [
                make_assistant_response("First answer."),
                make_assistant_response("Second answer."),
            ]
        )
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Assistant",
                model=model,
            )
        )

        first = await agent.run("first")
        second = await agent.run("second")

        assert first.turn_count == 1
        assert second.turn_count == 2
        assert agent.run_state is not None
        assert agent.run_state.turn_count == 2
        assert model.calls == [
            [
                {"role": "user", "content": "first"},
            ],
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "First answer."},
                {"role": "user", "content": "second"},
            ],
        ]

        agent.reset()
        assert agent.run_state is None

    asyncio.run(scenario())


def test_default_agent_supports_explicit_runtime_and_runner_injection() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = _RecordingRunner()
        agent = DefaultAgent(
            descriptor=AgentDescriptor(name="Planner"),
            runtime=runtime,
            runner=runner,
        )

        result = await agent.run("draft a plan")

        assert result.final_output == "Planner:draft a plan"
        assert result.run_state is not None
        assert result.run_state.turn_count == 1
        assert len(runner.calls) == 1
        assert runner.calls[0] == RunState(
            instructions=None,
            history=["draft a plan"],
            responses=[],
            turn_count=0,
        )
        assert not runtime.is_running

    asyncio.run(scenario())


def test_default_agent_supports_subclass_descriptor_and_explicit_run_state_resume() -> (
    None
):
    async def scenario() -> None:
        runner = _RecordingRunner()

        class SupportAgent(DefaultAgent):
            descriptor = AgentDescriptor(name="Support")

        agent = SupportAgent(runner=runner)

        first = await agent.run("first")
        saved_state = first.run_state
        if saved_state is None:
            raise AssertionError("Expected saved run state from first run.")

        second = await agent.run(saved_state)

        assert second.final_output == "Support:first"
        assert second.run_state is not None
        assert second.run_state.turn_count == 2
        assert len(runner.calls) == 2
        assert runner.calls[0] == RunState(
            instructions=None,
            history=["first"],
            responses=[],
            turn_count=0,
        )
        assert runner.calls[1].history[0] == "first"
        assert runner.calls[1].turn_count == 1
        assert (
            second.run_state.responses[-1].choices[0].message.content == "Support:first"
        )

    asyncio.run(scenario())


def test_default_agent_implements_agent_base_contract() -> None:
    agent = DefaultAgent(descriptor=AgentDescriptor(name="Support"))

    assert isinstance(agent, AgentBase)


def test_default_agent_run_stream_emits_events_and_commits_state() -> None:
    async def scenario() -> None:
        model = _StreamingSequenceModel([make_assistant_response("streamed reply")])
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Support",
                model=model,
                instructions="You are helpful.",
            )
        )

        stream = await agent.run_stream("hello")
        events = await _collect_run_stream(stream)
        result = await stream.result()

        assert [event.kind for event in events] == [
            ModelStreamEventKind.TEXT_DELTA,
            ModelStreamEventKind.COMPLETED,
        ]
        assert events[0].text == "streamed reply"
        assert result.final_output == "streamed reply"
        assert agent.run_state is not None
        assert agent.run_state.turn_count == 1
        assert model.calls == [
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ]
        ]

    asyncio.run(scenario())


def test_default_agent_run_stream_reuses_persisted_state() -> None:
    async def scenario() -> None:
        model = _StreamingSequenceModel(
            [
                make_assistant_response("first"),
                make_assistant_response("second"),
            ]
        )
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Support",
                model=model,
            )
        )

        first_stream = await agent.run_stream("one")
        await _collect_run_stream(first_stream)
        await first_stream.result()

        second_stream = await agent.run_stream("two")
        await _collect_run_stream(second_stream)
        second_result = await second_stream.result()

        assert second_result.turn_count == 2
        assert model.calls == [
            [{"role": "user", "content": "one"}],
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "two"},
            ],
        ]

    asyncio.run(scenario())


def test_default_agent_run_stream_close_cancels_without_committing_state() -> None:
    async def scenario() -> None:
        model = _StreamingSequenceModel([], wait_for_cancel=True)
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Support",
                model=model,
            )
        )

        stream = await agent.run_stream("hello")
        await asyncio.sleep(0)
        await stream.aclose()

        try:
            await stream.result()
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Expected cancelled stream result.")

        assert agent.run_state is None

    asyncio.run(scenario())


def test_default_agent_fork_branches_without_mutating_persisted_state() -> None:
    async def scenario() -> None:
        runner = _RecordingRunner()
        agent = DefaultAgent(
            descriptor=AgentDescriptor(name="Support"),
            runner=runner,
        )

        first = await agent.run("first")
        persisted_after_run = copy_run_state(agent.run_state)
        if persisted_after_run is None:
            raise AssertionError("Expected persisted state after first run.")

        branch = await agent.fork("branch")

        if branch.run_state is None:
            raise AssertionError("Expected fork result to expose branch run state.")
        assert first.run_state is not None
        assert branch.final_output == "Support:branch"
        assert branch.run_state.turn_count == 2
        assert agent.run_state == persisted_after_run
        assert agent.run_state is not None
        assert agent.run_state.turn_count == 1
        assert len(runner.calls) == 2
        assert runner.calls[1].history[0] == "first"
        assert runner.calls[1].turn_count == 1
        assert runner.calls[1].history[2] == "branch"

    asyncio.run(scenario())


def test_default_agent_fork_supports_explicit_run_state_without_persisting_it() -> None:
    async def scenario() -> None:
        runner = _RecordingRunner()
        agent = DefaultAgent(
            descriptor=AgentDescriptor(name="Support"),
            runner=runner,
        )

        first = await agent.run("first")
        saved_state = first.run_state
        if saved_state is None:
            raise AssertionError("Expected saved run state from first run.")

        branch = await agent.fork(saved_state)

        assert branch.final_output == "Support:first"
        assert branch.run_state is not None
        assert branch.run_state.turn_count == 2
        assert agent.run_state is not None
        assert agent.run_state.turn_count == 1
        assert len(runner.calls) == 2
        assert runner.calls[1].history[0] == "first"
        assert runner.calls[1].turn_count == 1

    asyncio.run(scenario())
