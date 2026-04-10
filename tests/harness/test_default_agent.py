import asyncio

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    Runner,
    RunnerHooks,
    RunResult,
    RunState,
    Task,
)
from agentlane.harness._run import copy_run_state
from agentlane.harness.agents import AgentBase, DefaultAgent
from agentlane.models import MessageDict, Model, ModelResponse
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


def _last_user_input(run_state: RunState) -> object:
    """Return the latest user-side item from one run state."""
    for item in reversed(run_state.continuation_history):
        if isinstance(item, ModelResponse):
            continue
        return item
    return run_state.original_input


class _RecordingRunner(Runner):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[RunState] = []

    async def run(
        self,
        agent: Task,
        state: RunState,
        *,
        hooks: RunnerHooks | None = None,
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
        state.continuation_history.append(response)
        return RunResult(
            final_output=reply_text,
            responses=list(state.responses),
            turn_count=state.turn_count,
        )


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
            original_input="draft a plan",
            continuation_history=[],
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
            original_input="first",
            continuation_history=[],
            responses=[],
            turn_count=0,
        )
        assert runner.calls[1].original_input == "first"
        assert runner.calls[1].turn_count == 1
        assert (
            second.run_state.responses[-1].choices[0].message.content == "Support:first"
        )

    asyncio.run(scenario())


def test_default_agent_implements_agent_base_contract() -> None:
    agent = DefaultAgent(descriptor=AgentDescriptor(name="Support"))

    assert isinstance(agent, AgentBase)


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
        assert runner.calls[1].original_input == "first"
        assert runner.calls[1].turn_count == 1
        assert runner.calls[1].continuation_history[1] == "branch"

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
        assert runner.calls[1].original_input == "first"
        assert runner.calls[1].turn_count == 1

    asyncio.run(scenario())
