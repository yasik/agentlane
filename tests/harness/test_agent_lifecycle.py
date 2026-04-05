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
from agentlane.messaging import AgentId, DeliveryStatus
from agentlane.models import ModelResponse
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine


def _assistant_response(content: str) -> ModelResponse:
    """Build one canonical assistant response for lifecycle tests."""
    return ModelResponse.model_validate(
        {
            "id": "chatcmpl_lifecycle",
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


def _copy_run_state(run_state: RunState) -> RunState:
    """Return a shallow copy of one run state for assertions."""
    original_input = run_state.original_input
    copied_input = (
        original_input if isinstance(original_input, str) else list(original_input)
    )
    return RunState(
        original_input=copied_input,
        continuation_history=list(run_state.continuation_history),
        responses=list(run_state.responses),
        turn_count=run_state.turn_count,
    )


def _last_user_input(run_state: RunState) -> object:
    """Return the latest user-side input represented in one run state."""
    for item in reversed(run_state.continuation_history):
        if isinstance(item, ModelResponse):
            continue
        return item
    return run_state.original_input


def _require_run_result(payload: object | None) -> RunResult:
    """Return a strongly typed lifecycle result for assertions."""
    if not isinstance(payload, RunResult):
        raise AssertionError("Expected a `RunResult` payload from the runtime.")
    return payload


class _RecordingRunner(Runner):
    def __init__(self, *, block_first_turn: bool = False) -> None:
        super().__init__()
        self.calls: list[RunState] = []
        self._block_first_turn = block_first_turn
        self.first_turn_started = asyncio.Event()
        self.release_first_turn = asyncio.Event()

    async def run(
        self,
        agent: Task,
        state: RunState,
        *,
        hooks: RunnerHooks | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        _ = hooks
        _ = cancellation_token
        self.calls.append(_copy_run_state(state))

        if self._block_first_turn and len(self.calls) == 1:
            self.first_turn_started.set()
            await self.release_first_turn.wait()

        if not isinstance(agent, Agent):
            raise AssertionError(
                "Expected the default harness Agent in lifecycle tests."
            )

        state.turn_count += 1
        reply_text = f"{agent.name}:{_last_user_input(state)}"
        response = _assistant_response(reply_text)
        state.responses.append(response)
        state.continuation_history.append(response)
        return RunResult(
            final_output=reply_text,
            responses=list(state.responses),
            turn_count=state.turn_count,
        )


class _TestableAgent(Agent):
    async def queue_for_test(
        self, run_input: str | list[object] | RunState
    ) -> RunResult:
        return await self._enqueue_input(run_input)


def test_agent_starts_new_run_with_string_input() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = _RecordingRunner()

        Agent.register(
            runtime,
            "assistant-agent",
            runner=runner,
            descriptor=AgentDescriptor(
                name="Support",
                description="Handles support requests",
                instructions="You are a helpful support agent.",
                skills=("triage",),
            ),
        )

        outcome = await runtime.send_message(
            "hello",
            recipient=AgentId.from_values("assistant-agent", "session-1"),
        )

        await runtime.stop_when_idle()

        assert outcome.status == DeliveryStatus.DELIVERED
        assert isinstance(outcome.response_payload, RunResult)
        assert outcome.response_payload.final_output == "Support:hello"
        assert runner.calls == [
            RunState(
                original_input="hello",
                continuation_history=[],
                responses=[],
                turn_count=0,
            )
        ]

    asyncio.run(scenario())


def test_agent_continues_existing_run_after_idle() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = _RecordingRunner()
        agent_id = AgentId.from_values("assistant-agent", "session-2")
        agent = Agent.bind(
            runtime,
            agent_id,
            runner=runner,
            descriptor=AgentDescriptor(
                name="Planner",
                description="Plans next steps",
                instructions="You plan carefully.",
                skills=("analysis",),
                context={"team": "ops"},
                memory={"kind": "ephemeral"},
            ),
        )

        first = await runtime.send_message("first", recipient=agent_id)
        second = await runtime.send_message("second", recipient=agent_id)

        await runtime.stop_when_idle()

        first_result = _require_run_result(first.response_payload)
        second_result = _require_run_result(second.response_payload)
        assert first_result.final_output == "Planner:first"
        assert second_result.final_output == "Planner:second"
        assert agent.name == "Planner"
        assert agent.description == "Plans next steps"
        assert agent.instructions == "You plan carefully."
        assert agent.skills == ("analysis",)
        assert agent.context == {"team": "ops"}
        assert agent.memory == {"kind": "ephemeral"}
        assert not agent.is_running
        assert agent.pending_input_count == 0
        assert len(runner.calls) == 2
        assert runner.calls[0] == RunState(
            original_input="first",
            continuation_history=[],
            responses=[],
            turn_count=0,
        )
        assert runner.calls[1].original_input == "first"
        assert runner.calls[1].turn_count == 1
        assert runner.calls[1].continuation_history[1] == "second"
        assert agent.run_state is not None
        assert agent.run_state.original_input == "first"
        assert agent.run_state.turn_count == 2
        assert len(agent.run_state.responses) == 2

    asyncio.run(scenario())


def test_agent_queues_inputs_for_next_turn_while_running() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = _RecordingRunner(block_first_turn=True)
        agent_id = AgentId.from_values("assistant-agent", "session-3")
        agent = _TestableAgent.bind(
            runtime,
            agent_id,
            runner=runner,
            descriptor=AgentDescriptor(
                name="Researcher",
                instructions="You research methodically.",
            ),
        )

        first_turn = asyncio.create_task(agent.queue_for_test("first"))
        await asyncio.wait_for(runner.first_turn_started.wait(), timeout=1.0)

        second_turn = asyncio.create_task(agent.queue_for_test("second"))
        await asyncio.sleep(0)

        assert agent.is_running
        assert agent.pending_input_count == 1

        runner.release_first_turn.set()

        first_result = await first_turn
        second_result = await second_turn

        assert first_result.final_output == "Researcher:first"
        assert second_result.final_output == "Researcher:second"
        assert not agent.is_running
        assert agent.pending_input_count == 0
        assert len(runner.calls) == 2
        assert runner.calls[1].continuation_history[1] == "second"

    asyncio.run(scenario())


def test_agent_continues_from_preloaded_run_state() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = _RecordingRunner()
        agent_id = AgentId.from_values("assistant-agent", "session-recovered")
        prior_response = _assistant_response("Recovered:before")
        recovered_state = RunState(
            original_input="before",
            continuation_history=[prior_response],
            responses=[prior_response],
            turn_count=1,
        )

        agent = Agent.bind(
            runtime,
            agent_id,
            runner=runner,
            descriptor=AgentDescriptor(
                name="Recovered",
                instructions="You should not be reseeded.",
            ),
            run_state=recovered_state,
        )

        outcome = await runtime.send_message("after", recipient=agent_id)

        await runtime.stop_when_idle()

        assert outcome.status == DeliveryStatus.DELIVERED
        result = _require_run_result(outcome.response_payload)
        assert result.final_output == "Recovered:after"
        assert len(runner.calls) == 1
        assert runner.calls[0].original_input == "before"
        assert runner.calls[0].turn_count == 1
        assert runner.calls[0].continuation_history[1] == "after"
        assert agent.run_state is not None
        assert agent.run_state.turn_count == 2
        assert len(agent.run_state.responses) == 2
        assert recovered_state.turn_count == 1

    asyncio.run(scenario())


def test_agent_accepts_run_state_as_runtime_input() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = _RecordingRunner()
        resumed_state = RunState(
            original_input="before",
            continuation_history=[_assistant_response("Support:before")],
            responses=[_assistant_response("Support:before")],
            turn_count=1,
        )

        Agent.register(
            runtime,
            "assistant-agent",
            runner=runner,
            descriptor=AgentDescriptor(name="Support"),
        )

        outcome = await runtime.send_message(
            resumed_state,
            recipient=AgentId.from_values("assistant-agent", "session-resume"),
        )

        await runtime.stop_when_idle()

        assert outcome.status == DeliveryStatus.DELIVERED
        result = _require_run_result(outcome.response_payload)
        assert result.final_output == "Support:before"
        assert len(runner.calls) == 1
        assert runner.calls[0].turn_count == 1

    asyncio.run(scenario())


def test_runtime_serializes_multiple_inputs_to_same_agent_id() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = _RecordingRunner()
        agent_id = AgentId.from_values("assistant-agent", "session-4")
        agent = Agent.bind(
            runtime,
            agent_id,
            runner=runner,
            descriptor=AgentDescriptor(
                name="Coordinator",
                instructions="You coordinate carefully.",
            ),
        )

        outcomes = await asyncio.gather(
            runtime.send_message("one", recipient=agent_id),
            runtime.send_message("two", recipient=agent_id),
            runtime.send_message("three", recipient=agent_id),
        )

        await runtime.stop_when_idle()

        assert [outcome.status for outcome in outcomes] == [
            DeliveryStatus.DELIVERED,
            DeliveryStatus.DELIVERED,
            DeliveryStatus.DELIVERED,
        ]
        results = [
            _require_run_result(outcome.response_payload) for outcome in outcomes
        ]
        assert [result.final_output for result in results] == [
            "Coordinator:one",
            "Coordinator:two",
            "Coordinator:three",
        ]
        assert not agent.is_running
        assert agent.run_state is not None
        assert agent.run_state.turn_count == 3
        assert len(agent.run_state.responses) == 3

    asyncio.run(scenario())
