import asyncio
from collections.abc import Sequence

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    Runner,
    RunnerHooks,
    RunResult,
    RunState,
    Task,
)
from agentlane.harness._run import RunHistoryItem, copy_run_state
from agentlane.harness.shims import Shim
from agentlane.messaging import AgentId, DeliveryStatus
from agentlane.models import ModelResponse
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine


def make_assistant_response(content: str) -> ModelResponse:
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


def _last_user_input(run_state: RunState) -> object:
    """Return the latest user-side input represented in one run state."""
    for item in reversed(run_state.history):
        if isinstance(item, ModelResponse):
            continue
        return item
    return None


def _require_run_result(payload: object | None) -> RunResult:
    """Return a strongly typed lifecycle result for assertions."""
    if not isinstance(payload, RunResult):
        raise AssertionError("Expected a `RunResult` payload from the runtime.")
    return payload


class _NoopShim(Shim):
    @property
    def name(self) -> str:
        return "noop"


class _SharedRecordingHooks(RunnerHooks):
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


class _HookShim(Shim):
    def __init__(self, hook: RunnerHooks) -> None:
        self._hook = hook

    @property
    def name(self) -> str:
        return "hook-shim"

    def runner_hooks(self) -> tuple[RunnerHooks, ...]:
        return (self._hook,)


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
        hooks: RunnerHooks | Sequence[RunnerHooks] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        _ = hooks
        _ = cancellation_token
        copied = copy_run_state(state)
        assert copied is not None
        self.calls.append(copied)

        if self._block_first_turn and len(self.calls) == 1:
            self.first_turn_started.set()
            await self.release_first_turn.wait()

        if not isinstance(agent, Agent):
            raise AssertionError(
                "Expected the default harness Agent in lifecycle tests."
            )

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


class _HookAwareRecordingRunner(_RecordingRunner):
    def __init__(self) -> None:
        super().__init__()
        self.seen_hooks: list[RunnerHooks | None] = []

    async def run(
        self,
        agent: Task,
        state: RunState,
        *,
        hooks: RunnerHooks | Sequence[RunnerHooks] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        if isinstance(hooks, Sequence):
            raise AssertionError("Lifecycle should pass one resolved hook object.")
        self.seen_hooks.append(hooks)
        if hooks is not None:
            await hooks.on_agent_start(agent, state)
        result: RunResult | None = None
        try:
            result = await super().run(
                agent,
                state,
                hooks=hooks,
                cancellation_token=cancellation_token,
            )
            return result
        finally:
            if hooks is not None:
                await hooks.on_agent_end(agent, result)


class _TestableAgent(Agent):
    async def queue_for_test(
        self, run_input: str | list[RunHistoryItem] | RunState
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
                instructions="You are a helpful support agent.",
                history=["hello"],
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
                shims=(_NoopShim(),),
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
        agent_shims: Sequence[Shim] = agent.shims or ()
        assert len(agent_shims) == 1
        assert not agent.is_running
        assert agent.pending_input_count == 0
        assert len(runner.calls) == 2
        assert runner.calls[0] == RunState(
            instructions="You plan carefully.",
            history=["first"],
            responses=[],
            turn_count=0,
        )
        assert runner.calls[1].history[0] == "first"
        assert runner.calls[1].turn_count == 1
        assert runner.calls[1].history[2] == "second"
        assert agent.run_state is not None
        assert agent.run_state.history[0] == "first"
        assert agent.run_state.turn_count == 2
        assert len(agent.run_state.responses) == 2

    asyncio.run(scenario())


def test_agent_caches_composed_hooks_from_explicit_and_shim_sources() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = _HookAwareRecordingRunner()
        events: list[tuple[str, str]] = []
        agent_id = AgentId.from_values("assistant-agent", "session-hooks")
        Agent.bind(
            runtime,
            agent_id,
            runner=runner,
            descriptor=AgentDescriptor(
                name="Planner",
                instructions="You plan carefully.",
                shims=(_HookShim(_SharedRecordingHooks("shim", events)),),
            ),
            hooks=(
                _SharedRecordingHooks("explicit-first", events),
                _SharedRecordingHooks("explicit-second", events),
            ),
        )

        first = await runtime.send_message("first", recipient=agent_id)
        second = await runtime.send_message("second", recipient=agent_id)

        await runtime.stop_when_idle()

        first_result = _require_run_result(first.response_payload)
        second_result = _require_run_result(second.response_payload)
        assert first_result.final_output == "Planner:first"
        assert second_result.final_output == "Planner:second"
        assert len(runner.seen_hooks) == 2
        assert runner.seen_hooks[0] is runner.seen_hooks[1]
        assert events == [
            ("explicit-first", "agent_start"),
            ("explicit-second", "agent_start"),
            ("shim", "agent_start"),
            ("explicit-first", "agent_end"),
            ("explicit-second", "agent_end"),
            ("shim", "agent_end"),
            ("explicit-first", "agent_start"),
            ("explicit-second", "agent_start"),
            ("shim", "agent_start"),
            ("explicit-first", "agent_end"),
            ("explicit-second", "agent_end"),
            ("shim", "agent_end"),
        ]

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
        assert runner.calls[1].history[2] == "second"

    asyncio.run(scenario())


def test_agent_continues_from_preloaded_run_state() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = _RecordingRunner()
        agent_id = AgentId.from_values("assistant-agent", "session-recovered")
        prior_response = make_assistant_response("Recovered:before")
        recovered_state = RunState(
            instructions="You should not be reseeded.",
            history=["before", prior_response],
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
        assert runner.calls[0].history[0] == "before"
        assert runner.calls[0].turn_count == 1
        assert runner.calls[0].history[2] == "after"
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
            instructions="You help with support.",
            history=["before", make_assistant_response("Support:before")],
            responses=[make_assistant_response("Support:before")],
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
