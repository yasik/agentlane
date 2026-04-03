import asyncio

from agentlane.harness import Agent, Runner, RunnerHooks, Task, UserMessage
from agentlane.messaging import AgentId, DeliveryStatus
from agentlane.models import MessageDict
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine


def _message(role: str, content: object) -> MessageDict:
    """Build one expected conversation message for assertions."""
    return {
        "role": role,
        "content": content,
    }


def _copy_messages(messages: list[MessageDict]) -> list[MessageDict]:
    """Return a shallow copy of message dictionaries for assertions."""
    return [dict(message) for message in messages]


class _RecordingRunner(Runner):
    def __init__(self, *, block_first_turn: bool = False) -> None:
        self.calls: list[list[MessageDict]] = []
        self._block_first_turn = block_first_turn
        self.first_turn_started = asyncio.Event()
        self.release_first_turn = asyncio.Event()

    async def run(
        self,
        agent: Task,
        messages: list[MessageDict],
        *,
        hooks: RunnerHooks | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> list[object]:
        _ = hooks
        _ = cancellation_token
        self.calls.append(_copy_messages(messages))

        if self._block_first_turn and len(self.calls) == 1:
            self.first_turn_started.set()
            await self.release_first_turn.wait()

        if not isinstance(agent, Agent):
            raise AssertionError(
                "Expected the default harness Agent in lifecycle tests."
            )

        reply_text = f"{agent.name}:{_last_user_content(messages)}"
        messages.append(_message("assistant", reply_text))
        return [reply_text]


class _TestableAgent(Agent):
    async def queue_for_test(self, content: object) -> object:
        return await self._enqueue_user_message(content)


def _last_user_content(messages: list[MessageDict]) -> object:
    """Return the most recent user content from one conversation."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content")
    raise AssertionError("Expected at least one user message in the conversation.")


def test_agent_starts_new_conversation_with_system_prompt() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = _RecordingRunner()

        Agent.register(
            runtime,
            "assistant-agent",
            runner=runner,
            name="Support",
            description="Handles support requests",
            system_prompt="You are a helpful support agent.",
            skills=["triage"],
        )

        outcome = await runtime.send_message(
            UserMessage(content="hello"),
            recipient=AgentId.from_values("assistant-agent", "session-1"),
        )

        await runtime.stop_when_idle()

        assert outcome.status == DeliveryStatus.DELIVERED
        assert outcome.response_payload == "Support:hello"
        assert runner.calls == [
            [
                _message("system", "You are a helpful support agent."),
                _message("user", "hello"),
            ]
        ]

    asyncio.run(scenario())


def test_agent_continues_existing_history_after_idle() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = _RecordingRunner()
        agent_id = AgentId.from_values("assistant-agent", "session-2")
        agent = Agent.bind(
            runtime,
            agent_id,
            runner=runner,
            name="Planner",
            description="Plans next steps",
            system_prompt="You plan carefully.",
            skills=["analysis"],
            context={"team": "ops"},
            memory={"kind": "ephemeral"},
        )

        first = await runtime.send_message(
            Agent.user_message("first"),
            recipient=agent_id,
        )
        second = await runtime.send_message(
            Agent.user_message("second"),
            recipient=agent_id,
        )

        await runtime.stop_when_idle()

        assert first.response_payload == "Planner:first"
        assert second.response_payload == "Planner:second"
        assert agent.name == "Planner"
        assert agent.description == "Plans next steps"
        assert agent.system_prompt == "You plan carefully."
        assert agent.skills == ("analysis",)
        assert agent.context == {"team": "ops"}
        assert agent.memory == {"kind": "ephemeral"}
        assert not agent.is_running
        assert agent.pending_user_message_count == 0
        assert runner.calls == [
            [
                _message("system", "You plan carefully."),
                _message("user", "first"),
            ],
            [
                _message("system", "You plan carefully."),
                _message("user", "first"),
                _message("assistant", "Planner:first"),
                _message("user", "second"),
            ],
        ]
        assert agent.message_history == [
            _message("system", "You plan carefully."),
            _message("user", "first"),
            _message("assistant", "Planner:first"),
            _message("user", "second"),
            _message("assistant", "Planner:second"),
        ]

    asyncio.run(scenario())


def test_agent_queues_messages_for_next_turn_while_running() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = _RecordingRunner(block_first_turn=True)
        agent_id = AgentId.from_values("assistant-agent", "session-3")
        agent = _TestableAgent.bind(
            runtime,
            agent_id,
            runner=runner,
            name="Researcher",
            system_prompt="You research methodically.",
        )

        first_turn = asyncio.create_task(agent.queue_for_test("first"))
        await asyncio.wait_for(runner.first_turn_started.wait(), timeout=1.0)

        second_turn = asyncio.create_task(agent.queue_for_test("second"))
        await asyncio.sleep(0)

        assert agent.is_running
        assert agent.pending_user_message_count == 1

        runner.release_first_turn.set()

        first_result = await first_turn
        second_result = await second_turn

        assert first_result == "Researcher:first"
        assert second_result == "Researcher:second"
        assert not agent.is_running
        assert agent.pending_user_message_count == 0
        assert runner.calls == [
            [
                _message("system", "You research methodically."),
                _message("user", "first"),
            ],
            [
                _message("system", "You research methodically."),
                _message("user", "first"),
                _message("assistant", "Researcher:first"),
                _message("user", "second"),
            ],
        ]

    asyncio.run(scenario())


def test_agent_drains_multiple_queued_messages_one_turn_at_a_time() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = _RecordingRunner(block_first_turn=True)
        agent_id = AgentId.from_values("assistant-agent", "session-3b")
        agent = _TestableAgent.bind(
            runtime,
            agent_id,
            runner=runner,
            name="Sequencer",
            system_prompt="You process one message per turn.",
        )

        first_turn = asyncio.create_task(agent.queue_for_test("first"))
        await asyncio.wait_for(runner.first_turn_started.wait(), timeout=1.0)

        second_turn = asyncio.create_task(agent.queue_for_test("second"))
        third_turn = asyncio.create_task(agent.queue_for_test("third"))
        await asyncio.sleep(0)

        assert agent.pending_user_message_count == 2

        runner.release_first_turn.set()

        assert await first_turn == "Sequencer:first"
        assert await second_turn == "Sequencer:second"
        assert await third_turn == "Sequencer:third"
        assert runner.calls == [
            [
                _message("system", "You process one message per turn."),
                _message("user", "first"),
            ],
            [
                _message("system", "You process one message per turn."),
                _message("user", "first"),
                _message("assistant", "Sequencer:first"),
                _message("user", "second"),
            ],
            [
                _message("system", "You process one message per turn."),
                _message("user", "first"),
                _message("assistant", "Sequencer:first"),
                _message("user", "second"),
                _message("assistant", "Sequencer:second"),
                _message("user", "third"),
            ],
        ]

    asyncio.run(scenario())


def test_runtime_serializes_multiple_messages_to_same_agent_id() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = _RecordingRunner()
        agent_id = AgentId.from_values("assistant-agent", "session-4")
        agent = Agent.bind(
            runtime,
            agent_id,
            runner=runner,
            name="Coordinator",
            system_prompt="You coordinate carefully.",
        )

        outcomes = await asyncio.gather(
            runtime.send_message(UserMessage(content="one"), recipient=agent_id),
            runtime.send_message(UserMessage(content="two"), recipient=agent_id),
            runtime.send_message(UserMessage(content="three"), recipient=agent_id),
        )

        await runtime.stop_when_idle()

        assert [outcome.status for outcome in outcomes] == [
            DeliveryStatus.DELIVERED,
            DeliveryStatus.DELIVERED,
            DeliveryStatus.DELIVERED,
        ]
        assert [outcome.response_payload for outcome in outcomes] == [
            "Coordinator:one",
            "Coordinator:two",
            "Coordinator:three",
        ]
        assert not agent.is_running
        assert runner.calls == [
            [
                _message("system", "You coordinate carefully."),
                _message("user", "one"),
            ],
            [
                _message("system", "You coordinate carefully."),
                _message("user", "one"),
                _message("assistant", "Coordinator:one"),
                _message("user", "two"),
            ],
            [
                _message("system", "You coordinate carefully."),
                _message("user", "one"),
                _message("assistant", "Coordinator:one"),
                _message("user", "two"),
                _message("assistant", "Coordinator:two"),
                _message("user", "three"),
            ],
        ]

    asyncio.run(scenario())
