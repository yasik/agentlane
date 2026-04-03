import asyncio

import agentlane.harness.context as harness_context
import agentlane.harness.memory as harness_memory
from agentlane.harness import Agent, Runner, RunnerHooks, Task, UserMessage
from agentlane.messaging import (
    AgentId,
    AgentType,
    CorrelationId,
    DeliveryStatus,
    MessageContext,
)
from agentlane.models import MessageDict
from agentlane.runtime import (
    CancellationToken,
    Engine,
    SingleThreadedRuntimeEngine,
    on_message,
)


class _CounterTask(Task):
    def __init__(self, engine: Engine, prefix: str) -> None:
        super().__init__(engine)
        self._prefix = prefix
        self.calls: list[str] = []

    @on_message
    async def handle(self, payload: str, context: MessageContext) -> object:
        _ = context
        self.calls.append(payload)
        return {
            "prefix": self._prefix,
            "count": len(self.calls),
            "task_id": self.task_id.key.value,
        }


class _RecorderTask(Task):
    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)
        self.observed_senders: list[AgentId | None] = []
        self.observed_correlations: list[CorrelationId | None] = []

    @on_message
    async def handle(self, payload: str, context: MessageContext) -> object:
        self.observed_senders.append(context.sender)
        self.observed_correlations.append(context.correlation_id)
        return payload.upper()


class _RelayTask(Task):
    def __init__(self, engine: Engine, target: AgentId) -> None:
        super().__init__(engine)
        self._target = target

    @on_message
    async def handle(self, payload: str, context: MessageContext) -> object:
        outcome = await self.send_message(
            payload,
            recipient=self._target,
            correlation_id=context.correlation_id,
        )
        if outcome.status != DeliveryStatus.DELIVERED:
            return {"status": outcome.status.value}
        return {
            "status": outcome.status.value,
            "response": outcome.response_payload,
        }


class _FakeAgent(Agent):
    pass


class _FakeRunner(Runner):
    async def run(
        self,
        agent: Task,
        messages: list[MessageDict],
        *,
        hooks: RunnerHooks | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> list[object]:
        _ = agent
        _ = messages
        _ = hooks
        _ = cancellation_token
        return []


def test_harness_public_exports_are_defined() -> None:
    assert Task is not None
    assert Agent is not None
    assert Runner is not None
    assert RunnerHooks is not None
    assert UserMessage is not None


def test_harness_subpackages_import() -> None:
    assert harness_context.__all__ == []
    assert harness_memory.__all__ == []


def test_task_register_creates_runtime_factory_instances() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()

        registered_type = _CounterTask.register(
            runtime,
            "counter-task",
            prefix="factory",
        )

        first = await runtime.send_message(
            "one",
            recipient=AgentId.from_values("counter-task", "k1"),
        )
        second = await runtime.send_message(
            "two",
            recipient=AgentId.from_values("counter-task", "k1"),
        )
        third = await runtime.send_message(
            "three",
            recipient=AgentId.from_values("counter-task", "k2"),
        )

        await runtime.stop_when_idle()

        assert registered_type == AgentType("counter-task")
        assert first.response_payload == {
            "prefix": "factory",
            "count": 1,
            "task_id": "k1",
        }
        assert second.response_payload == {
            "prefix": "factory",
            "count": 2,
            "task_id": "k1",
        }
        assert third.response_payload == {
            "prefix": "factory",
            "count": 1,
            "task_id": "k2",
        }

    asyncio.run(scenario())


def test_task_bind_registers_one_stateful_instance() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        task_id = AgentId.from_values("counter-task", "stateful")

        task = _CounterTask.bind(
            runtime,
            task_id,
            prefix="bound",
        )

        first = await runtime.send_message("one", recipient=task_id)
        second = await runtime.send_message("two", recipient=task_id)

        await runtime.stop_when_idle()

        assert task.task_id == task_id
        assert task.calls == ["one", "two"]
        assert first.response_payload == {
            "prefix": "bound",
            "count": 1,
            "task_id": "stateful",
        }
        assert second.response_payload == {
            "prefix": "bound",
            "count": 2,
            "task_id": "stateful",
        }

    asyncio.run(scenario())


def test_task_can_orchestrate_other_runtime_recipients() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        relay_id = AgentId.from_values("relay-task", "relay-1")
        target_id = AgentId.from_values("recorder-task", "recorder-1")

        recorder = _RecorderTask.bind(runtime, target_id)
        _RelayTask.bind(runtime, relay_id, target=target_id)

        correlation_id = CorrelationId.new()
        outcome = await runtime.send_message(
            "hello",
            recipient=relay_id,
            correlation_id=correlation_id,
        )

        await runtime.stop_when_idle()

        assert outcome.status == DeliveryStatus.DELIVERED
        assert outcome.response_payload == {
            "status": DeliveryStatus.DELIVERED.value,
            "response": "HELLO",
        }
        assert recorder.observed_senders == [relay_id]
        assert recorder.observed_correlations == [correlation_id]

    asyncio.run(scenario())


def test_runner_can_be_specialized() -> None:
    assert isinstance(_FakeRunner(), Runner)


def test_task_and_agent_extend_base_harness_contracts() -> None:
    assert issubclass(Agent, Task)
    assert issubclass(_FakeAgent, Task)
