import asyncio

from agentlane.agents import on_message
from agentlane.messaging import (
    AgentId,
    AgentType,
    DeliveryStatus,
    MessageContext,
    Subscription,
    SubscriptionKind,
    TopicId,
)
from agentlane.runtime import RuntimeEngine


class CounterAgent:
    def __init__(self) -> None:
        self.calls: list[object] = []

    @on_message
    async def handle(self, payload: str, context: MessageContext) -> object:
        _ = context
        self.calls.append(payload)
        return {"count": len(self.calls)}


class PingMessage:
    def __init__(self, value: str) -> None:
        self.value = value


class PongMessage:
    def __init__(self, value: str) -> None:
        self.value = value


class MultiHandlerAgent:
    @on_message
    async def handle_ping(
        self, payload: PingMessage, context: MessageContext
    ) -> object:
        _ = context
        return f"ping:{payload.value}"

    @on_message
    async def handle_pong(
        self, payload: PongMessage, context: MessageContext
    ) -> object:
        _ = context
        return f"pong:{payload.value}"

    @on_message
    async def handle_int(self, payload: int, context: MessageContext) -> object:
        _ = context
        return f"int:{payload}"


def test_runtime_reuses_registered_instance_by_agent_id() -> None:
    async def scenario() -> None:
        runtime = RuntimeEngine()
        agent_id = AgentId.from_values("counter", "stateful")
        agent = CounterAgent()
        runtime.register_instance(agent_id, agent)

        first = await runtime.send_message("one", recipient=agent_id)
        second = await runtime.send_message("two", recipient=agent_id)

        await runtime.stop_when_idle()

        assert first.status == DeliveryStatus.DELIVERED
        assert second.status == DeliveryStatus.DELIVERED
        assert second.response_payload == {"count": 2}
        assert agent.calls == ["one", "two"]

    asyncio.run(scenario())


def test_runtime_creates_isolated_instances_for_unique_keys() -> None:
    async def scenario() -> None:
        runtime = RuntimeEngine()
        runtime.register_factory("counter", CounterAgent)

        one = await runtime.send_message("first", recipient="counter", key="k1")
        two = await runtime.send_message("second", recipient="counter", key="k1")
        three = await runtime.send_message("third", recipient="counter", key="k2")

        await runtime.stop_when_idle()

        assert one.response_payload == {"count": 1}
        assert two.response_payload == {"count": 2}
        assert three.response_payload == {"count": 1}

    asyncio.run(scenario())


def test_runtime_rejects_ambiguous_keyless_target() -> None:
    async def scenario() -> None:
        runtime = RuntimeEngine()
        runtime.register_instance(AgentId.from_values("worker", "a"), CounterAgent())
        runtime.register_instance(AgentId.from_values("worker", "b"), CounterAgent())

        outcome = await runtime.send_message("ping", recipient="worker")

        await runtime.stop_when_idle()

        assert outcome.status == DeliveryStatus.POLICY_REJECTED

    asyncio.run(scenario())


def test_publish_returns_enqueue_ack_only() -> None:
    async def scenario() -> None:
        runtime = RuntimeEngine()
        received: list[object] = []

        class Listener:
            @on_message
            async def handle(self, payload: dict, context: MessageContext) -> object:
                _ = context
                received.append(payload)
                return None

        runtime.register_factory(AgentType("listener"), Listener)
        runtime.add_subscription(
            Subscription(
                kind=SubscriptionKind.TYPE_EXACT,
                agent_type=AgentType("listener"),
                topic_pattern="alerts",
            )
        )

        ack = await runtime.publish_message(
            {"event": "ready"},
            topic=TopicId(type="alerts", source="session-1"),
        )
        await runtime.stop_when_idle()

        assert ack.enqueued_recipient_count == 1
        assert received == [{"event": "ready"}]

    asyncio.run(scenario())


def test_per_agent_ordering_for_stateful_handler() -> None:
    async def scenario() -> None:
        runtime = RuntimeEngine()
        runtime.register_factory("ordered", CounterAgent)
        recipient = AgentId.from_values("ordered", "k")

        first, second, third = await asyncio.gather(
            runtime.send_message("one", recipient=recipient),
            runtime.send_message("two", recipient=recipient),
            runtime.send_message("three", recipient=recipient),
        )
        await runtime.stop_when_idle()

        assert first.response_payload == {"count": 1}
        assert second.response_payload == {"count": 2}
        assert third.response_payload == {"count": 3}

    asyncio.run(scenario())


def test_stop_cancels_inflight_and_queued_deliveries() -> None:
    async def scenario() -> None:
        runtime = RuntimeEngine()
        first_started = asyncio.Event()

        class BlockingAgent:
            @on_message
            async def handle(self, payload: str, context: MessageContext) -> object:
                _ = payload
                first_started.set()
                await context.cancellation_token.wait_cancelled()
                await asyncio.Event().wait()
                return None

        recipient = AgentId.from_values("blocking", "k")
        runtime.register_factory("blocking", BlockingAgent)

        first_send = asyncio.create_task(
            runtime.send_message("first", recipient=recipient)
        )
        await asyncio.wait_for(first_started.wait(), timeout=1.0)
        second_send = asyncio.create_task(
            runtime.send_message("second", recipient=recipient)
        )
        await asyncio.sleep(0.01)

        await runtime.stop()

        first_outcome = await first_send
        second_outcome = await second_send

        assert first_outcome.status == DeliveryStatus.CANCELED
        assert second_outcome.status == DeliveryStatus.CANCELED

    asyncio.run(scenario())


def test_multiple_on_message_handlers_route_by_payload_type() -> None:
    async def scenario() -> None:
        runtime = RuntimeEngine()
        recipient = AgentId.from_values("multi-handler", "k")
        runtime.register_factory("multi-handler", MultiHandlerAgent)

        ping_outcome = await runtime.send_message(PingMessage("a"), recipient=recipient)
        pong_outcome = await runtime.send_message(PongMessage("b"), recipient=recipient)
        int_outcome = await runtime.send_message(123, recipient=recipient)

        await runtime.stop_when_idle()

        assert ping_outcome.status == DeliveryStatus.DELIVERED
        assert pong_outcome.status == DeliveryStatus.DELIVERED
        assert int_outcome.status == DeliveryStatus.DELIVERED
        assert ping_outcome.response_payload == "ping:a"
        assert pong_outcome.response_payload == "pong:b"
        assert int_outcome.response_payload == "int:123"

    asyncio.run(scenario())
