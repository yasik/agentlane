import asyncio
import inspect
from typing import Any, cast

import pytest

from agentlane.messaging import (
    AgentId,
    AgentType,
    CorrelationId,
    DeliveryMode,
    DeliveryOutcome,
    DeliveryStatus,
    MessageContext,
    TopicId,
)
from agentlane.runtime import (
    BaseAgent,
    DistributedRuntimeEngine,
    SingleThreadedRuntimeEngine,
    distributed_runtime,
    on_message,
    single_threaded_runtime,
)


class _ProtocolAgentMixin:
    def __init__(self) -> None:
        self._id: AgentId | None = None

    @property
    def id(self) -> AgentId:
        if self._id is None:
            raise RuntimeError("Agent id was not bound by runtime.")
        return self._id

    def bind_agent_id(self, agent_id: AgentId) -> None:
        self._id = agent_id


class CounterAgent(_ProtocolAgentMixin):
    def __init__(self) -> None:
        super().__init__()
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


class MultiHandlerAgent(_ProtocolAgentMixin):
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


class RelayAgent(BaseAgent):
    @on_message
    async def handle(self, payload: str, context: MessageContext) -> object:
        outcome = await self.send_message(
            payload.upper(),
            recipient=AgentId.from_values("sink", "s1"),
            correlation_id=context.correlation_id,
        )
        if outcome.status != DeliveryStatus.DELIVERED:
            return outcome.status.value
        return outcome.response_payload


class PublisherAgent(BaseAgent):
    @on_message
    async def handle(self, payload: str, context: MessageContext) -> object:
        await self.publish_message(
            {"event": payload},
            topic=TopicId.from_values(type_value="updates", route_key="rk"),
            correlation_id=context.correlation_id,
        )
        return "published"


def test_runtime_reuses_registered_instance_by_agent_id() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
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
        runtime = SingleThreadedRuntimeEngine()
        runtime.register_factory("counter", lambda _engine: CounterAgent())

        one = await runtime.send_message(
            "first", recipient=AgentId.from_values("counter", "k1")
        )
        two = await runtime.send_message(
            "second", recipient=AgentId.from_values("counter", "k1")
        )
        three = await runtime.send_message(
            "third", recipient=AgentId.from_values("counter", "k2")
        )

        await runtime.stop_when_idle()

        assert one.response_payload == {"count": 1}
        assert two.response_payload == {"count": 2}
        assert three.response_payload == {"count": 1}

    asyncio.run(scenario())


def test_runtime_rejects_ambiguous_keyless_target() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runtime.register_instance(AgentId.from_values("worker", "a"), CounterAgent())
        runtime.register_instance(AgentId.from_values("worker", "b"), CounterAgent())

        outcome = await runtime.send_message("ping", recipient="worker")

        await runtime.stop_when_idle()

        assert outcome.status == DeliveryStatus.POLICY_REJECTED

    asyncio.run(scenario())


def test_publish_returns_enqueue_ack_only() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        received: list[object] = []

        class Listener(_ProtocolAgentMixin):
            @on_message
            async def handle(
                self,
                payload: dict[str, object],
                context: MessageContext,
            ) -> object:
                _ = context
                received.append(payload)
                return None

        runtime.register_factory(AgentType("listener"), lambda _engine: Listener())
        runtime.subscribe_exact(
            topic_type="alerts",
            agent_type=AgentType("listener"),
        )

        ack = await runtime.publish_message(
            {"event": "ready"},
            topic=TopicId.from_values(type_value="alerts", route_key="session-1"),
        )
        await runtime.stop_when_idle()

        assert ack.enqueued_recipient_count == 1
        assert received == [{"event": "ready"}]

    asyncio.run(scenario())


def test_base_agent_can_send_message_with_runtime_capability() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        relay_id = AgentId.from_values("relay", "r1")
        observed_senders: list[AgentId | None] = []
        observed_correlations: list[CorrelationId | None] = []

        class SinkAgent(_ProtocolAgentMixin):
            @on_message
            async def handle(self, payload: str, context: MessageContext) -> object:
                observed_senders.append(context.sender)
                observed_correlations.append(context.correlation_id)
                return f"sink:{payload}"

        runtime.register_factory("relay", RelayAgent)
        runtime.register_factory("sink", lambda _engine: SinkAgent())

        correlation_id = CorrelationId.new()
        outcome = await runtime.send_message(
            "ping",
            recipient=relay_id,
            correlation_id=correlation_id,
        )
        await runtime.stop_when_idle()

        assert outcome.status == DeliveryStatus.DELIVERED
        assert outcome.response_payload == "sink:PING"
        assert observed_senders == [relay_id]
        assert observed_correlations == [correlation_id]

    asyncio.run(scenario())


def test_base_agent_can_publish_message_with_runtime_capability() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        publisher_id = AgentId.from_values("publisher", "p1")
        received: list[dict[str, object]] = []
        observed_senders: list[AgentId | None] = []
        observed_correlations: list[CorrelationId | None] = []

        class ListenerAgent(_ProtocolAgentMixin):
            @on_message
            async def handle(
                self,
                payload: dict[str, object],
                context: MessageContext,
            ) -> object:
                received.append(payload)
                observed_senders.append(context.sender)
                observed_correlations.append(context.correlation_id)
                return None

        runtime.register_factory("publisher", PublisherAgent)
        runtime.register_factory("listener", lambda _engine: ListenerAgent())
        runtime.subscribe_exact(topic_type="updates", agent_type="listener")

        correlation_id = CorrelationId.new()
        outcome = await runtime.send_message(
            "ready",
            recipient=publisher_id,
            correlation_id=correlation_id,
        )
        await runtime.stop_when_idle()

        assert outcome.status == DeliveryStatus.DELIVERED
        assert outcome.response_payload == "published"
        assert received == [{"event": "ready"}]
        assert observed_senders == [publisher_id]
        assert observed_correlations == [correlation_id]

    asyncio.run(scenario())


def test_per_agent_ordering_for_stateful_handler() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runtime.register_factory("ordered", lambda _engine: CounterAgent())
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
        runtime = SingleThreadedRuntimeEngine()
        first_started = asyncio.Event()

        class BlockingAgent(_ProtocolAgentMixin):
            @on_message
            async def handle(self, payload: str, context: MessageContext) -> object:
                _ = payload
                _ = context
                first_started.set()
                await asyncio.Event().wait()
                return None

        recipient = AgentId.from_values("blocking", "k")
        runtime.register_factory("blocking", lambda _engine: BlockingAgent())

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
        runtime = SingleThreadedRuntimeEngine()
        recipient = AgentId.from_values("multi-handler", "k")
        runtime.register_factory("multi-handler", lambda _engine: MultiHandlerAgent())

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


def test_runtime_fails_when_agent_has_no_on_message_handler() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()

        class NoHandlerAgent(_ProtocolAgentMixin):
            async def handle(self, payload: str, context: MessageContext) -> object:
                _ = context
                return payload

        runtime.register_factory("no-handler", lambda _engine: NoHandlerAgent())
        outcome = await runtime.send_message(
            "ping",
            recipient=AgentId.from_values("no-handler", "k"),
        )
        await runtime.stop_when_idle()

        assert outcome.status == DeliveryStatus.HANDLER_ERROR
        assert outcome.error is not None
        assert "does not define an `@on_message` handler" in outcome.error.message

    asyncio.run(scenario())


def test_runtime_fails_when_on_message_handler_missing_context() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()

        class MissingContextAgent(_ProtocolAgentMixin):
            @on_message
            async def handle(self, payload: str) -> object:
                return payload

        runtime.register_factory(
            "missing-context", lambda _engine: MissingContextAgent()
        )
        outcome = await runtime.send_message(
            "ping",
            recipient=AgentId.from_values("missing-context", "k"),
        )
        await runtime.stop_when_idle()

        assert outcome.status == DeliveryStatus.HANDLER_ERROR
        assert outcome.error is not None
        assert "must have signature `(payload, context)`" in outcome.error.message

    asyncio.run(scenario())


def test_runtime_fails_when_on_message_handler_has_wrong_arity() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()

        class WrongArityAgent(_ProtocolAgentMixin):
            @on_message
            async def handle(
                self,
                payload: str,
                context: MessageContext,
                extra: int,
            ) -> object:
                _ = context
                _ = extra
                return payload

        runtime.register_factory("wrong-arity", lambda _engine: WrongArityAgent())
        outcome = await runtime.send_message(
            "ping",
            recipient=AgentId.from_values("wrong-arity", "k"),
        )
        await runtime.stop_when_idle()

        assert outcome.status == DeliveryStatus.HANDLER_ERROR
        assert outcome.error is not None
        assert "must have signature `(payload, context)`" in outcome.error.message

    asyncio.run(scenario())


def test_runtime_fails_when_on_message_handler_missing_payload_annotation() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()

        class MissingPayloadAnnotationAgent(_ProtocolAgentMixin):
            @on_message
            async def handle(
                self,
                payload: object,
                context: MessageContext,
            ) -> object:
                _ = payload
                _ = context
                return None

        signature = inspect.signature(MissingPayloadAnnotationAgent.handle)
        parameters = list(signature.parameters.values())
        cast(Any, MissingPayloadAnnotationAgent.handle).__signature__ = (
            signature.replace(
                parameters=[
                    parameters[0],
                    parameters[1].replace(annotation=inspect.Signature.empty),
                    parameters[2],
                ]
            )
        )

        runtime.register_factory(
            "missing-payload-annotation",
            lambda _engine: MissingPayloadAnnotationAgent(),
        )
        outcome = await runtime.send_message(
            "ping",
            recipient=AgentId.from_values("missing-payload-annotation", "k"),
        )
        await runtime.stop_when_idle()

        assert outcome.status == DeliveryStatus.HANDLER_ERROR
        assert outcome.error is not None
        assert (
            "must declare an explicit payload type annotation" in outcome.error.message
        )

    asyncio.run(scenario())


def test_runtime_fails_when_on_message_handler_payload_annotation_not_concrete() -> (
    None
):
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()

        class NonConcretePayloadTypeAgent(_ProtocolAgentMixin):
            @on_message
            async def handle(
                self,
                payload: str | int,
                context: MessageContext,
            ) -> object:
                _ = payload
                _ = context
                return None

        runtime.register_factory(
            "non-concrete-payload-type",
            lambda _engine: NonConcretePayloadTypeAgent(),
        )
        outcome = await runtime.send_message(
            "ping",
            recipient=AgentId.from_values("non-concrete-payload-type", "k"),
        )
        await runtime.stop_when_idle()

        assert outcome.status == DeliveryStatus.HANDLER_ERROR
        assert outcome.error is not None
        assert "must annotate payload with a concrete type" in outcome.error.message

    asyncio.run(scenario())


def test_runtime_resolves_string_payload_annotations() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()

        class StringAnnotatedPayloadAgent(_ProtocolAgentMixin):
            @on_message
            async def handle(
                self,
                payload: "str",
                context: "MessageContext",
            ) -> object:
                _ = context
                return payload.upper()

        runtime.register_factory(
            "string-annotated-payload",
            lambda _engine: StringAnnotatedPayloadAgent(),
        )
        outcome = await runtime.send_message(
            "ping",
            recipient=AgentId.from_values("string-annotated-payload", "k"),
        )
        await runtime.stop_when_idle()

        assert outcome.status == DeliveryStatus.DELIVERED
        assert outcome.response_payload == "PING"

    asyncio.run(scenario())


def test_runtime_fails_when_on_message_handlers_are_ambiguous() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()

        class AmbiguousHandlersAgent(_ProtocolAgentMixin):
            @on_message
            async def handle_a(self, payload: str, context: MessageContext) -> object:
                _ = context
                return payload

            @on_message
            async def handle_b(self, payload: str, context: MessageContext) -> object:
                _ = context
                return payload

        runtime.register_factory(
            "ambiguous-handlers", lambda _engine: AmbiguousHandlersAgent()
        )
        outcome = await runtime.send_message(
            "ping",
            recipient=AgentId.from_values("ambiguous-handlers", "k"),
        )
        await runtime.stop_when_idle()

        assert outcome.status == DeliveryStatus.HANDLER_ERROR
        assert outcome.error is not None
        assert "ambiguous `@on_message` handlers" in outcome.error.message
        assert "handle_a" in outcome.error.message
        assert "handle_b" in outcome.error.message

    asyncio.run(scenario())


def test_runtime_fails_when_on_message_handler_is_not_async() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()

        class SyncHandlerAgent(_ProtocolAgentMixin):
            def _sync_handle(self, payload: str, context: MessageContext) -> object:
                _ = context
                return payload

            handle = on_message(cast(Any, _sync_handle))

        runtime.register_factory("sync-handler", lambda _engine: SyncHandlerAgent())
        outcome = await runtime.send_message(
            "ping",
            recipient=AgentId.from_values("sync-handler", "k"),
        )
        await runtime.stop_when_idle()

        assert outcome.status == DeliveryStatus.HANDLER_ERROR
        assert outcome.error is not None
        assert "must be declared as `async def`" in outcome.error.message

    asyncio.run(scenario())


def test_different_agent_ids_can_process_concurrently() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine(worker_count=2)
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        release = asyncio.Event()

        class ParallelAgent(_ProtocolAgentMixin):
            @on_message
            async def handle(self, payload: str, context: MessageContext) -> object:
                _ = context
                if payload == "one":
                    first_started.set()
                if payload == "two":
                    second_started.set()
                await release.wait()
                return payload

        runtime.register_factory("parallel", lambda _engine: ParallelAgent())

        one_task = asyncio.create_task(
            runtime.send_message(
                "one",
                recipient=AgentId.from_values("parallel", "one"),
            )
        )
        two_task = asyncio.create_task(
            runtime.send_message(
                "two",
                recipient=AgentId.from_values("parallel", "two"),
            )
        )

        await asyncio.wait_for(
            asyncio.gather(first_started.wait(), second_started.wait()),
            timeout=1.0,
        )
        release.set()

        one_outcome, two_outcome = await asyncio.gather(one_task, two_task)
        await runtime.stop_when_idle()

        assert one_outcome.status == DeliveryStatus.DELIVERED
        assert two_outcome.status == DeliveryStatus.DELIVERED
        assert one_outcome.response_payload == "one"
        assert two_outcome.response_payload == "two"

    asyncio.run(scenario())


def test_single_threaded_runtime_context_starts_and_drains_on_exit() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runtime.register_factory("counter", lambda _engine: CounterAgent())

        assert runtime.is_running is False
        async with single_threaded_runtime(runtime) as context:
            assert context is runtime
            assert runtime.is_running is True
            first = await runtime.send_message(
                "one",
                recipient=AgentId.from_values("counter", "k"),
            )
            second = await runtime.send_message(
                "two",
                recipient=AgentId.from_values("counter", "k"),
            )
            assert first.status == DeliveryStatus.DELIVERED
            assert second.status == DeliveryStatus.DELIVERED
            assert second.response_payload == {"count": 2}

        assert runtime.is_running is False

    asyncio.run(scenario())


def test_single_threaded_runtime_context_preserves_prestarted_runtime() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        await runtime.start()

        assert runtime.is_running is True
        async with single_threaded_runtime(runtime):
            assert runtime.is_running is True

        # Context manager must not teardown a runtime it did not start.
        assert runtime.is_running is True
        await runtime.stop()
        assert runtime.is_running is False

    asyncio.run(scenario())


def test_single_threaded_runtime_context_rejects_wrong_runtime_type() -> None:
    async def scenario() -> None:
        runtime = DistributedRuntimeEngine()
        with pytest.raises(ValueError):
            async with single_threaded_runtime(runtime):
                return

    asyncio.run(scenario())


def test_single_threaded_runtime_context_stops_immediately_on_exception() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        first_started = asyncio.Event()

        class BlockingAgent(_ProtocolAgentMixin):
            @on_message
            async def handle(self, payload: str, context: MessageContext) -> object:
                _ = payload
                _ = context
                first_started.set()
                await asyncio.Event().wait()
                return None

        recipient = AgentId.from_values("scope-blocking", "k")
        runtime.register_factory("scope-blocking", lambda _engine: BlockingAgent())

        first_send: asyncio.Task[DeliveryOutcome] | None = None
        second_send: asyncio.Task[DeliveryOutcome] | None = None
        try:
            async with single_threaded_runtime(runtime):
                first_send = asyncio.create_task(
                    runtime.send_message("first", recipient=recipient)
                )
                await asyncio.wait_for(first_started.wait(), timeout=1.0)
                second_send = asyncio.create_task(
                    runtime.send_message("second", recipient=recipient)
                )
                await asyncio.sleep(0.01)
                raise RuntimeError("fail scope")
        except RuntimeError:
            pass

        if first_send is None or second_send is None:
            raise AssertionError("Test setup failed to create both send tasks.")

        first_outcome = await first_send
        second_outcome = await second_send
        assert first_outcome.status == DeliveryStatus.CANCELED
        assert second_outcome.status == DeliveryStatus.CANCELED
        assert runtime.is_running is False

    asyncio.run(scenario())


def test_single_threaded_runtime_context_builds_default_runtime_when_none() -> None:
    async def scenario() -> None:
        async with single_threaded_runtime() as runtime:
            assert isinstance(runtime, SingleThreadedRuntimeEngine)
            assert runtime.is_running is True

        assert runtime.is_running is False

    asyncio.run(scenario())


def test_distributed_runtime_context_builds_default_runtime_when_none() -> None:
    async def scenario() -> None:
        async with distributed_runtime() as runtime:
            assert isinstance(runtime, DistributedRuntimeEngine)
            assert runtime.is_running is True

        assert runtime.is_running is False

    asyncio.run(scenario())


def test_runtime_subscription_convenience_api_round_trip() -> None:
    runtime = SingleThreadedRuntimeEngine()

    subscription_id = runtime.subscribe_exact(
        topic_type="alerts",
        agent_type="listener",
        delivery_mode=DeliveryMode.STATEFUL,
    )
    subscriptions = runtime.list_subscriptions()
    assert len(subscriptions) == 1
    assert subscriptions[0].id == subscription_id
    assert subscriptions[0].topic_pattern == "alerts"
    assert subscriptions[0].delivery_mode == DeliveryMode.STATEFUL

    runtime.unsubscribe(subscription_id)
    assert runtime.list_subscriptions() == ()


def test_publish_stateful_reuses_instance_for_same_route_key() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        observed_counts: list[int] = []

        class StatefulListener(_ProtocolAgentMixin):
            def __init__(self) -> None:
                super().__init__()
                self.count = 0

            @on_message
            async def handle(
                self,
                payload: dict[str, object],
                context: MessageContext,
            ) -> object:
                _ = payload
                _ = context
                self.count += 1
                observed_counts.append(self.count)
                return None

        runtime.register_factory("listener", lambda _engine: StatefulListener())
        runtime.subscribe_exact(
            topic_type="alerts",
            agent_type="listener",
            delivery_mode=DeliveryMode.STATEFUL,
        )

        await runtime.publish_message(
            {"event": "one"},
            topic=TopicId.from_values(type_value="alerts", route_key="session-1"),
        )
        await runtime.publish_message(
            {"event": "two"},
            topic=TopicId.from_values(type_value="alerts", route_key="session-1"),
        )
        await runtime.stop_when_idle()

        assert observed_counts == [1, 2]

    asyncio.run(scenario())


def test_publish_stateless_creates_transient_instance_per_delivery() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        observed_counts: list[int] = []

        class StatelessListener(_ProtocolAgentMixin):
            def __init__(self) -> None:
                super().__init__()
                self.count = 0

            @on_message
            async def handle(
                self,
                payload: dict[str, object],
                context: MessageContext,
            ) -> object:
                _ = payload
                _ = context
                self.count += 1
                observed_counts.append(self.count)
                return None

        runtime.register_factory("listener", lambda _engine: StatelessListener())
        runtime.subscribe_exact(
            topic_type="alerts",
            agent_type="listener",
            delivery_mode=DeliveryMode.STATELESS,
        )

        await runtime.publish_message(
            {"event": "one"},
            topic=TopicId.from_values(type_value="alerts", route_key="session-1"),
        )
        await runtime.publish_message(
            {"event": "two"},
            topic=TopicId.from_values(type_value="alerts", route_key="session-1"),
        )
        await runtime.stop_when_idle()

        assert observed_counts == [1, 1]

    asyncio.run(scenario())
