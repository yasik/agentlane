import asyncio
from base64 import b64encode
from typing import Any, cast

import pytest

from agentlane.messaging import (
    AgentId,
    CorrelationId,
    DeliveryMode,
    DeliveryStatus,
    MessageContext,
    MessageId,
    MessageKind,
    Payload,
    PayloadFormat,
    TopicId,
)
from agentlane.runtime import (
    WorkerAgentRuntime,
    WorkerAgentRuntimeHost,
    distributed_runtime,
    on_message,
)
from agentlane.transport import (
    create_default_serializer_registry,
    infer_content_type_for_value,
    infer_schema_id_for_value,
    payload_to_wire_payload,
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


def _payload_for_value(value: object) -> Payload:
    payload_data = value
    payload_format = PayloadFormat.JSON
    schema_source = value
    if isinstance(value, bytes):
        payload_format = PayloadFormat.BYTES
    elif isinstance(value, bytearray):
        payload_data = bytes(value)
        schema_source = payload_data
        payload_format = PayloadFormat.BYTES
    elif isinstance(value, memoryview):
        payload_data = value.tobytes()
        schema_source = payload_data
        payload_format = PayloadFormat.BYTES

    return Payload(
        schema_name=infer_schema_id_for_value(schema_source).value,
        content_type=infer_content_type_for_value(schema_source).value,
        format=payload_format,
        data=payload_data,
    )


def _wire_envelope_json(
    *,
    message: object,
    kind: MessageKind,
    recipient: AgentId | None = None,
    topic: TopicId | None = None,
) -> dict[str, object]:
    wire_payload = payload_to_wire_payload(
        _payload_for_value(message),
        registry=create_default_serializer_registry(),
    )
    return {
        "message_id": MessageId.new().value,
        "correlation_id": None,
        "kind": kind.value,
        "sender": None,
        "recipient": recipient.to_json() if recipient is not None else None,
        "topic": topic.to_json() if topic is not None else None,
        "payload": {
            "schema_id": wire_payload.schema_id.value,
            "content_type": wire_payload.content_type.value,
            "encoding": wire_payload.encoding.value,
            "body": b64encode(wire_payload.body).decode("ascii"),
        },
        "created_at_ms": 0,
        "deadline_ms": None,
        "trace_id": None,
        "idempotency_key": None,
    }


def test_worker_runtime_routes_direct_rpc_across_workers() -> None:
    async def scenario() -> None:
        host = WorkerAgentRuntimeHost(address="127.0.0.1:0")
        await host.start()

        sender_worker = WorkerAgentRuntime(host_address=host.address)
        receiver_worker = WorkerAgentRuntime(host_address=host.address)
        receiver_worker.register_factory("counter", lambda _engine: CounterAgent())

        await sender_worker.start()
        await receiver_worker.start()

        try:
            outcome = await sender_worker.send_message(
                "ping",
                recipient=AgentId.from_values("counter", "remote-1"),
            )
            assert outcome.status == DeliveryStatus.DELIVERED
            assert outcome.response_payload == {"count": 1}
        finally:
            await sender_worker.stop_when_idle()
            await receiver_worker.stop_when_idle()
            await host.stop_when_idle()

    asyncio.run(scenario())


def test_worker_runtime_routes_publish_across_workers() -> None:
    async def scenario() -> None:
        host = WorkerAgentRuntimeHost(address="127.0.0.1:0")
        await host.start()

        publisher_worker = WorkerAgentRuntime(host_address=host.address)
        listener_worker = WorkerAgentRuntime(host_address=host.address)
        received: list[dict[str, object]] = []

        class ListenerAgent(_ProtocolAgentMixin):
            @on_message
            async def handle(
                self,
                payload: dict[str, object],
                context: MessageContext,
            ) -> object:
                _ = context
                received.append(payload)
                return None

        listener_worker.register_factory("listener", lambda _engine: ListenerAgent())
        listener_worker.subscribe_exact(
            topic_type="alerts",
            agent_type="listener",
        )

        await publisher_worker.start()
        await listener_worker.start()

        try:
            ack = await publisher_worker.publish_message(
                {"event": "ready"},
                topic=TopicId.from_values(type_value="alerts", route_key="session-1"),
            )
            assert ack.enqueued_recipient_count == 1
            assert received == [{"event": "ready"}]
        finally:
            await publisher_worker.stop_when_idle()
            await listener_worker.stop_when_idle()
            await host.stop_when_idle()

    asyncio.run(scenario())


def test_distributed_runtime_context_supports_registration_after_start() -> None:
    async def scenario() -> None:
        async with distributed_runtime() as runtime:
            runtime.register_factory("counter", lambda _engine: CounterAgent())

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

    asyncio.run(scenario())


def test_worker_runtime_rejects_duplicate_agent_type_ownership() -> None:
    async def scenario() -> None:
        host = WorkerAgentRuntimeHost(address="127.0.0.1:0")
        await host.start()

        first_worker = WorkerAgentRuntime(host_address=host.address)
        second_worker = WorkerAgentRuntime(host_address=host.address)
        first_worker.register_factory("counter", lambda _engine: CounterAgent())
        second_worker.register_factory("counter", lambda _engine: CounterAgent())

        await first_worker.start()
        try:
            with pytest.raises(RuntimeError):
                await second_worker.start()
        finally:
            await first_worker.stop_when_idle()
            await second_worker.stop()
            await host.stop_when_idle()

    asyncio.run(scenario())


def test_worker_runtime_preserves_correlation_id_when_host_stops_during_rpc() -> None:
    async def scenario() -> None:
        host = WorkerAgentRuntimeHost(address="127.0.0.1:0")
        await host.start()

        sender_worker = WorkerAgentRuntime(host_address=host.address)
        receiver_worker = WorkerAgentRuntime(host_address=host.address)
        handler_started = asyncio.Event()
        handler_release = asyncio.Event()

        class BlockingAgent(_ProtocolAgentMixin):
            @on_message
            async def handle(self, payload: str, context: MessageContext) -> object:
                _ = payload
                _ = context
                handler_started.set()
                await handler_release.wait()
                return {"ok": True}

        receiver_worker.register_factory("blocking", lambda _engine: BlockingAgent())

        await sender_worker.start()
        await receiver_worker.start()

        correlation_id = CorrelationId.new()
        send_task: asyncio.Task[object] | None = None
        try:
            send_task = asyncio.create_task(
                sender_worker.send_message(
                    "ping",
                    recipient=AgentId.from_values("blocking", "remote-1"),
                    correlation_id=correlation_id,
                )
            )
            await handler_started.wait()

            await host.stop()

            outcome = await send_task
            assert outcome.status in {
                DeliveryStatus.CANCELED,
                DeliveryStatus.UNDELIVERABLE,
            }
            assert outcome.correlation_id == correlation_id
        finally:
            handler_release.set()
            if send_task is not None:
                await asyncio.gather(send_task, return_exceptions=True)
            await sender_worker.stop()
            await receiver_worker.stop()
            await host.stop()

    asyncio.run(scenario())


def test_worker_runtime_start_rolls_back_when_host_address_missing() -> None:
    async def scenario() -> None:
        worker = WorkerAgentRuntime(host_address=None)

        with pytest.raises(
            RuntimeError, match="Worker runtime requires a host address"
        ):
            await worker.start()

        assert worker.is_running is False
        assert worker.worker_id is None

    asyncio.run(scenario())


def test_worker_runtime_rejects_malformed_inbound_requests() -> None:
    async def scenario() -> None:
        worker = WorkerAgentRuntime(host_address="127.0.0.1:1")

        rpc_response = await worker.deliver_rpc(
            {
                "envelope": _wire_envelope_json(
                    message="ping",
                    kind=MessageKind.RPC_REQUEST,
                )
            }
        )
        assert rpc_response["status"] == DeliveryStatus.UNDELIVERABLE.value

        invalid_list = await worker.deliver_publish({"deliveries": "invalid"})
        assert invalid_list == {
            "ok": False,
            "error": "Publish delivery requires a list.",
        }

        invalid_entry = await worker.deliver_publish({"deliveries": [123]})
        assert invalid_entry == {
            "ok": False,
            "error": "Invalid publish delivery payload.",
        }

        missing_recipient = await worker.deliver_publish(
            {
                "deliveries": [
                    {
                        "envelope": _wire_envelope_json(
                            message={"event": "ready"},
                            kind=MessageKind.PUBLISH_EVENT,
                            topic=TopicId.from_values(
                                type_value="alerts",
                                route_key="session-1",
                            ),
                        )
                    }
                ]
            }
        )
        assert missing_recipient == {
            "ok": False,
            "error": "Host publish delivery missing recipient.",
        }

    asyncio.run(scenario())


def test_host_rpc_timeout_returns_timeout_outcome() -> None:
    async def scenario() -> None:
        host = WorkerAgentRuntimeHost(
            address="127.0.0.1:0",
            rpc_timeout_seconds=0.01,
            health_check_failure_threshold=2,
        )
        await host.start()

        sender_worker = WorkerAgentRuntime(host_address=host.address)
        receiver_worker = WorkerAgentRuntime(host_address=host.address)
        handler_release = asyncio.Event()

        class BlockingAgent(_ProtocolAgentMixin):
            @on_message
            async def handle(self, payload: str, context: MessageContext) -> object:
                _ = payload
                _ = context
                await handler_release.wait()
                return {"ok": True}

        receiver_worker.register_factory("blocking", lambda _engine: BlockingAgent())

        await sender_worker.start()
        await receiver_worker.start()

        try:
            timed_out = await sender_worker.send_message(
                "ping",
                recipient=AgentId.from_values("blocking", "remote-1"),
            )
            assert timed_out.status == DeliveryStatus.TIMEOUT
        finally:
            handler_release.set()
            await sender_worker.stop()
            await receiver_worker.stop()
            await host.stop()

    asyncio.run(scenario())


def test_host_publish_rejects_invalid_worker_enqueue_count() -> None:
    async def scenario() -> None:
        host = WorkerAgentRuntimeHost(address="127.0.0.1:0")
        await host.start()

        publisher_worker = WorkerAgentRuntime(host_address=host.address)
        listener_worker = WorkerAgentRuntime(host_address=host.address)

        class ListenerAgent(_ProtocolAgentMixin):
            @on_message
            async def handle(
                self, payload: dict[str, object], context: MessageContext
            ) -> object:
                _ = payload
                _ = context
                return None

        listener_worker.register_factory("listener", lambda _engine: ListenerAgent())
        listener_worker.subscribe_exact(
            topic_type="alerts",
            agent_type="listener",
        )

        await publisher_worker.start()
        await listener_worker.start()

        async def invalid_deliver_publish(
            request: dict[str, object],
        ) -> dict[str, object]:
            _ = request
            return {"ok": True, "enqueued_recipient_count": "bad"}

        cast(Any, listener_worker).deliver_publish = invalid_deliver_publish

        try:
            with pytest.raises(
                RuntimeError, match="Worker returned invalid publish enqueue count."
            ):
                await publisher_worker.publish_message(
                    {"event": "ready"},
                    topic=TopicId.from_values(
                        type_value="alerts",
                        route_key="session-1",
                    ),
                )
        finally:
            await publisher_worker.stop()
            await listener_worker.stop()
            await host.stop()

    asyncio.run(scenario())


def test_worker_runtime_catalog_sync_replaces_stale_subscription_state() -> None:
    async def scenario() -> None:
        host = WorkerAgentRuntimeHost(address="127.0.0.1:0")
        await host.start()

        publisher_worker = WorkerAgentRuntime(host_address=host.address)
        listener_worker = WorkerAgentRuntime(host_address=host.address)
        listener_worker.register_factory("listener", lambda _engine: CounterAgent())
        subscription_id = listener_worker.subscribe_exact(
            topic_type="alerts",
            agent_type="listener",
            delivery_mode=DeliveryMode.STATELESS,
        )

        await publisher_worker.start()
        await listener_worker.start()

        try:
            first_ack = await publisher_worker.publish_message(
                "first",
                topic=TopicId.from_values(type_value="alerts", route_key="session-1"),
            )
            assert first_ack.enqueued_recipient_count == 1

            listener_worker.remove_subscription(subscription_id)
            await cast(Any, listener_worker)._await_catalog_sync()

            second_ack = await publisher_worker.publish_message(
                "second",
                topic=TopicId.from_values(type_value="alerts", route_key="session-1"),
            )
            assert second_ack.enqueued_recipient_count == 0
        finally:
            await publisher_worker.stop_when_idle()
            await listener_worker.stop_when_idle()
            await host.stop_when_idle()

    asyncio.run(scenario())
