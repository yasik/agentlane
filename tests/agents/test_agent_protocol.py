import asyncio
from dataclasses import dataclass
from typing import cast

from agentlane.messaging import (
    AgentId,
    AgentType,
    CancellationToken,
    CorrelationId,
    DeliveryOutcome,
    IdempotencyKey,
    MessageContext,
    MessageId,
    PublishAck,
    TopicId,
)
from agentlane.runtime import (
    Agent,
    BaseAgent,
    Engine,
    is_on_message_handler,
    on_message,
)


class _EchoAgent:
    @on_message
    async def process(self, payload: object, context: MessageContext) -> object:
        _ = context
        return payload


def test_agent_protocol_shape() -> None:
    _ = cast(Agent, _EchoAgent())
    assert is_on_message_handler(_EchoAgent.process)


@dataclass(slots=True)
class _SendCapture:
    message: object
    recipient: AgentId | AgentType | str
    sender: AgentId | None
    correlation_id: CorrelationId | None
    cancellation_token: CancellationToken | None


@dataclass(slots=True)
class _PublishCapture:
    message: object
    topic: TopicId
    sender: AgentId | None
    correlation_id: CorrelationId | None
    cancellation_token: CancellationToken | None


class _StubEngine(Engine):
    def __init__(self) -> None:
        self.last_send: _SendCapture | None = None
        self.last_publish: _PublishCapture | None = None

    async def send_message(
        self,
        message: object,
        recipient: AgentId | AgentType | str,
        *,
        sender: AgentId | None = None,
        correlation_id: CorrelationId | None = None,
        cancellation_token: CancellationToken | None = None,
        idempotency_key: IdempotencyKey | None = None,
    ) -> DeliveryOutcome:
        _ = idempotency_key
        self.last_send = _SendCapture(
            message=message,
            recipient=recipient,
            sender=sender,
            correlation_id=correlation_id,
            cancellation_token=cancellation_token,
        )
        return DeliveryOutcome.delivered(
            message_id=MessageId.new(),
            correlation_id=correlation_id,
            response_payload="ok",
        )

    async def publish_message(
        self,
        message: object,
        topic: TopicId,
        *,
        sender: AgentId | None = None,
        correlation_id: CorrelationId | None = None,
        cancellation_token: CancellationToken | None = None,
        idempotency_key: IdempotencyKey | None = None,
    ) -> PublishAck:
        _ = idempotency_key
        self.last_publish = _PublishCapture(
            message=message,
            topic=topic,
            sender=sender,
            correlation_id=correlation_id,
            cancellation_token=cancellation_token,
        )
        return PublishAck(
            message_id=MessageId.new(),
            correlation_id=correlation_id,
            enqueued_recipient_count=1,
            enqueued_at_ms=0,
        )


class _BaseAgentUnderTest(BaseAgent):
    @on_message
    async def process(self, payload: object, context: MessageContext) -> object:
        _ = payload
        _ = context
        return None


def test_base_agent_forwards_explicit_messaging_metadata() -> None:
    async def scenario() -> None:
        engine = _StubEngine()
        agent = _BaseAgentUnderTest(engine)
        sender = AgentId.from_values("runner", "r1")
        correlation_id = CorrelationId.new()
        cancellation_token = CancellationToken()

        await agent.send_message(
            "work",
            recipient="worker",
            sender=sender,
            correlation_id=correlation_id,
            cancellation_token=cancellation_token,
        )
        await agent.publish_message(
            "event",
            topic=TopicId.from_values(type_value="updates", route_key="rk"),
            sender=sender,
            correlation_id=correlation_id,
            cancellation_token=cancellation_token,
        )

        if engine.last_send is None or engine.last_publish is None:
            raise AssertionError("Expected engine captures for send and publish.")

        assert engine.last_send.sender == sender
        assert engine.last_send.correlation_id == correlation_id
        assert engine.last_send.cancellation_token is cancellation_token
        assert engine.last_publish.sender == sender
        assert engine.last_publish.correlation_id == correlation_id
        assert engine.last_publish.cancellation_token is cancellation_token

    asyncio.run(scenario())
