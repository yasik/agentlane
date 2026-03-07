"""Base agent primitive with scoped runtime messaging helpers."""

from agentlane.messaging import (
    AgentId,
    AgentType,
    CancellationToken,
    CorrelationId,
    DeliveryOutcome,
    IdempotencyKey,
    PublishAck,
    TopicId,
)

from ._engine import Engine


class BaseAgent:
    """Base agent primitive with scoped runtime messaging helpers."""

    def __init__(self, engine: Engine) -> None:
        """Initialize base agent with restricted engine capability."""
        self._engine = engine

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
        """Send a message through this agent's engine capability."""
        return await self._engine.send_message(
            message,
            recipient,
            sender=sender,
            correlation_id=correlation_id,
            cancellation_token=cancellation_token,
            idempotency_key=idempotency_key,
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
        """Publish a message through this agent's engine capability."""
        return await self._engine.publish_message(
            message,
            topic,
            sender=sender,
            correlation_id=correlation_id,
            cancellation_token=cancellation_token,
            idempotency_key=idempotency_key,
        )
