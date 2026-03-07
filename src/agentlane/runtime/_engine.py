"""Engine capability contract shared across runtime and agent primitives."""

import abc

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


class Engine(abc.ABC):
    """Restricted engine interface exposed to agent instances.

    This interface intentionally contains only messaging operations.
    Runtime control-plane operations are kept outside of this contract.
    """

    @abc.abstractmethod
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
        """Send one direct message and await delivery outcome."""
        raise NotImplementedError

    @abc.abstractmethod
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
        """Publish one event message and return enqueue acknowledgment."""
        raise NotImplementedError
