"""Engine capability contract shared across runtime and agent primitives."""

import abc

from agentlane.messaging import (
    AgentId,
    AgentType,
    CorrelationId,
    DeliveryOutcome,
    IdempotencyKey,
    PublishAck,
    TopicId,
)

from ._cancellation import CancellationToken


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
        """Send one direct message and await delivery outcome.

        Args:
            message: Application payload to send.
            recipient: Target agent id or type identifier.
            sender: Optional sender id propagated in context.
            correlation_id: Optional causal chain id.
            cancellation_token: Optional shared cancellation token.
            idempotency_key: Optional deduplication key.

        Returns:
            DeliveryOutcome: Terminal delivery result for this call.
        """
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
        """Publish one event message and return enqueue acknowledgment.

        Args:
            message: Application payload to publish.
            topic: Topic used for subscription routing.
            sender: Optional sender id propagated in context.
            correlation_id: Optional causal chain id.
            cancellation_token: Optional shared cancellation token.
            idempotency_key: Optional deduplication key.

        Returns:
            PublishAck: Enqueue acknowledgment for routed publish deliveries.
        """
        raise NotImplementedError
