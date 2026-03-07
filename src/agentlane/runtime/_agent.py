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

    def __init__(self, engine: Engine, bind_id: AgentId | None = None) -> None:
        """Initialize base agent with restricted engine capability.

        Args:
            engine: Runtime engine messaging capability exposed to this agent.
            bind_id: Optional pre-bound agent id (primarily for tests).
        """
        self._engine = engine
        self._id = bind_id or None

    @property
    def id(self) -> AgentId:
        """Return this runtime-bound agent instance id.

        Returns:
            AgentId: Runtime-assigned id for this agent instance.

        Raises:
            RuntimeError: If id has not been bound by runtime registration.
        """
        if self._id is None:
            raise RuntimeError(
                "Agent id is not bound. Register this agent with RuntimeEngine "
                "before using messaging helpers."
            )
        return self._id

    def bind_agent_id(self, agent_id: AgentId) -> None:
        """Bind runtime-assigned id onto this agent instance.

        Args:
            agent_id: Runtime-assigned id for this concrete agent instance.

        Returns:
            None: Always returns after binding.
        """
        self._id = agent_id

    async def send_message(
        self,
        message: object,
        recipient: AgentId | AgentType | str,
        *,
        correlation_id: CorrelationId | None = None,
        cancellation_token: CancellationToken | None = None,
        idempotency_key: IdempotencyKey | None = None,
    ) -> DeliveryOutcome:
        """Send a message through this agent's engine capability.

        Args:
            message: Application payload to send.
            recipient: Target agent id or type identifier.
            correlation_id: Optional causal chain id.
            cancellation_token: Optional shared cancellation token.
            idempotency_key: Optional deduplication key.

        Returns:
            DeliveryOutcome: Terminal delivery outcome for the RPC-style call.
        """
        return await self._engine.send_message(
            message,
            recipient,
            sender=self._id,
            correlation_id=correlation_id,
            cancellation_token=cancellation_token,
            idempotency_key=idempotency_key,
        )

    async def publish_message(
        self,
        message: object,
        topic: TopicId,
        *,
        correlation_id: CorrelationId | None = None,
        cancellation_token: CancellationToken | None = None,
        idempotency_key: IdempotencyKey | None = None,
    ) -> PublishAck:
        """Publish a message through this agent's engine capability.

        Args:
            message: Application payload to publish.
            topic: Topic id used for subscription routing.
            correlation_id: Optional causal chain id.
            cancellation_token: Optional shared cancellation token.
            idempotency_key: Optional deduplication key.

        Returns:
            PublishAck: Enqueue acknowledgment for publish fan-out.
        """
        return await self._engine.publish_message(
            message,
            topic,
            sender=self._id,
            correlation_id=correlation_id,
            cancellation_token=cancellation_token,
            idempotency_key=idempotency_key,
        )
