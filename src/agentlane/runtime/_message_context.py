"""Message context passed to handlers."""

from dataclasses import dataclass

from agentlane.messaging import AgentId, CorrelationId, MessageId, TopicId

from ._cancellation import CancellationToken


@dataclass(slots=True)
class MessageContext:
    """Normalized context for a message handler invocation."""

    recipient: AgentId
    """Current handler recipient identity (the executing agent id)."""

    sender: AgentId | None
    """Optional sender identity."""

    topic: TopicId | None
    """Optional publish topic."""

    is_rpc: bool
    """True for RPC-style delivery, false for publish-style delivery."""

    message_id: MessageId
    """Current envelope id."""

    correlation_id: CorrelationId | None
    """Correlation chain id."""

    attempt: int
    """Delivery attempt number (1-based)."""

    cancellation_token: CancellationToken
    """Cooperative cancellation token for this delivery; forward to outbound sends."""
