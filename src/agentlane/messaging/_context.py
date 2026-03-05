"""Message context passed to handlers."""

from dataclasses import dataclass

from ._cancellation import CancellationToken
from ._identity import AgentId, CorrelationId, MessageId, TopicId


@dataclass(slots=True)
class MessageContext:
    """Normalized context for a message handler invocation."""

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

    cancellation_token: CancellationToken
    """Cooperative cancellation token triggered on runtime shutdown."""

    attempt: int
    """Delivery attempt number (1-based)."""
