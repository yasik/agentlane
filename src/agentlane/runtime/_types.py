"""Runtime-internal data models."""

from asyncio import Future
from dataclasses import dataclass, field

from agentlane.messaging import AgentId, CancellationToken, MessageEnvelope


@dataclass(slots=True)
class DeliveryTask:
    """Scheduler work item for one envelope-recipient delivery attempt."""

    envelope: MessageEnvelope
    """Envelope to be delivered."""

    recipient: AgentId
    """Resolved recipient mailbox owner."""

    attempt: int = 1
    """One-based delivery attempt counter."""

    cancellation_token: CancellationToken = field(default_factory=CancellationToken)
    """Cooperative cancellation token shared with MessageContext."""

    response_future: Future[object] | None = None
    """Optional future completed with the terminal delivery outcome."""
