"""Routing policy interfaces and default implementation."""

from collections.abc import Sequence
from typing import Protocol

from ._envelope import MessageEnvelope
from ._identity import AgentId
from ._subscription import Subscription


class RoutingPolicy(Protocol):
    """Routing policy contract."""

    def resolve_rpc_recipient(self, envelope: MessageEnvelope) -> AgentId:
        """Resolve exactly one RPC recipient."""
        ...

    def resolve_publish_recipients(
        self,
        envelope: MessageEnvelope,
        subscriptions: Sequence[Subscription],
    ) -> list[AgentId]:
        """Resolve zero or more publish recipients."""
        ...


class SourceKeyAffinityRoutingPolicy:
    """Compatibility policy preserving topic.source to agent key mapping."""

    def resolve_rpc_recipient(self, envelope: MessageEnvelope) -> AgentId:
        """Resolve RPC recipient from envelope."""
        if envelope.recipient is None:
            raise LookupError("RPC recipient is missing.")
        return envelope.recipient

    def resolve_publish_recipients(
        self,
        envelope: MessageEnvelope,
        subscriptions: Sequence[Subscription],
    ) -> list[AgentId]:
        """Resolve publish recipients deterministically."""
        if envelope.topic is None:
            raise LookupError("Publish topic is missing.")

        recipients: dict[tuple[str, str], AgentId] = {}
        for subscription in subscriptions:
            # Match subscription against topic, skipping non-matching subscriptions.
            # The first matching subscription for a given topic.source will win,
            # ensuring deterministic routing based on source-key affinity.
            if not subscription.is_match(envelope.topic):
                continue

            # Map topic to recipient agent, skipping duplicate recipients for
            # multiple matching subscriptions.
            recipient = subscription.map_to_agent(envelope.topic)
            recipients[(recipient.type.value, recipient.key.value)] = recipient

        # Sort recipients by type and key for deterministic delivery
        # order across multiple subscriptions.
        sorted_recipients = sorted(
            recipients.values(),
            key=lambda recipient: (recipient.type.value, recipient.key.value),
        )
        return sorted_recipients
