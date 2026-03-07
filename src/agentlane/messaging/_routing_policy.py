"""Routing policy interfaces and default implementation."""

from collections.abc import Sequence
from typing import Protocol

from ._envelope import MessageEnvelope
from ._identity import AgentId
from ._subscription import (
    DeliveryMode,
    PublishRoute,
    Subscription,
)


class RoutingPolicy(Protocol):
    """Routing policy contract."""

    def resolve_rpc_recipient(self, envelope: MessageEnvelope) -> AgentId:
        """Resolve exactly one RPC recipient.

        Args:
            envelope: RPC envelope requiring one recipient.

        Returns:
            AgentId: Resolved RPC recipient id.

        Raises:
            LookupError: If recipient cannot be resolved.
        """
        ...

    def resolve_publish_routes(
        self,
        envelope: MessageEnvelope,
        subscriptions: Sequence[Subscription],
    ) -> list[PublishRoute]:
        """Resolve zero or more publish routes.

        Args:
            envelope: Publish envelope requiring fan-out routing.
            subscriptions: Active subscriptions to match against.

        Returns:
            list[PublishRoute]: Resolved and deduplicated publish routes.

        Raises:
            LookupError: If publish routing cannot proceed due to missing topic data.
        """
        ...


class SourceKeyAffinityRoutingPolicy:
    """Compatibility policy preserving topic.source to agent key mapping."""

    def resolve_rpc_recipient(self, envelope: MessageEnvelope) -> AgentId:
        """Resolve RPC recipient from envelope.

        Args:
            envelope: RPC envelope requiring one recipient.

        Returns:
            AgentId: Recipient from envelope.

        Raises:
            LookupError: If envelope recipient is missing.
        """
        if envelope.recipient is None:
            raise LookupError("RPC recipient is missing.")
        return envelope.recipient

    def resolve_publish_routes(
        self,
        envelope: MessageEnvelope,
        subscriptions: Sequence[Subscription],
    ) -> list[PublishRoute]:
        """Resolve publish routes deterministically.

        Args:
            envelope: Publish envelope used for topic matching.
            subscriptions: Active subscriptions to evaluate.

        Returns:
            list[PublishRoute]: Stable ordered publish routes.

        Raises:
            LookupError: If envelope topic is missing.
        """
        if envelope.topic is None:
            raise LookupError("Publish topic is missing.")

        stateful_routes: dict[tuple[str, str], PublishRoute] = {}
        stateless_routes: dict[tuple[str, str], PublishRoute] = {}
        for subscription in subscriptions:
            # Match subscription against topic, skipping non-matching subscriptions.
            if not subscription.is_match(envelope.topic):
                continue

            recipient = subscription.map_to_agent(envelope.topic)
            route = PublishRoute(
                subscription_id=subscription.id,
                recipient=recipient,
                delivery_mode=subscription.delivery_mode,
            )

            if route.delivery_mode == DeliveryMode.STATEFUL:
                # Stateful dedup is by concrete recipient id.
                stateful_routes[(recipient.type.value, recipient.key.value)] = route
                continue

            # Stateless dedup is by subscription and recipient type so each
            # matching subscription contributes at most one delivery route.
            stateless_routes[(subscription.id, recipient.type.value)] = route

        # Stable ordering keeps fan-out deterministic across runs.
        sorted_stateful = sorted(
            stateful_routes.values(),
            key=lambda route: (
                route.recipient.type.value,
                route.recipient.key.value,
            ),
        )
        sorted_stateless = sorted(
            stateless_routes.values(),
            key=lambda route: (
                route.recipient.type.value,
                route.subscription_id,
            ),
        )
        return [*sorted_stateful, *sorted_stateless]
