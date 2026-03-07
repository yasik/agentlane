"""Routing engine managing subscriptions and recipient resolution."""

from collections.abc import Sequence
from threading import RLock

from ._envelope import MessageEnvelope
from ._identity import AgentId
from ._routing_policy import (
    RoutingPolicy,
    SourceKeyAffinityRoutingPolicy,
)
from ._subscription import PublishRoute, Subscription


class RoutingEngine:
    """Resolve recipients for RPC and publish envelopes."""

    def __init__(self, *, policy: RoutingPolicy | None = None) -> None:
        """Create a routing engine with an optional custom policy."""
        self._policy = policy or SourceKeyAffinityRoutingPolicy()
        self._subscriptions_lock = RLock()
        self._subscriptions: dict[str, Subscription] = {}

    @property
    def subscriptions(self) -> Sequence[Subscription]:
        """Return current subscriptions.

        Returns:
            Sequence[Subscription]: Immutable snapshot of active subscriptions.
        """
        with self._subscriptions_lock:
            return tuple(self._subscriptions.values())

    def add_subscription(self, subscription: Subscription) -> None:
        """Add or replace a subscription.

        Args:
            subscription: Subscription to register by its stable id.

        Returns:
            None: Always returns after updating routing state.
        """
        with self._subscriptions_lock:
            self._subscriptions[subscription.id] = subscription

    def remove_subscription(self, subscription_id: str) -> None:
        """Remove a subscription by id.

        Args:
            subscription_id: Stable id of the subscription to remove.

        Returns:
            None: Always returns after removing the subscription.

        Raises:
            LookupError: If the id does not exist.
        """
        with self._subscriptions_lock:
            if subscription_id not in self._subscriptions:
                raise LookupError(f"Subscription '{subscription_id}' does not exist.")
            del self._subscriptions[subscription_id]

    def resolve_rpc_recipient(self, envelope: MessageEnvelope) -> AgentId:
        """Resolve a single RPC recipient.

        Args:
            envelope: RPC envelope to resolve.

        Returns:
            AgentId: Resolved recipient id.
        """
        return self._policy.resolve_rpc_recipient(envelope)

    def resolve_publish_routes(self, envelope: MessageEnvelope) -> list[PublishRoute]:
        """Resolve publish routes including delivery-mode metadata.

        Args:
            envelope: Publish envelope to fan out.

        Returns:
            list[PublishRoute]: Routed recipients with delivery mode metadata.
        """
        return self._policy.resolve_publish_routes(envelope, self.subscriptions)

    def resolve_publish_recipients(self, envelope: MessageEnvelope) -> list[AgentId]:
        """Resolve publish recipients only (legacy helper).

        Args:
            envelope: Publish envelope to fan out.

        Returns:
            list[AgentId]: Recipient ids extracted from resolved publish routes.
        """
        return [route.recipient for route in self.resolve_publish_routes(envelope)]
