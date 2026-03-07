"""Agent decorators and protocol for runtime-managed handlers."""

from collections.abc import Awaitable, Callable
from typing import Protocol

from agentlane.messaging import (
    AgentId,
    AgentType,
    CancellationToken,
    CorrelationId,
    DeliveryOutcome,
    IdempotencyKey,
    MessageContext,
    PublishAck,
    TopicId,
)
from agentlane.runtime import Engine

_ON_MESSAGE_ATTR = "__agentlane_on_message__"

type MessageHandler = Callable[[object, MessageContext], Awaitable[object]]
"""Resolved runtime handler signature after binding `self`."""


class BaseAgent:
    """Base agent primitive with scoped runtime messaging helpers.

    `BaseAgent` intentionally exposes only send/publish operations and does not
    provide access to runtime control-plane APIs (start/stop/registration).
    """

    def __init__(self, engine: Engine) -> None:
        """Initialize base agent with restricted engine capability."""
        self._engine = engine

    @property
    def engine(self) -> Engine:
        """Return engine capability available to this agent."""
        return self._engine

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


def on_message[HandlerT: Callable[..., Awaitable[object]]](
    handler: HandlerT,
) -> HandlerT:
    """Mark a method as the runtime message handler entrypoint."""
    setattr(handler, _ON_MESSAGE_ATTR, True)
    return handler


def is_on_message_handler(candidate: object) -> bool:
    """Return whether a callable is marked with `@on_message`."""
    if not callable(candidate):
        return False
    if bool(getattr(candidate, _ON_MESSAGE_ATTR, False)):
        return True
    function = getattr(candidate, "__func__", None)
    return bool(getattr(function, _ON_MESSAGE_ATTR, False))


class Agent(Protocol):
    """Marker protocol for runtime-managed agent instances."""
