"""Agent protocol and message-handler markers used by runtime dispatch."""

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from agentlane.messaging import AgentId, MessageContext

_ON_MESSAGE_ATTR = "__agentlane_on_message__"

type MessageHandler = Callable[[object, MessageContext], Awaitable[object]]
"""Resolved runtime handler signature after binding `self`."""


def on_message[HandlerT: Callable[..., Awaitable[object]]](
    handler: HandlerT,
) -> HandlerT:
    """Mark a method as the runtime message handler entrypoint.

    Args:
        handler: Async method to expose as a dispatch handler.

    Returns:
        HandlerT: Same callable with runtime marker metadata attached.
    """
    setattr(handler, _ON_MESSAGE_ATTR, True)
    return handler


def is_on_message_handler(candidate: object) -> bool:
    """Return whether a callable is marked with `@on_message`.

    Args:
        candidate: Object to inspect for marker metadata.

    Returns:
        bool: True when candidate is callable and marked as `@on_message`.
    """
    if not callable(candidate):
        return False
    if bool(getattr(candidate, _ON_MESSAGE_ATTR, False)):
        return True
    function = getattr(candidate, "__func__", None)
    return bool(getattr(function, _ON_MESSAGE_ATTR, False))


@runtime_checkable
class Agent(Protocol):
    """Protocol for runtime-managed agent instances."""

    @property
    def id(self) -> AgentId:
        """Runtime-assigned agent identity for this instance.

        Returns:
            AgentId: Bound runtime id for this agent instance.
        """
        ...

    def bind_agent_id(self, agent_id: AgentId) -> None:
        """Bind runtime-assigned agent identity onto this instance.

        Args:
            agent_id: Runtime-assigned id for this agent instance.

        Returns:
            None: Always returns after binding.
        """
        ...
