"""Agent decorators and protocol for runtime-managed handlers."""

from collections.abc import Awaitable, Callable
from typing import Protocol

from agentlane.messaging import MessageContext

_ON_MESSAGE_ATTR = "__agentlane_on_message__"

type MessageHandler = Callable[[object, MessageContext], Awaitable[object]]
"""Resolved runtime handler signature after binding `self`."""


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
