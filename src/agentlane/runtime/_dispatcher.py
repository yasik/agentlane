"""Task dispatcher that invokes agent handlers."""

import inspect
from collections.abc import Awaitable, Callable
from typing import cast

from agentlane.agents import Agent, is_on_message_handler
from agentlane.messaging import (
    DeliveryOutcome,
    DeliveryStatus,
    MessageContext,
    MessageKind,
)

from ._registry import AgentRegistry
from ._types import DeliveryTask


class Dispatcher:
    """Resolves an agent instance, invokes its handler, and returns DeliveryOutcome."""

    def __init__(self, *, registry: AgentRegistry) -> None:
        """Create dispatcher with a registry dependency."""
        self._registry = registry

    async def dispatch(self, task: DeliveryTask) -> DeliveryOutcome:
        """Dispatch one delivery task and return a structured outcome."""
        try:
            agent = await self._registry.get_or_create(task.recipient)
        except LookupError as exc:
            return DeliveryOutcome.failed(
                status=DeliveryStatus.UNDELIVERABLE,
                message_id=task.envelope.message_id,
                correlation_id=task.envelope.correlation_id,
                message=str(exc),
                retryable=False,
            )

        context = MessageContext(
            sender=task.envelope.sender,
            topic=task.envelope.topic,
            is_rpc=task.envelope.kind == MessageKind.RPC_REQUEST,
            message_id=task.envelope.message_id,
            correlation_id=task.envelope.correlation_id,
            cancellation_token=task.cancellation_token,
            attempt=task.attempt,
        )

        try:
            response_payload = await self._invoke_agent(
                agent, task.envelope.payload.data, context
            )
            return DeliveryOutcome.delivered(
                message_id=task.envelope.message_id,
                correlation_id=task.envelope.correlation_id,
                response_payload=response_payload,
            )
        except Exception as exc:  # noqa: BLE001
            return DeliveryOutcome.failed(
                status=DeliveryStatus.HANDLER_ERROR,
                message_id=task.envelope.message_id,
                correlation_id=task.envelope.correlation_id,
                message=str(exc),
                retryable=False,
            )

    async def _invoke_agent(
        self,
        agent: Agent,
        payload: object,
        context: MessageContext,
    ) -> object:
        """Invoke an `@on_message` handler selected by payload type."""
        handler = self._resolve_on_message_handler(agent=agent, payload=payload)
        return await handler(payload, context)

    @staticmethod
    def _resolve_on_message_handler(
        agent: Agent,
        payload: object,
    ) -> Callable[[object, MessageContext], Awaitable[object]]:
        """Resolve an `@on_message` handler using payload-type routing."""
        discovered_handlers = _discover_on_message_handlers(agent=agent)
        validated_handlers = _validate_on_message_handlers(
            agent=agent,
            discovered_handlers=discovered_handlers,
        )
        matching_handlers = _find_matching_handlers(
            payload=payload,
            validated_handlers=validated_handlers,
        )
        return _select_single_matching_handler(
            agent=agent,
            payload=payload,
            matching_handlers=matching_handlers,
        )


def _discover_on_message_handlers(
    agent: Agent,
) -> list[Callable[..., Awaitable[object]]]:
    """Discover methods marked with `@on_message` on a bound agent instance."""
    # Resolve from the instance (not class) so descriptor binding has already happened.
    discovered_handlers = [
        cast(Callable[..., Awaitable[object]], member)
        for _, member in inspect.getmembers(agent, predicate=callable)
        if is_on_message_handler(member)
    ]
    if not discovered_handlers:
        raise TypeError(
            f"Agent '{type(agent).__name__}' does not define an `@on_message` handler."
        )
    return discovered_handlers


def _validate_on_message_handlers(
    agent: Agent,
    discovered_handlers: list[Callable[..., Awaitable[object]]],
) -> list[tuple[Callable[[object, MessageContext], Awaitable[object]], type[object]]]:
    """Validate discovered handlers and return typed routing descriptors."""
    # Fail fast on malformed signatures before attempting payload matching.
    return [
        _validate_on_message_handler(agent=agent, handler=handler)
        for handler in discovered_handlers
    ]


def _find_matching_handlers(
    payload: object,
    validated_handlers: list[
        tuple[Callable[[object, MessageContext], Awaitable[object]], type[object]]
    ],
) -> list[Callable[[object, MessageContext], Awaitable[object]]]:
    """Return handlers whose payload type exactly matches the runtime payload type."""
    # Exact runtime type match only. This keeps resolution efficient and
    # deterministic, and encourages explicit annotations.
    matching_handlers: list[Callable[[object, MessageContext], Awaitable[object]]] = []
    for handler, payload_type in validated_handlers:
        is_match = _payload_type_matches_exact(
            payload=payload,
            payload_type=payload_type,
        )
        if not is_match:
            continue
        matching_handlers.append(handler)
    return matching_handlers


def _select_single_matching_handler(
    agent: Agent,
    payload: object,
    matching_handlers: list[Callable[[object, MessageContext], Awaitable[object]]],
) -> Callable[[object, MessageContext], Awaitable[object]]:
    """Select one matching handler or raise explicit no-match/ambiguous errors."""
    if not matching_handlers:
        raise TypeError(
            f"Agent '{type(agent).__name__}' has no `@on_message` handler for "
            f"payload type '{type(payload).__name__}'."
        )

    if len(matching_handlers) > 1:
        # Multiple exact matches would make dispatch order-dependent.
        # Reject explicitly instead of picking arbitrarily.
        raise TypeError(
            f"Agent '{type(agent).__name__}' has ambiguous `@on_message` handlers for "
            f"payload type '{type(payload).__name__}'."
        )

    return matching_handlers[0]

def _validate_on_message_handler(
    agent: Agent,
    handler: Callable[..., Awaitable[object]],
) -> tuple[Callable[[object, MessageContext], Awaitable[object]], type[object]]:
    """Validate `@on_message` signature and return callable with payload annotation."""
    parameters = inspect.signature(handler).parameters
    # Contract: every handler receives both payload and runtime context.
    # Context is required so cancellation and metadata are always available.
    if len(parameters) != 2 or "context" not in parameters:
        raise TypeError(
            f"`@on_message` handler on agent '{type(agent).__name__}' must have signature "
            "`(payload, context)`."
        )

    # By convention the first declared parameter is treated as payload.
    # Its type annotation drives routing.
    payload_parameter = next(iter(parameters.values()))
    if payload_parameter.annotation is inspect.Signature.empty:
        raise TypeError(
            f"`@on_message` handler on agent '{type(agent).__name__}' must declare "
            "an explicit payload type annotation."
        )
    # Restrict to concrete runtime types to keep matching deterministic and cheap.
    if not isinstance(payload_parameter.annotation, type):
        raise TypeError(
            f"`@on_message` handler on agent '{type(agent).__name__}' must annotate "
            "payload with a concrete type."
        )
    return cast(Callable[[object, MessageContext], Awaitable[object]], handler), cast(
        type[object], payload_parameter.annotation
    )

def _payload_type_matches_exact(payload: object, payload_type: object) -> bool:
    """Return whether handler payload annotation exactly matches runtime payload type."""
    return isinstance(payload_type, type) and type(payload) is payload_type
