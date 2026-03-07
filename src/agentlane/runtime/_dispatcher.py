"""Task dispatcher that invokes agent handlers."""

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast

from agentlane.agents import Agent, is_on_message_handler
from agentlane.messaging import (
    DeliveryOutcome,
    DeliveryStatus,
    MessageContext,
    MessageKind,
)

from ._registry import AgentRegistry
from ._shared import Engine
from ._types import DeliveryTask


class Dispatcher:
    """Resolves an agent instance, invokes its handler, and returns DeliveryOutcome."""

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        engine: Engine,
    ) -> None:
        """Create dispatcher with registry and engine dependencies."""
        self._registry = registry
        self._engine = engine

    async def dispatch(self, task: DeliveryTask) -> DeliveryOutcome:
        """Dispatch one delivery task and return a structured outcome."""
        try:
            agent = await self._registry.get_or_create(
                task.recipient,
                engine=self._engine,
            )
        except LookupError as exc:
            return DeliveryOutcome.failed(
                status=DeliveryStatus.UNDELIVERABLE,
                message_id=task.envelope.message_id,
                correlation_id=task.envelope.correlation_id,
                message=str(exc),
                retryable=False,
            )

        context = MessageContext(
            recipient=task.recipient,
            sender=task.envelope.sender,
            topic=task.envelope.topic,
            is_rpc=task.envelope.kind == MessageKind.RPC_REQUEST,
            message_id=task.envelope.message_id,
            correlation_id=task.envelope.correlation_id,
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


@dataclass(frozen=True, slots=True)
class _HandlerDescriptor:
    """Validated and bound `@on_message` handler metadata."""

    method_name: str
    """Handler method name used for diagnostics."""

    message_type: type[object]
    """Concrete payload type accepted by this handler."""

    handler: Callable[[object, MessageContext], Awaitable[object]]
    """Bound async handler callable."""


def _discover_on_message_handlers(
    agent: Agent,
) -> list[tuple[str, Callable[..., Awaitable[object]]]]:
    """Discover methods marked with `@on_message` on a bound agent instance."""
    # Resolve from the instance (not class) so descriptor binding has already happened.
    discovered_handlers = [
        (name, cast(Callable[..., Awaitable[object]], member))
        for name, member in inspect.getmembers(agent, predicate=callable)
        if is_on_message_handler(member)
    ]
    if not discovered_handlers:
        raise TypeError(
            f"Agent '{type(agent).__name__}' does not define an `@on_message` handler."
        )
    return discovered_handlers


def _validate_on_message_handlers(
    agent: Agent,
    discovered_handlers: list[tuple[str, Callable[..., Awaitable[object]]]],
) -> list[_HandlerDescriptor]:
    """Validate discovered handlers and return typed routing descriptors."""
    # Fail fast on malformed signatures before attempting payload matching.
    return [
        _validate_on_message_handler(
            agent=agent,
            method_name=method_name,
            handler=handler,
        )
        for method_name, handler in discovered_handlers
    ]


def _find_matching_handlers(
    payload: object,
    validated_handlers: list[_HandlerDescriptor],
) -> list[_HandlerDescriptor]:
    """Return handlers whose payload type exactly matches the runtime payload type."""
    # Exact runtime type match only. This keeps resolution efficient and
    # deterministic, and encourages explicit annotations.
    matching_handlers: list[_HandlerDescriptor] = []
    for descriptor in validated_handlers:
        is_match = _payload_type_matches_exact(
            payload=payload,
            payload_type=descriptor.message_type,
        )
        if not is_match:
            continue
        matching_handlers.append(descriptor)
    return matching_handlers


def _select_single_matching_handler(
    agent: Agent,
    payload: object,
    matching_handlers: list[_HandlerDescriptor],
) -> Callable[[object, MessageContext], Awaitable[object]]:
    """Select one matching handler or raise explicit no-match/ambiguous errors."""
    if not matching_handlers:
        raise TypeError(
            f"Agent '{type(agent).__name__}' has no `@on_message` handler for "
            f"payload type '{type(payload).__name__}'."
        )

    if len(matching_handlers) > 1:
        handler_names = ", ".join(
            sorted(descriptor.method_name for descriptor in matching_handlers)
        )
        # Multiple exact matches would make dispatch order-dependent.
        # Reject explicitly instead of picking arbitrarily.
        raise TypeError(
            f"Agent '{type(agent).__name__}' has ambiguous `@on_message` handlers for "
            f"payload type '{type(payload).__name__}': {handler_names}."
        )

    return matching_handlers[0].handler


def _validate_on_message_handler(
    agent: Agent,
    method_name: str,
    handler: Callable[..., Awaitable[object]],
) -> _HandlerDescriptor:
    """Validate `@on_message` signature and return callable with payload annotation."""
    if not inspect.iscoroutinefunction(handler):
        raise TypeError(
            f"`@on_message` handler '{method_name}' on agent "
            f"'{type(agent).__name__}' must be declared as `async def`."
        )

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
    return _HandlerDescriptor(
        method_name=method_name,
        message_type=cast(type[object], payload_parameter.annotation),
        handler=cast(Callable[[object, MessageContext], Awaitable[object]], handler),
    )


def _payload_type_matches_exact(payload: object, payload_type: object) -> bool:
    """Return whether handler payload annotation exactly matches runtime payload type."""
    return isinstance(payload_type, type) and type(payload) is payload_type
