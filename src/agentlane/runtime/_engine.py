"""Runtime engine implementing v1 messaging/runtime contracts."""

import asyncio
from time import time
from typing import cast

from agentlane.agents import Agent
from agentlane.messaging import (
    AgentId,
    AgentKey,
    AgentType,
    CorrelationId,
    DeliveryOutcome,
    DeliveryStatus,
    MessageEnvelope,
    MessageId,
    Payload,
    PayloadFormat,
    PublishAck,
    RoutingEngine,
    Subscription,
    TopicId,
)

from ._dispatcher import Dispatcher
from ._distributed_env import DistributedEnvironment
from ._environment import RuntimeEnvironment
from ._registry import AgentFactory, AgentRegistry
from ._scheduler import (
    PerAgentMailboxScheduler,
    SchedulerRejectedError,
)
from ._single_threaded_env import SingleThreadedEnvironment
from ._types import DeliveryTask, RuntimeMode


def utc_now_ms() -> int:
    """Return current UTC epoch milliseconds."""
    return int(time() * 1000)


class RuntimeEngine:
    """High-level API that wires routing, registry, scheduling, and execution environment."""

    def __init__(
        self,
        *,
        mode: RuntimeMode = RuntimeMode.SINGLE_THREADED,
        routing: RoutingEngine | None = None,
        registry: AgentRegistry | None = None,
        scheduler: PerAgentMailboxScheduler | None = None,
    ) -> None:
        """Create a runtime engine with default in-process components."""
        self._routing = routing or RoutingEngine()
        self._registry = registry or AgentRegistry()
        self._scheduler = scheduler or PerAgentMailboxScheduler()
        self._dispatcher = Dispatcher(registry=self._registry)
        self._mode = mode
        self._environment = self._build_environment()

    async def start(self) -> None:
        """Start runtime environment execution."""
        await self._environment.start()

    async def stop(self) -> None:
        """Stop runtime environment execution immediately."""
        await self._environment.stop()

    async def stop_when_idle(self) -> None:
        """Drain pending work and stop runtime execution."""
        await self._environment.stop_when_idle()

    def register_factory(
        self,
        agent_type: AgentType | str,
        factory: AgentFactory,
    ) -> AgentType:
        """Register a lazy agent factory."""
        resolved_type = self._coerce_agent_type(agent_type)
        self._registry.register_factory(resolved_type, factory)
        return resolved_type

    def register_instance(self, agent_id: AgentId, instance: Agent) -> AgentId:
        """Register an already instantiated agent."""
        self._registry.register_instance(agent_id, instance)
        return agent_id

    def add_subscription(self, subscription: Subscription) -> str:
        """Add a publish subscription and return its id."""
        self._routing.add_subscription(subscription)
        return subscription.id

    def remove_subscription(self, subscription_id: str) -> None:
        """Remove a publish subscription by id."""
        self._routing.remove_subscription(subscription_id)

    async def send_message(
        self,
        message: object,
        recipient: AgentId | AgentType | str,
        *,
        sender: AgentId | None = None,
        key: str | None = None,
        correlation_id: CorrelationId | None = None,
        attributes: dict[str, str] | None = None,
    ) -> DeliveryOutcome:
        """Send an RPC-style message and await terminal delivery outcome."""
        await self.start()
        correlation = correlation_id or CorrelationId.new()
        try:
            recipient_id = self._resolve_recipient(recipient=recipient, key=key)
        except LookupError as exc:
            return DeliveryOutcome.failed(
                status=DeliveryStatus.POLICY_REJECTED,
                message_id=MessageId.new(),
                correlation_id=correlation,
                message=str(exc),
                retryable=False,
            )

        envelope = MessageEnvelope.new_rpc_request(
            sender=sender,
            recipient=recipient_id,
            payload=self._payload_for(message),
            correlation_id=correlation,
            attributes=attributes,
        )
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        task = DeliveryTask(
            envelope=envelope,
            recipient=recipient_id,
            response_future=future,
        )
        try:
            await self._environment.submit(task)
        except SchedulerRejectedError as exc:
            return DeliveryOutcome.failed(
                status=DeliveryStatus.POLICY_REJECTED,
                message_id=envelope.message_id,
                correlation_id=envelope.correlation_id,
                message=str(exc),
                retryable=False,
            )
        outcome = await future
        return cast(DeliveryOutcome, outcome)

    async def publish_message(
        self,
        message: object,
        topic: TopicId,
        *,
        sender: AgentId | None = None,
        correlation_id: CorrelationId | None = None,
        attributes: dict[str, str] | None = None,
    ) -> PublishAck:
        """Publish a message and return enqueue acknowledgment only."""
        await self.start()
        correlation = correlation_id or CorrelationId.new()
        base_envelope = MessageEnvelope.new_publish_event(
            sender=sender,
            topic=topic,
            payload=self._payload_for(message),
            correlation_id=correlation,
            attributes=attributes,
        )

        recipients = self._routing.resolve_publish_recipients(base_envelope)
        for recipient in recipients:
            recipient_envelope = MessageEnvelope.new_publish_event(
                sender=sender,
                topic=topic,
                payload=self._payload_for(message),
                correlation_id=correlation,
                attributes=attributes,
            )
            task = DeliveryTask(
                envelope=recipient_envelope,
                recipient=recipient,
                response_future=None,
            )
            await self._environment.submit(task)

        return PublishAck(
            message_id=base_envelope.message_id,
            correlation_id=base_envelope.correlation_id,
            enqueued_recipient_count=len(recipients),
            enqueued_at_ms=utc_now_ms(),
        )

    def _build_environment(self) -> RuntimeEnvironment:
        """Build runtime environment for the configured mode."""
        if self._mode == RuntimeMode.SINGLE_THREADED:
            return SingleThreadedEnvironment(
                scheduler=self._scheduler,
                dispatcher=self._dispatcher,
            )
        if self._mode == RuntimeMode.DISTRIBUTED:
            return DistributedEnvironment()
        raise ValueError(f"Unsupported runtime mode: {self._mode}")

    @staticmethod
    def _coerce_agent_type(agent_type: AgentType | str) -> AgentType:
        """Normalize agent type argument."""
        if isinstance(agent_type, AgentType):
            return agent_type
        return AgentType(agent_type)

    def _resolve_recipient(
        self,
        *,
        recipient: AgentId | AgentType | str,
        key: str | None,
    ) -> AgentId:
        """Resolve recipient from explicit id or type(+optional key)."""
        if isinstance(recipient, AgentId):
            return recipient
        agent_type = self._coerce_agent_type(recipient)
        agent_key = AgentKey(key) if key is not None else None
        return self._registry.resolve_agent_id(agent_type, agent_key)

    @staticmethod
    def _payload_for(message: object) -> Payload:
        """Wrap an application message into canonical payload shape."""
        return Payload(
            schema_name=type(message).__name__,
            content_type="application/python-object",
            format=PayloadFormat.JSON,
            data=message,
        )
