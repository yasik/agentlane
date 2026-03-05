"""Runtime engine implementations for different execution environments."""

import abc
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
from ._registry import AgentFactory, AgentRegistry
from ._scheduler import (
    PerAgentMailboxScheduler,
    SchedulerRejectedError,
)
from ._types import DeliveryTask

_IN_FLIGHT_CANCELED = "Runtime shutdown canceled an in-flight delivery."
_QUEUED_CANCELED = "Runtime shutdown canceled a queued delivery."


def utc_now_ms() -> int:
    """Return current UTC epoch milliseconds."""
    return int(time() * 1000)


class RuntimeEngine(abc.ABC):
    """Base runtime engine that owns shared routing/registry/message orchestration."""

    def __init__(
        self,
        *,
        routing: RoutingEngine | None = None,
        registry: AgentRegistry | None = None,
        scheduler: PerAgentMailboxScheduler | None = None,
    ) -> None:
        """Create runtime components shared across runtime implementations."""
        self._routing = routing or RoutingEngine()
        self._registry = registry or AgentRegistry()
        self._scheduler = scheduler or PerAgentMailboxScheduler()
        self._dispatcher = Dispatcher(registry=self._registry)
        self._is_started = False

    async def start(self) -> None:
        """Start runtime execution if not already running."""
        if self._is_started:
            return

        # Mark started before creating worker tasks so they observe running state.
        self._is_started = True
        try:
            await self._start_runtime()
        except BaseException:
            self._is_started = False
            raise

    async def stop(self) -> None:
        """Stop runtime immediately if it is running."""
        if not self._is_started:
            return
        try:
            await self._stop_runtime()
        finally:
            self._is_started = False

    async def stop_when_idle(self) -> None:
        """Drain pending work and stop runtime if it is running."""
        if not self._is_started:
            return
        try:
            await self._stop_when_idle_runtime()
        finally:
            self._is_started = False

    @property
    def is_running(self) -> bool:
        """Return True when runtime was started and not stopped yet."""
        return self._is_started

    @abc.abstractmethod
    async def _start_runtime(self) -> None:
        """Start runtime-specific execution resources."""

    @abc.abstractmethod
    async def _stop_runtime(self) -> None:
        """Stop runtime-specific execution resources immediately."""

    @abc.abstractmethod
    async def _stop_when_idle_runtime(self) -> None:
        """Drain pending work and then stop runtime-specific resources."""

    @abc.abstractmethod
    async def _submit(self, task: DeliveryTask) -> None:
        """Submit one resolved delivery task to runtime-specific execution path."""

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
            await self._submit(task)
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
            await self._submit(task)

        return PublishAck(
            message_id=base_envelope.message_id,
            correlation_id=base_envelope.correlation_id,
            enqueued_recipient_count=len(recipients),
            enqueued_at_ms=utc_now_ms(),
        )

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


class SingleThreadedRuntimeEngine(RuntimeEngine):
    """In-process runtime with worker pool execution and mailbox-ordered dispatch."""

    def __init__(
        self,
        *,
        routing: RoutingEngine | None = None,
        registry: AgentRegistry | None = None,
        scheduler: PerAgentMailboxScheduler | None = None,
        worker_count: int = 10,
    ) -> None:
        """Create a single-threaded runtime with configurable worker count."""
        if worker_count <= 0:
            raise ValueError("worker_count must be greater than zero.")

        self._worker_count = worker_count
        # Worker pool executes tasks concurrently across distinct recipients.
        self._worker_pool: set[asyncio.Task[None]] = set()
        # Tracks current dispatch per worker so shutdown can cancel in-flight tasks.
        self._inflight_by_worker: dict[asyncio.Task[None], DeliveryTask] = {}

        super().__init__(routing=routing, registry=registry, scheduler=scheduler)

    async def _start_runtime(self) -> None:
        """Start in-process worker pool."""
        # Workers are detached; lifecycle is controlled by start/stop methods.
        for _ in range(self._worker_count):
            worker_task = asyncio.create_task(self._run_worker())
            self._worker_pool.add(worker_task)

    async def _stop_runtime(self) -> None:
        """Stop immediately and cancel both in-flight and queued deliveries."""
        # Ask currently running handlers to cooperate via context cancellation token.
        for inflight_task in self._inflight_by_worker.values():
            inflight_task.cancellation_token.cancel()

        if self._worker_pool:
            # Forcefully stop worker loops; in-flight tasks become CANCELED outcomes.
            for worker_task in self._worker_pool:
                worker_task.cancel()
            await asyncio.gather(*self._worker_pool, return_exceptions=True)
            self._worker_pool.clear()
            self._inflight_by_worker.clear()

        # Any task not yet dispatched is canceled eagerly during shutdown.
        queued_tasks = await self._scheduler.drain()
        for task in queued_tasks:
            self._cancel_task(task=task, message=_QUEUED_CANCELED)

    async def _stop_when_idle_runtime(self) -> None:
        """Wait for scheduled work to finish, then stop immediately."""
        await self._scheduler.wait_idle()
        await self._stop_runtime()

    async def _submit(self, task: DeliveryTask) -> None:
        """Queue one task through scheduler for worker dispatch."""
        await self._scheduler.enqueue(task)

    async def _run_worker(self) -> None:
        """Worker loop processing tasks one-by-one for multiple recipients."""
        while self.is_running:
            task = await self._scheduler.pop_next()
            worker_task = asyncio.current_task()
            if worker_task is not None:
                # Mark inflight before dispatch so stop() can target active work.
                self._inflight_by_worker[worker_task] = task
            try:
                outcome = await self._dispatcher.dispatch(task)
                # RPC path: complete awaiting caller with terminal delivery outcome.
                if task.response_future is not None and not task.response_future.done():
                    task.response_future.set_result(outcome)
            except asyncio.CancelledError:
                # Worker cancellation during dispatch maps to a canceled delivery outcome.
                self._cancel_task(task=task, message=_IN_FLIGHT_CANCELED)
                raise
            finally:
                if worker_task is not None:
                    self._inflight_by_worker.pop(worker_task, None)
                # Every popped task must decrement pending count exactly once.
                await self._scheduler.mark_done(task.recipient)

    def _cancel_task(self, task: DeliveryTask, message: str) -> None:
        """Request cancellation and resolve pending response future if present."""
        # Token cancellation lets cooperative handlers terminate quickly.
        task.cancellation_token.cancel()
        if task.response_future is None or task.response_future.done():
            return

        # Normalize cancellation into DeliveryOutcome so callers get a terminal result.
        task.response_future.set_result(
            DeliveryOutcome.failed(
                status=DeliveryStatus.CANCELED,
                message_id=task.envelope.message_id,
                correlation_id=task.envelope.correlation_id,
                message=message,
                retryable=True,
            )
        )


class DistributedRuntimeEngine(RuntimeEngine):
    """Distributed runtime placeholder that keeps lifecycle API but rejects submit."""

    async def _start_runtime(self) -> None:
        """Start distributed runtime resources."""
        return

    async def _stop_runtime(self) -> None:
        """Stop distributed runtime resources."""
        return

    async def _stop_when_idle_runtime(self) -> None:
        """Drain distributed runtime work."""
        return

    async def _submit(self, task: DeliveryTask) -> None:
        """Reject task submission until distributed transport/placement exists."""
        _ = task
        raise NotImplementedError(
            "DistributedRuntimeEngine is not implemented yet. "
            "Use SingleThreadedRuntimeEngine for v1 execution."
        )
