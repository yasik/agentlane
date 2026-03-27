"""Runtime engine implementations for different execution environments.

This module provides:

1. `RuntimeEngine`: a shared orchestration layer for messaging APIs.
2. `SingleThreadedRuntimeEngine`: in-process execution via scheduler + worker pool.

`RuntimeEngine` wires high-level behavior:

1. Routing (`RoutingEngine`) for publish recipient resolution.
2. Registry (`AgentRegistry`) for factory/instance lifecycle.
3. Scheduler (`PerAgentMailboxScheduler`) as execution queue boundary.
4. Dispatcher (`Dispatcher`) for handler invocation and delivery outcomes.
"""

import abc
import asyncio
from collections.abc import Sequence
from typing import cast

from agentlane.messaging import (
    AgentId,
    AgentType,
    CorrelationId,
    DeliveryMode,
    DeliveryOutcome,
    DeliveryStatus,
    IdempotencyKey,
    MessageEnvelope,
    MessageId,
    Payload,
    PublishAck,
    RoutingEngine,
    Subscription,
    TopicId,
)
from agentlane.transport import (
    MessageSerializer,
    SerializerRegistry,
    WirePayload,
    create_default_serializer_registry,
    payload_to_wire_payload,
    wire_payload_to_payload,
)
from agentlane.util import utc_now_ms

from ._cancellation import CancellationToken
from ._dispatcher import Dispatcher
from ._engine import Engine
from ._message_helpers import payload_from_value, recipient_for_publish_route
from ._protocol import Agent
from ._registry import AgentFactory, AgentRegistry
from ._scheduler import (
    PerAgentMailboxScheduler,
    SchedulerRejectedError,
)
from ._types import DeliveryTask

_IN_FLIGHT_CANCELED = "Runtime shutdown canceled an in-flight delivery."
_QUEUED_CANCELED = "Runtime shutdown canceled a queued delivery."


class RuntimeEngine(Engine, abc.ABC):
    """Base runtime engine that coordinates messaging APIs and runtime components.

    The base class owns the public contract (`send_message`, `publish_message`,
    registration, subscriptions) and delegates execution details to subclasses via
    runtime hooks (`_start_runtime`, `_submit`, `_stop_runtime`).

    Concrete runtimes differ in *where/how* tasks execute, while all share:

    1. Envelope construction and correlation-id handling.
    2. Routing and recipient resolution semantics.
    3. Delivery outcome shape and policy rejection behavior.
    """

    def __init__(
        self,
        *,
        routing: RoutingEngine | None = None,
        registry: AgentRegistry | None = None,
        scheduler: PerAgentMailboxScheduler | None = None,
        serializer_registry: SerializerRegistry | None = None,
    ) -> None:
        """Create shared runtime dependencies and baseline state.

        Args:
            routing: Publish/RPC routing resolver. Defaults to `RoutingEngine()`.
            registry: Agent factory/instance registry. Defaults to `AgentRegistry()`.
            scheduler: Runtime scheduler used by concrete execution paths.
            serializer_registry: Transport serializer registry used for wire boundaries.
        """
        self._routing = routing or RoutingEngine()
        self._registry = registry or AgentRegistry()
        self._scheduler = scheduler or PerAgentMailboxScheduler()
        self._serializer_registry = (
            serializer_registry or create_default_serializer_registry()
        )
        self._dispatcher = Dispatcher(
            registry=self._registry,
            engine=self,
        )
        self._is_started = False

    async def start(self) -> None:
        """Start runtime execution if not already running.

        This method is idempotent and is also called implicitly by `send_message`
        and `publish_message`, so callers can use runtime APIs without a separate
        explicit startup step.

        Returns:
            None: Always returns after runtime startup attempt.

        Raises:
            Exception: Propagates runtime-specific startup failures.
        """
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
        """Stop runtime immediately if it is running.

        Concrete runtimes should treat this as eager shutdown:

        1. Cancel in-flight deliveries when possible.
        2. Reject or cancel queued deliveries.

        Returns:
            None: Always returns after runtime shutdown attempt.
        """
        if not self._is_started:
            return
        try:
            await self._stop_runtime()
        finally:
            self._is_started = False

    async def stop_when_idle(self) -> None:
        """Drain pending work and then stop runtime if it is running.

        This is the graceful shutdown path: queued work is allowed to complete
        before runtime-specific resources are torn down.

        Returns:
            None: Always returns after graceful shutdown attempt.
        """
        if not self._is_started:
            return
        try:
            await self._stop_when_idle_runtime()
        finally:
            self._is_started = False

    @property
    def is_running(self) -> bool:
        """Return True when runtime was started and not stopped yet.

        Returns:
            bool: Runtime running state.
        """
        return self._is_started

    @property
    def serializer_registry(self) -> SerializerRegistry:
        """Return transport serializer registry used by this runtime.

        Returns:
            SerializerRegistry: Runtime-owned serializer registry instance.
        """
        return self._serializer_registry

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
        """Submit one resolved delivery task to runtime-specific execution path.

        Subclasses implement how tasks are queued/executed, but must preserve
        delivery semantics expected by the shared messaging APIs.
        """

    def register_factory(
        self,
        agent_type: AgentType | str,
        factory: AgentFactory,
    ) -> AgentType:
        """Register a lazy factory for an agent type.

        The returned `AgentType` is normalized and can be reused by callers for
        subscription setup and direct addressing.

        Factory signature:

        1. `factory(engine)` where `engine` is this runtime's restricted engine view.

        Args:
            agent_type: Logical agent type string or typed wrapper.
            factory: Sync or async factory receiving this runtime's `Engine` view.

        Returns:
            AgentType: Normalized registered agent type.
        """
        resolved_type = self._coerce_agent_type(agent_type)
        self._registry.register_factory(resolved_type, factory)
        return resolved_type

    def register_instance(self, agent_id: AgentId, instance: Agent) -> AgentId:
        """Register an already instantiated agent instance by `AgentId`.

        Args:
            agent_id: Concrete id to bind and register.
            instance: Agent instance to register.

        Returns:
            AgentId: Same registered id for call-site chaining.
        """
        self._registry.register_instance(agent_id, instance)
        return agent_id

    def add_subscription(self, subscription: Subscription) -> str:
        """Add a publish subscription object and return its stable id.

        Args:
            subscription: Subscription to add or replace by id.

        Returns:
            str: Stable subscription id.
        """
        self._routing.add_subscription(subscription)
        return subscription.id

    def remove_subscription(self, subscription_id: str) -> None:
        """Remove a publish subscription by id.

        Args:
            subscription_id: Stable subscription id to remove.

        Returns:
            None: Always returns after removal.

        Raises:
            LookupError: If subscription does not exist.
        """
        self._routing.remove_subscription(subscription_id)

    def subscribe_exact(
        self,
        *,
        topic_type: str,
        agent_type: AgentType | str,
        delivery_mode: DeliveryMode = DeliveryMode.STATEFUL,
    ) -> str:
        """Create and register an exact topic subscription.

        This is the preferred API over constructing `Subscription` manually.

        Args:
            topic_type: Exact topic type to match.
            agent_type: Agent type receiving matched messages.
            delivery_mode: Stateful or stateless delivery behavior.

        Returns:
            str: Stable subscription id.
        """
        subscription = Subscription.exact(
            topic_type=topic_type,
            agent_type=agent_type,
            delivery_mode=delivery_mode,
        )
        self._routing.add_subscription(subscription)
        return subscription.id

    def subscribe_prefix(
        self,
        *,
        topic_prefix: str,
        agent_type: AgentType | str,
        delivery_mode: DeliveryMode = DeliveryMode.STATEFUL,
    ) -> str:
        """Create and register a prefix topic subscription.

        Args:
            topic_prefix: Topic prefix matched with `startswith`.
            agent_type: Agent type receiving matched messages.
            delivery_mode: Stateful or stateless delivery behavior.

        Returns:
            str: Stable subscription id.
        """
        subscription = Subscription.prefix(
            topic_prefix=topic_prefix,
            agent_type=agent_type,
            delivery_mode=delivery_mode,
        )
        self._routing.add_subscription(subscription)
        return subscription.id

    def unsubscribe(self, subscription_id: str) -> None:
        """Alias for remove_subscription for explicit public API naming.

        Args:
            subscription_id: Stable subscription id to remove.

        Returns:
            None: Always returns after removal.
        """
        self.remove_subscription(subscription_id)

    def list_subscriptions(self) -> tuple[Subscription, ...]:
        """Return immutable snapshot of current subscriptions.

        Returns:
            tuple[Subscription, ...]: Current subscription snapshot.
        """
        return tuple(self._routing.subscriptions)

    def register_serializer(
        self,
        serializer: MessageSerializer | Sequence[MessageSerializer],
        *,
        replace: bool = False,
    ) -> None:
        """Register one or more transport serializers on runtime-owned registry.

        This is an advanced escape hatch. Common dataclass/pydantic/protobuf
        payloads are auto-inferred by default and do not require registration.

        Args:
            serializer: One serializer or sequence of serializers.
            replace: Whether to replace an existing key on conflict.

        Returns:
            None: Always returns after registry updates.

        Raises:
            SerializerConflictError: If key exists and `replace=False`.
        """
        if isinstance(serializer, MessageSerializer):
            self._serializer_registry.register(serializer, replace=replace)
            return
        self._serializer_registry.register_many(serializer, replace=replace)

    def register_message_type(
        self,
        message_type: type[object],
        *,
        replace: bool = False,
    ) -> None:
        """Register one message type using default serializer inference.

        Args:
            message_type: Python type to register.
            replace: Whether to replace an existing serializer key.

        Returns:
            None: Always returns after successful registration.

        Raises:
            TypeError: If no default serializer can be inferred for the type.
            SerializerConflictError: If key exists and `replace=False`.
        """
        self._serializer_registry.register_type(message_type, replace=replace)

    def payload_to_wire_payload(self, payload: Payload) -> WirePayload:
        """Convert messaging payload into wire payload using runtime registry.

        Args:
            payload: Canonical messaging payload.

        Returns:
            WirePayload: Transport-ready payload bytes and metadata.

        Raises:
            SerializationError: If payload cannot be encoded consistently.
            UnknownSerializerError: If no serializer exists for payload key.
        """
        return payload_to_wire_payload(payload, registry=self._serializer_registry)

    def wire_payload_to_payload(self, wire_payload: WirePayload) -> Payload:
        """Convert wire payload into messaging payload using runtime registry.

        Args:
            wire_payload: Transport payload bytes and metadata.

        Returns:
            Payload: Canonical in-memory payload.

        Raises:
            SerializationError: If payload cannot be decoded consistently.
            UnknownSerializerError: If no serializer exists for payload key.
        """
        return wire_payload_to_payload(wire_payload, registry=self._serializer_registry)

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
        """Send one direct RPC-style message and await terminal delivery outcome.

        Flow:

        1. Normalize recipient (`AgentId` or type resolution).
        2. Build one RPC request envelope.
        3. Submit one delivery task into runtime execution path.
        4. Await dispatcher-completed future for terminal outcome.

        Args:
            message: Application payload to send.
            recipient: Target agent id or logical agent type.
            sender: Optional sender id propagated in context.
            correlation_id: Optional causal chain id.
            cancellation_token: Optional shared cancellation token.
            idempotency_key: Optional deduplication key.

        Returns:
            DeliveryOutcome: Terminal delivery outcome.
        """
        await self.start()
        correlation = correlation_id or CorrelationId.new()
        try:
            # Recipient may be explicit AgentId or type lookup.
            recipient_id = self._resolve_recipient(recipient=recipient)
        except LookupError as exc:
            # Unresolvable targets are rejected before enqueue.
            return DeliveryOutcome.failed(
                status=DeliveryStatus.POLICY_REJECTED,
                message_id=MessageId.new(),
                correlation_id=correlation,
                message=str(exc),
                retryable=False,
            )

        # RPC request envelope carries fixed recipient and shared correlation id.
        envelope = MessageEnvelope.new_rpc_request(
            sender=sender,
            recipient=recipient_id,
            payload=payload_from_value(message),
            correlation_id=correlation,
            idempotency_key=idempotency_key,
        )
        return await self._submit_rpc_task(
            envelope=envelope,
            recipient=recipient_id,
            cancellation_token=cancellation_token or CancellationToken(),
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
        """Publish one event message and return enqueue acknowledgment only.

        Flow:

        1. Build a base publish envelope for routing and ack identity.
        2. Resolve publish routes from subscriptions (`stateful`/`stateless`).
        3. Fan out one delivery task per route.
        4. Return `PublishAck` once all route tasks are enqueued.

        Note:
            The returned ack confirms enqueue, not handler completion.

        Args:
            message: Application payload to publish.
            topic: Topic id used for subscription routing.
            sender: Optional sender id propagated in context.
            correlation_id: Optional causal chain id.
            cancellation_token: Optional shared cancellation token.
            idempotency_key: Optional deduplication key.

        Returns:
            PublishAck: Enqueue acknowledgment across routed recipients.

        Raises:
            SchedulerRejectedError: If scheduler rejects a routed task enqueue.
        """
        await self.start()
        correlation = correlation_id or CorrelationId.new()
        # Base envelope is used for routing and enqueue acknowledgment.
        # Fan-out deliveries are emitted as separate envelopes per route.
        base_envelope = MessageEnvelope.new_publish_event(
            sender=sender,
            topic=topic,
            payload=payload_from_value(message),
            correlation_id=correlation,
            idempotency_key=idempotency_key,
        )

        # Routing resolves recipient + delivery mode per matching subscription.
        routes = self._routing.resolve_publish_routes(base_envelope)
        publish_cancellation_token = cancellation_token or CancellationToken()
        for route in routes:
            # Publish fan-out uses a new envelope per recipient so each delivery
            # has its own message_id while keeping shared correlation_id.
            recipient_envelope = MessageEnvelope.new_publish_event(
                sender=sender,
                topic=topic,
                payload=payload_from_value(message),
                correlation_id=correlation,
                idempotency_key=idempotency_key,
            )
            # Recipient identity may be rewritten for stateless mode.
            task_recipient = recipient_for_publish_route(
                route=route,
                publish_envelope=base_envelope,
            )
            task = DeliveryTask(
                envelope=recipient_envelope,
                recipient=task_recipient,
                cancellation_token=publish_cancellation_token,
                # Publish is fire-and-forget; callers receive PublishAck only.
                response_future=None,
            )
            await self._submit_publish_task(
                envelope=task.envelope,
                recipient=task.recipient,
                cancellation_token=task.cancellation_token,
            )

        # Publish ack confirms enqueue only, not downstream handler completion.
        return PublishAck(
            message_id=base_envelope.message_id,
            correlation_id=base_envelope.correlation_id,
            enqueued_recipient_count=len(routes),
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
    ) -> AgentId:
        """Resolve recipient from explicit id or type.

        Type-only targets delegate to registry resolution policy (for example,
        default instance resolution or ambiguity rejection when multiple active
        instances exist).
        """
        if isinstance(recipient, AgentId):
            return recipient
        agent_type = self._coerce_agent_type(recipient)
        return self._registry.resolve_agent_id(agent_type)

    async def _submit_rpc_task(
        self,
        *,
        envelope: MessageEnvelope,
        recipient: AgentId,
        cancellation_token: CancellationToken,
    ) -> DeliveryOutcome:
        """Submit one already-built RPC delivery task and await its outcome."""
        await self.start()
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        task = DeliveryTask(
            envelope=envelope,
            recipient=recipient,
            cancellation_token=cancellation_token,
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
        return cast(DeliveryOutcome, await future)

    async def _submit_publish_task(
        self,
        *,
        envelope: MessageEnvelope,
        recipient: AgentId,
        cancellation_token: CancellationToken,
    ) -> None:
        """Submit one already-built publish delivery task."""
        await self.start()
        await self._submit(
            DeliveryTask(
                envelope=envelope,
                recipient=recipient,
                cancellation_token=cancellation_token,
                response_future=None,
            )
        )


class SingleThreadedRuntimeEngine(RuntimeEngine):
    """In-process runtime with scheduler-backed worker pool execution.

    The scheduler enforces per-recipient serialization, while worker count
    controls cross-recipient concurrency.
    """

    def __init__(
        self,
        *,
        routing: RoutingEngine | None = None,
        registry: AgentRegistry | None = None,
        scheduler: PerAgentMailboxScheduler | None = None,
        serializer_registry: SerializerRegistry | None = None,
        worker_count: int = 10,
    ) -> None:
        """Create single-threaded runtime with configurable worker pool size."""
        if worker_count <= 0:
            raise ValueError("worker_count must be greater than zero.")

        self._worker_count = worker_count
        # Worker pool executes tasks concurrently across distinct recipients.
        self._worker_pool: set[asyncio.Task[None]] = set()
        # Tracks current dispatch per worker so shutdown can cancel in-flight tasks.
        self._inflight_by_worker: dict[asyncio.Task[None], DeliveryTask] = {}

        super().__init__(
            routing=routing,
            registry=registry,
            scheduler=scheduler,
            serializer_registry=serializer_registry,
        )

    async def _start_runtime(self) -> None:
        """Start detached worker tasks that pull from scheduler."""
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
        """Wait for scheduler idle, then perform eager stop cleanup."""
        await self._scheduler.wait_idle()
        await self._stop_runtime()

    async def _submit(self, task: DeliveryTask) -> None:
        """Queue one task into scheduler for eventual worker dispatch."""
        await self._scheduler.enqueue(task)

    async def _run_worker(self) -> None:
        """Worker loop that repeatedly pops tasks and dispatches handlers."""
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
