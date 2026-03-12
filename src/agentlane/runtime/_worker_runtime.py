"""Distributed worker runtime implementations.

`WorkerAgentRuntime` is the execution node in a distributed deployment. It keeps
the existing in-process scheduler/dispatcher behavior, but forwards RPC and
publish traffic through a host so message routing and worker ownership decisions
stay centralized.
"""

import asyncio
from typing import cast

from grpc import aio as grpc_aio

from agentlane.messaging import (
    AgentId,
    AgentType,
    CancellationToken,
    CorrelationId,
    DeliveryMode,
    DeliveryOutcome,
    DeliveryStatus,
    IdempotencyKey,
    MessageEnvelope,
    MessageId,
    PublishAck,
    Subscription,
    TopicId,
)
from agentlane.util import utc_now_ms

from ._distributed_grpc import (
    HostServiceStub,
    JsonObject,
    create_worker_generic_handler,
)
from ._distributed_wire import (
    WireDeliveryOutcome,
    WireEnvelope,
    serialize_agent_types,
    serialize_subscriptions,
)
from ._message_helpers import payload_from_value
from ._network import resolve_bound_address
from ._protocol import Agent
from ._registry import AgentFactory
from ._runtime import SingleThreadedRuntimeEngine
from ._worker_runtime_host import WorkerAgentRuntimeHost


def _failed_outcome(
    *,
    status: DeliveryStatus,
    message_id: MessageId,
    correlation_id: CorrelationId | None,
    message: str,
    retryable: bool,
) -> DeliveryOutcome:
    return DeliveryOutcome.failed(
        status=status,
        message_id=message_id,
        correlation_id=correlation_id,
        message=message,
        retryable=retryable,
        started_at_ms=utc_now_ms(),
    )


class WorkerAgentRuntime(SingleThreadedRuntimeEngine):
    """Distributed worker runtime backed by the local single-threaded executor.

    The worker owns application code execution, local registry state, and local
    subscriptions. The host owns cross-worker routing, worker discovery, and the
    lifecycle of in-flight RPC sessions.
    """

    def __init__(
        self,
        *,
        host_address: str | None,
        address: str = "127.0.0.1:0",
        worker_count: int = 10,
    ) -> None:
        super().__init__(worker_count=worker_count)
        self._host_address = host_address
        self._bind_address = address
        self._resolved_address = address
        self._host_channel: grpc_aio.Channel | None = None
        self._host_client: HostServiceStub | None = None
        self._server: grpc_aio.Server | None = None
        self._worker_id: str | None = None
        self._advertised_agent_types: set[AgentType] = set()
        self._catalog_dirty = False
        self._catalog_sync_task: asyncio.Task[None] | None = None

    @property
    def address(self) -> str:
        """Return the resolved worker bind address."""
        return self._resolved_address

    @property
    def worker_id(self) -> str | None:
        """Return the assigned worker id after registration."""
        return self._worker_id

    def register_factory(
        self,
        agent_type: AgentType | str,
        factory: AgentFactory,
    ) -> AgentType:
        """Register one agent factory and advertise the new type to the host."""
        resolved_type = super().register_factory(agent_type, factory)
        self._advertised_agent_types.add(resolved_type)
        self._schedule_catalog_sync()
        return resolved_type

    def register_instance(self, agent_id: AgentId, instance: Agent) -> AgentId:
        """Register one bound agent instance and advertise its type to the host."""
        registered_id = super().register_instance(agent_id, instance)
        self._advertised_agent_types.add(agent_id.type)
        self._schedule_catalog_sync()
        return registered_id

    def add_subscription(self, subscription: Subscription) -> str:
        """Add one subscription locally and push the updated snapshot to the host."""
        subscription_id = super().add_subscription(subscription)
        self._schedule_catalog_sync()
        return subscription_id

    def remove_subscription(self, subscription_id: str) -> None:
        """Remove one subscription locally and push the updated snapshot to the host."""
        super().remove_subscription(subscription_id)
        self._schedule_catalog_sync()

    def subscribe_exact(
        self,
        *,
        topic_type: str,
        agent_type: AgentType | str,
        delivery_mode: DeliveryMode = DeliveryMode.STATEFUL,
    ) -> str:
        """Register one exact-topic subscription and advertise it to the host."""
        subscription = Subscription.exact(
            topic_type=topic_type,
            agent_type=agent_type,
            delivery_mode=delivery_mode,
        )
        return self.add_subscription(subscription)

    def subscribe_prefix(
        self,
        *,
        topic_prefix: str,
        agent_type: AgentType | str,
        delivery_mode: DeliveryMode = DeliveryMode.STATEFUL,
    ) -> str:
        """Register one prefix subscription and advertise it to the host."""
        subscription = Subscription.prefix(
            topic_prefix=topic_prefix,
            agent_type=agent_type,
            delivery_mode=delivery_mode,
        )
        return self.add_subscription(subscription)

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
        """Send one direct RPC through the host after catalog state is synchronized.

        The current distributed transport does not propagate
        `cancellation_token` across the process boundary yet.
        """
        await self.start()
        # A freshly registered factory or subscription must be visible to the host
        # before it decides where this message should be routed.
        await self._await_catalog_sync()
        if self._host_client is None:
            raise RuntimeError("Worker runtime is not connected to a host.")

        correlation = correlation_id or CorrelationId.new()
        try:
            recipient_id = self._resolve_recipient(recipient=recipient)
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
            payload=payload_from_value(message),
            correlation_id=correlation,
            idempotency_key=idempotency_key,
        )
        try:
            # The host remains the single routing authority even for worker-originated
            # RPCs, so every outbound direct message crosses the host first.
            response = await self._host_client.send_rpc(
                {
                    "envelope": WireEnvelope.from_envelope(
                        envelope,
                        serializer_registry=self.serializer_registry,
                    ).to_json()
                }
            )
        except grpc_aio.AioRpcError as exc:
            return _failed_outcome(
                status=DeliveryStatus.UNDELIVERABLE,
                message_id=envelope.message_id,
                correlation_id=envelope.correlation_id,
                message=f"Host RPC submission failed: {exc}",
                retryable=True,
            )

        # The host returns a wire-shaped outcome so response payload decoding still
        # happens at the worker edge that initiated the RPC.
        return WireDeliveryOutcome.from_json(response).to_outcome(
            serializer_registry=self.serializer_registry
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
        """Publish one event through the host after catalog state is synchronized.

        The current distributed transport does not propagate
        `cancellation_token` across the process boundary yet.
        """
        await self.start()
        # Publish routing also depends on the latest subscription snapshot having
        # reached the host before the topic is resolved.
        await self._await_catalog_sync()
        if self._host_client is None:
            raise RuntimeError("Worker runtime is not connected to a host.")

        envelope = MessageEnvelope.new_publish_event(
            sender=sender,
            topic=topic,
            payload=payload_from_value(message),
            correlation_id=correlation_id or CorrelationId.new(),
            idempotency_key=idempotency_key,
        )
        try:
            response = await self._host_client.publish(
                {
                    "envelope": WireEnvelope.from_envelope(
                        envelope,
                        serializer_registry=self.serializer_registry,
                    ).to_json()
                }
            )
        except grpc_aio.AioRpcError as exc:
            raise RuntimeError(f"Host publish submission failed: {exc}") from exc

        if response.get("ok") is not True:
            raise RuntimeError(str(response.get("error", "Publish rejected by host.")))

        return PublishAck.from_json(response["ack"])

    async def deliver_rpc(self, request: JsonObject) -> JsonObject:
        """Handle one host-originated direct delivery."""
        wire_envelope = WireEnvelope.from_json(request["envelope"])
        envelope = wire_envelope.to_envelope(
            serializer_registry=self.serializer_registry
        )

        if envelope.recipient is None:
            outcome = _failed_outcome(
                status=DeliveryStatus.UNDELIVERABLE,
                message_id=envelope.message_id,
                correlation_id=envelope.correlation_id,
                message="Distributed direct delivery requires a recipient.",
                retryable=False,
            )
        else:
            # Once the host has picked this worker, execution falls back to the
            # exact same local submit path used by the in-process runtime.
            outcome = await self._submit_rpc_task(
                envelope=envelope,
                recipient=envelope.recipient,
                cancellation_token=CancellationToken(),
            )

        return WireDeliveryOutcome.from_outcome(
            outcome,
            serializer_registry=self.serializer_registry,
        ).to_json()

    async def deliver_publish(self, request: JsonObject) -> JsonObject:
        """Handle one host-originated publish fan-out batch."""
        deliveries = request.get("deliveries")
        if not isinstance(deliveries, list):
            return {"ok": False, "error": "Publish delivery requires a list."}
        delivery_entries = cast(list[object], deliveries)

        for delivery_entry in delivery_entries:
            if not isinstance(delivery_entry, dict):
                return {"ok": False, "error": "Invalid publish delivery payload."}

            delivery_mapping = cast(JsonObject, delivery_entry)
            wire_envelope = WireEnvelope.from_json(delivery_mapping["envelope"])
            envelope = wire_envelope.to_envelope(
                serializer_registry=self.serializer_registry
            )

            if envelope.recipient is None:
                return {
                    "ok": False,
                    "error": "Host publish delivery missing recipient.",
                }

            # The host already performed worker-level fan-out. At this point the
            # worker only needs to enqueue each concrete local delivery.
            await self._submit_publish_task(
                envelope=envelope,
                recipient=envelope.recipient,
                cancellation_token=CancellationToken(),
            )

        return {"ok": True, "enqueued_recipient_count": len(delivery_entries)}

    async def health_check(self, request: JsonObject) -> JsonObject:
        """Return worker readiness."""
        _ = request
        return {"ok": True, "worker_id": self._worker_id}

    async def _start_runtime(self) -> None:
        """Start local execution resources, then register this worker with the host."""
        await super()._start_runtime()

        server = grpc_aio.server()
        server.add_generic_rpc_handlers((create_worker_generic_handler(self),))
        bound_port = server.add_insecure_port(self._bind_address)
        if bound_port == 0:
            await super()._stop_runtime()
            raise RuntimeError(f"Failed to bind worker address '{self._bind_address}'.")

        self._resolved_address = resolve_bound_address(self._bind_address, bound_port)
        await server.start()
        self._server = server

        # The worker endpoint must be live before registration because the host
        # performs an immediate health check as part of accepting the worker.
        if self._host_address is None:
            await self._close_transport_resources()
            await super()._stop_runtime()
            raise RuntimeError("Worker runtime requires a host address before start().")

        self._host_channel = grpc_aio.insecure_channel(self._host_address)
        self._host_client = HostServiceStub(self._host_channel)

        try:
            # The local worker endpoint must already be listening so the host can
            # validate it immediately via health check during registration.
            register_response = await self._host_client.register_worker(
                {"address": self.address}
            )
            if register_response.get("ok") is not True:
                raise RuntimeError(
                    str(
                        register_response.get(
                            "error",
                            "Worker registration was rejected by host.",
                        )
                    )
                )

            worker_id = register_response.get("worker_id")
            if not isinstance(worker_id, str) or not worker_id:
                raise RuntimeError("Host returned an invalid worker id.")

            self._worker_id = worker_id
            # Registration only makes the worker reachable. A follow-up full
            # snapshot tells the host which agent types and subscriptions it owns.
            await self._sync_catalog_once()
        except Exception as e:
            await self._close_transport_resources()
            await super()._stop_runtime()
            raise e

    async def _stop_runtime(self) -> None:
        """Stop the worker immediately, deregistering before transport teardown."""
        await self._deregister_from_host()
        await self._close_transport_resources()
        await super()._stop_runtime()

    async def _stop_when_idle_runtime(self) -> None:
        """Drain local work before tearing down transport resources."""
        # Remove this worker from host routing first so no new cross-worker traffic
        # arrives while the local scheduler is draining its current queue.
        await self._deregister_from_host()
        await super()._stop_when_idle_runtime()
        await self._close_transport_resources()

    def _schedule_catalog_sync(self) -> None:
        """Queue one snapshot sync if the worker is already running."""
        if not self.is_running:
            return

        self._catalog_dirty = True
        if self._catalog_sync_task is None or self._catalog_sync_task.done():
            self._catalog_sync_task = asyncio.create_task(self._run_catalog_sync())

    async def _run_catalog_sync(self) -> None:
        """Coalesce local catalog changes into snapshot pushes to the host."""
        while self._catalog_dirty and self.is_running:
            # Clear first so a concurrent local mutation can mark the snapshot dirty
            # again and force one more sync after the current push completes.
            self._catalog_dirty = False
            await self._sync_catalog_once()

    async def _await_catalog_sync(self) -> None:
        """Wait for the most recent catalog push to finish, if one is queued."""
        if self._catalog_sync_task is None:
            return
        await self._catalog_sync_task

    async def _sync_catalog_once(self) -> None:
        """Push the current agent-type and subscription snapshot to the host."""
        if self._host_client is None or self._worker_id is None:
            return
        # Catalog sync is full-state replacement rather than incremental mutation.
        response = await self._host_client.sync_catalog(
            {
                "worker_id": self._worker_id,
                "agent_types": serialize_agent_types(self._advertised_agent_types),
                "subscriptions": serialize_subscriptions(self.list_subscriptions()),
            }
        )
        if response.get("ok") is not True:
            raise RuntimeError(
                str(response.get("error", "Worker catalog sync was rejected by host."))
            )

    async def _deregister_from_host(self) -> None:
        """Best-effort deregistration used by both immediate and graceful shutdown."""
        await self._await_catalog_sync()
        if self._host_client is None or self._worker_id is None:
            return
        try:
            await self._host_client.deregister_worker({"worker_id": self._worker_id})
        except grpc_aio.AioRpcError:
            return

    async def _close_transport_resources(self) -> None:
        """Close worker transport state without raising late sync failures."""
        if self._catalog_sync_task is not None:
            # Shutdown should not surface a stale sync error once the runtime is
            # already tearing down transport resources.
            await asyncio.gather(self._catalog_sync_task, return_exceptions=True)
            self._catalog_sync_task = None
        if self._host_channel is not None:
            # Drop the client reference together with the channel so later shutdown
            # steps cannot accidentally issue one more host RPC.
            await self._host_channel.close()
            self._host_channel = None
            self._host_client = None
        if self._server is not None:
            await self._server.stop(grace=None)
            self._server = None
        self._worker_id = None


class DistributedRuntimeEngine(WorkerAgentRuntime):
    """Zero-config convenience wrapper over one managed host and worker.

    This is the default distributed entry point used by `distributed_runtime()`.
    When no host address is supplied it provisions a host in-process, waits for it
    to start, and then connects the primary worker to that host.
    """

    def __init__(
        self, *, host_address: str | None = None, address: str = "127.0.0.1:0"
    ) -> None:
        self._managed_host = (
            None
            if host_address is not None
            else WorkerAgentRuntimeHost(address="127.0.0.1:0")
        )
        super().__init__(host_address=host_address, address=address)

    async def _start_runtime(self) -> None:
        """Start the managed host first, then start the worker runtime itself."""
        if self._managed_host is not None:
            await self._managed_host.start()
            self._host_address = self._managed_host.address
        await super()._start_runtime()

    async def _stop_runtime(self) -> None:
        """Stop the worker and then tear down the managed host immediately."""
        try:
            await super()._stop_runtime()
        finally:
            if self._managed_host is not None:
                await self._managed_host.stop()

    async def _stop_when_idle_runtime(self) -> None:
        """Drain worker work first, then gracefully stop the managed host."""
        try:
            await super()._stop_when_idle_runtime()
        finally:
            if self._managed_host is not None:
                await self._managed_host.stop_when_idle()
