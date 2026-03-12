"""Distributed runtime host service.

The host is the control plane for distributed runtimes. It tracks connected
workers, owns agent-type placement, mirrors subscription state for publish
routing, and keeps the authoritative record of in-flight cross-worker RPC
sessions.
"""

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

from grpc import aio as grpc_aio

from agentlane.messaging import (
    AgentType,
    CorrelationId,
    DeliveryError,
    DeliveryStatus,
    MessageEnvelope,
    MessageId,
    PublishAck,
    RoutingEngine,
    Subscription,
)
from agentlane.util import utc_now_ms

from ._distributed_grpc import (
    JsonObject,
    WorkerServiceStub,
    create_host_generic_handler,
)
from ._distributed_wire import (
    WireDeliveryOutcome,
    WireEnvelope,
    deserialize_agent_types,
    deserialize_subscriptions,
)
from ._message_helpers import recipient_for_publish_route
from ._network import resolve_bound_address


def _failed_outcome_json(
    *,
    status: DeliveryStatus,
    message_id: str,
    correlation_id: str | None,
    message: str,
    retryable: bool,
) -> JsonObject:
    """Build one JSON-safe failed `DeliveryOutcome` payload."""
    started_at_ms = utc_now_ms()
    return WireDeliveryOutcome(
        status=status.value,
        message_id=message_id,
        correlation_id=correlation_id,
        response_payload=None,
        error=DeliveryError(
            code=status,
            message=message,
            retryable=retryable,
        ),
        started_at_ms=started_at_ms,
        finished_at_ms=utc_now_ms(),
    ).to_json()


def _metadata_envelope_from_wire(wire_envelope: WireEnvelope) -> MessageEnvelope:
    """Build a routing-only envelope from opaque wire payload metadata."""
    return wire_envelope.to_metadata_envelope()


def _new_agent_type_set() -> set[AgentType]:
    """Return an empty agent-type set with a concrete element type."""
    return set()


def _new_subscription_map() -> dict[str, Subscription]:
    """Return an empty subscription map with concrete key/value types."""
    return {}


@dataclass(slots=True)
class _PendingRpcSession:
    """In-flight direct message tracked by the host."""

    target_worker_id: str | None
    correlation_id: str | None
    future: asyncio.Future[JsonObject]


@dataclass(slots=True)
class _WorkerRecord:
    """Connected worker state tracked by the host."""

    worker_id: str
    address: str
    channel: grpc_aio.Channel
    client: WorkerServiceStub
    agent_types: set[AgentType] = field(default_factory=_new_agent_type_set)
    subscriptions: dict[str, Subscription] = field(
        default_factory=_new_subscription_map
    )
    failed_health_checks: int = 0


class WorkerAgentRuntimeHost:
    """Route messages between distributed worker runtimes.

    The host has three core jobs:

    1. Keep worker connections and health state.
    2. Decide which worker owns each advertised `AgentType`.
    3. Resolve direct RPC and publish traffic across worker boundaries.
    """

    def __init__(
        self,
        *,
        address: str = "127.0.0.1:0",
        routing: RoutingEngine | None = None,
        rpc_timeout_seconds: float = 30.0,
        health_check_interval_seconds: float = 5.0,
        health_check_failure_threshold: int = 2,
    ) -> None:
        self._bind_address = address
        self._resolved_address = address
        self._routing = routing or RoutingEngine()
        self._rpc_timeout_seconds = rpc_timeout_seconds
        self._health_check_interval_seconds = health_check_interval_seconds
        self._health_check_failure_threshold = health_check_failure_threshold
        self._server: grpc_aio.Server | None = None
        self._workers: dict[str, _WorkerRecord] = {}
        self._owners_by_agent_type: dict[AgentType, str] = {}
        self._pending_sessions: dict[str, _PendingRpcSession] = {}
        self._lock = asyncio.Lock()
        self._health_task: asyncio.Task[None] | None = None
        self._accepting_traffic = False
        self._is_running = False

    @property
    def address(self) -> str:
        """Return the resolved bind address for this host."""
        return self._resolved_address

    @property
    def is_running(self) -> bool:
        """Return whether the host is currently running."""
        return self._is_running

    async def start(self) -> None:
        """Start the host gRPC server and background health loop."""
        if self._is_running:
            return

        server = grpc_aio.server()
        server.add_generic_rpc_handlers((create_host_generic_handler(self),))
        bound_port = server.add_insecure_port(self._bind_address)
        if bound_port == 0:
            raise RuntimeError(f"Failed to bind host address '{self._bind_address}'.")

        self._resolved_address = resolve_bound_address(self._bind_address, bound_port)
        await server.start()

        self._server = server
        # Flip acceptance only after the gRPC server is listening and state is ready.
        self._accepting_traffic = True
        self._is_running = True
        self._health_task = asyncio.create_task(self._run_health_loop())

    async def stop(self) -> None:
        """Stop the host immediately and fail pending RPC sessions."""
        if not self._is_running:
            return

        self._accepting_traffic = False
        if self._health_task is not None:
            self._health_task.cancel()
            await asyncio.gather(self._health_task, return_exceptions=True)
            self._health_task = None

        async with self._lock:
            for message_id, session in list(self._pending_sessions.items()):
                if session.future.done():
                    continue

                # Resolve every in-flight RPC before tearing down worker channels so
                # callers never observe a hung session after host shutdown begins.
                session.future.set_result(
                    _failed_outcome_json(
                        status=DeliveryStatus.CANCELED,
                        message_id=message_id,
                        correlation_id=session.correlation_id,
                        message="Distributed runtime host stopped before RPC completion.",
                        retryable=True,
                    )
                )

            worker_records = list(self._workers.values())
            self._workers.clear()
            self._owners_by_agent_type.clear()
            self._pending_sessions.clear()

            for subscription in list(self._routing.subscriptions):
                self._routing.remove_subscription(subscription.id)

        for record in worker_records:
            await record.channel.close()

        if self._server is not None:
            await self._server.stop(grace=None)
            self._server = None

        self._is_running = False

    async def stop_when_idle(self) -> None:
        """Stop the host after pending RPC sessions complete."""
        if not self._is_running:
            return

        self._accepting_traffic = False
        while True:
            async with self._lock:
                # Host idleness currently tracks direct RPC sessions only.
                if not self._pending_sessions:
                    break
            await asyncio.sleep(0.01)
        await self.stop()

    async def register_worker(self, request: JsonObject) -> JsonObject:
        """Register one worker connection after a successful health check."""
        if not self._accepting_traffic:
            return {"ok": False, "error": "Host is not accepting new workers."}

        worker_address = request.get("address")
        if not isinstance(worker_address, str) or not worker_address:
            return {"ok": False, "error": "Worker registration requires an address."}

        channel = grpc_aio.insecure_channel(worker_address)
        client = WorkerServiceStub(channel)
        try:
            # Registration is rejected unless the worker can already answer traffic
            # at the advertised endpoint.
            health = await client.health_check({"ping": True})
        except grpc_aio.AioRpcError as exc:
            await channel.close()
            return {
                "ok": False,
                "error": f"Worker health check failed during registration: {exc}",
            }

        if health.get("ok") is not True:
            await channel.close()
            return {"ok": False, "error": "Worker failed readiness check."}

        worker_id = str(uuid4())
        async with self._lock:
            self._workers[worker_id] = _WorkerRecord(
                worker_id=worker_id,
                address=worker_address,
                channel=channel,
                client=client,
            )
        return {"ok": True, "worker_id": worker_id, "host_address": self.address}

    async def sync_catalog(self, request: JsonObject) -> JsonObject:
        """Replace one worker's advertised catalog snapshot.

        Catalog sync is full-state replacement, not an incremental patch. The host
        treats the incoming agent-type list as the worker's complete ownership
        claim and the incoming subscriptions as that worker's complete publish
        routing snapshot.
        """
        worker_id = request.get("worker_id")
        if not isinstance(worker_id, str) or not worker_id:
            return {"ok": False, "error": "Catalog sync requires a worker id."}

        agent_types = deserialize_agent_types(request.get("agent_types", []))
        subscriptions = deserialize_subscriptions(request.get("subscriptions", []))

        async with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return {"ok": False, "error": f"Unknown worker '{worker_id}'."}

            for agent_type in agent_types:
                owner = self._owners_by_agent_type.get(agent_type)
                if owner is not None and owner != worker_id:
                    return {
                        "ok": False,
                        "error": (
                            f"Agent type '{agent_type.value}' is already owned by "
                            f"worker '{owner}'."
                        ),
                    }

            for agent_type in record.agent_types:
                current_owner = self._owners_by_agent_type.get(agent_type)
                if current_owner == worker_id:
                    del self._owners_by_agent_type[agent_type]

            record.agent_types = set(agent_types)
            # Ownership is exclusive per agent type, so the new snapshot
            # replaces the worker's previous claim set atomically under the lock.
            for agent_type in record.agent_types:
                self._owners_by_agent_type[agent_type] = worker_id

            for subscription in record.subscriptions.values():
                try:
                    self._routing.remove_subscription(subscription.id)
                except LookupError:
                    continue

            # Publish routing also uses full replacement so stale subscriptions do
            # not survive a worker catalog update.
            record.subscriptions = {
                subscription.id: subscription for subscription in subscriptions
            }
            for subscription in subscriptions:
                self._routing.add_subscription(subscription)

        return {"ok": True}

    async def send_rpc(self, request: JsonObject) -> JsonObject:
        """Route one direct RPC through the host.

        The host owns the RPC session future separately from the outbound worker
        call so that shutdown, worker eviction, and transport failures can all
        resolve the same in-flight request consistently.
        """
        if not self._accepting_traffic:
            wire_envelope = WireEnvelope.from_json(request["envelope"])
            return _failed_outcome_json(
                status=DeliveryStatus.CANCELED,
                message_id=wire_envelope.message_id,
                correlation_id=wire_envelope.correlation_id,
                message="Host is not accepting new RPCs.",
                retryable=True,
            )

        wire_envelope = WireEnvelope.from_json(request["envelope"])
        if wire_envelope.recipient is None:
            return _failed_outcome_json(
                status=DeliveryStatus.UNDELIVERABLE,
                message_id=wire_envelope.message_id,
                correlation_id=wire_envelope.correlation_id,
                message="Direct RPC requires a concrete recipient.",
                retryable=False,
            )

        recipient = wire_envelope.recipient
        message_id = wire_envelope.message_id
        session_future: asyncio.Future[JsonObject] = (
            asyncio.get_running_loop().create_future()
        )

        async with self._lock:
            target_worker_id = self._owners_by_agent_type.get(recipient.type)
            # Record the session before the outbound worker call so shutdown and
            # worker-health transitions can still resolve the RPC consistently.
            self._pending_sessions[message_id] = _PendingRpcSession(
                target_worker_id=target_worker_id,
                correlation_id=wire_envelope.correlation_id,
                future=session_future,
            )
            target_record = (
                self._workers.get(target_worker_id)
                if target_worker_id is not None
                else None
            )

        if target_record is None:
            session_future.set_result(
                _failed_outcome_json(
                    status=DeliveryStatus.UNDELIVERABLE,
                    message_id=message_id,
                    correlation_id=wire_envelope.correlation_id,
                    message=(
                        f"No worker is registered for agent type "
                        f"'{recipient.type.value}'."
                    ),
                    retryable=False,
                )
            )
        else:
            try:
                # The outbound worker RPC runs outside the host lock so other host
                # requests can continue while this session is waiting.
                outcome = await asyncio.wait_for(
                    target_record.client.deliver_rpc(
                        {"envelope": wire_envelope.to_json()}
                    ),
                    timeout=self._rpc_timeout_seconds,
                )
            except (TimeoutError, grpc_aio.AioRpcError) as exc:
                await self._mark_worker_unhealthy(target_record.worker_id)
                outcome = _failed_outcome_json(
                    status=DeliveryStatus.TIMEOUT,
                    message_id=message_id,
                    correlation_id=wire_envelope.correlation_id,
                    message=f"Distributed RPC failed before completion: {exc}",
                    retryable=True,
                )
            # Shutdown or worker eviction may have resolved the session already.
            if not session_future.done():
                session_future.set_result(outcome)

        try:
            return await session_future
        finally:
            async with self._lock:
                self._pending_sessions.pop(message_id, None)

    async def publish(self, request: JsonObject) -> JsonObject:
        """Route one publish request through the host."""
        if not self._accepting_traffic:
            return {"ok": False, "error": "Host is not accepting new publishes."}

        wire_envelope = WireEnvelope.from_json(request["envelope"])
        publish_envelope = _metadata_envelope_from_wire(wire_envelope)
        routes = self._routing.resolve_publish_routes(publish_envelope)
        if publish_envelope.topic is None:
            return {"ok": False, "error": "Publish delivery requires a topic."}

        deliveries_by_worker: dict[str, list[JsonObject]] = {}
        async with self._lock:
            # Batch per worker so one publish fan-out only becomes one gRPC call
            # per destination worker, even when many recipients match the topic.
            for route in routes:
                target_worker_id = self._owners_by_agent_type.get(route.recipient.type)
                if target_worker_id is None:
                    return {
                        "ok": False,
                        "error": (
                            f"No worker is registered for published agent type "
                            f"'{route.recipient.type.value}'."
                        ),
                    }

                recipient = recipient_for_publish_route(
                    route=route,
                    publish_envelope=publish_envelope,
                )
                # Preserve the original payload bytes and metadata, but stamp a
                # per-recipient message id and concrete recipient before fan-out.
                recipient_envelope = MessageEnvelope.new_publish_event(
                    sender=publish_envelope.sender,
                    topic=publish_envelope.topic,
                    payload=publish_envelope.payload,
                    correlation_id=publish_envelope.correlation_id,
                    deadline_ms=publish_envelope.deadline_ms,
                    trace_id=publish_envelope.trace_id,
                    idempotency_key=publish_envelope.idempotency_key,
                )
                delivery: JsonObject = {
                    "envelope": {
                        **wire_envelope.to_json(),
                        "message_id": recipient_envelope.message_id.value,
                        "recipient": {
                            "type": recipient.type.value,
                            "key": recipient.key.value,
                        },
                    },
                }
                deliveries_by_worker.setdefault(target_worker_id, []).append(delivery)

            worker_records = {
                worker_id: self._workers[worker_id]
                for worker_id in deliveries_by_worker
                if worker_id in self._workers
            }

        total_enqueued = 0
        for worker_id, deliveries in deliveries_by_worker.items():
            record = worker_records.get(worker_id)
            if record is None:
                return {
                    "ok": False,
                    "error": f"Worker '{worker_id}' became unavailable during publish.",
                }
            try:
                response = await record.client.deliver_publish(
                    {"deliveries": deliveries}
                )
            except grpc_aio.AioRpcError as exc:
                await self._mark_worker_unhealthy(worker_id)
                return {
                    "ok": False,
                    "error": f"Publish fan-out failed for worker '{worker_id}': {exc}",
                }

            if response.get("ok") is not True:
                return {
                    "ok": False,
                    "error": str(response.get("error", "Worker rejected publish.")),
                }
            count = response.get("enqueued_recipient_count")
            if not isinstance(count, int):
                return {
                    "ok": False,
                    "error": "Worker returned invalid publish enqueue count.",
                }
            total_enqueued += count

        ack = PublishAck(
            message_id=MessageId(wire_envelope.message_id),
            correlation_id=(
                CorrelationId(wire_envelope.correlation_id)
                if wire_envelope.correlation_id is not None
                else None
            ),
            enqueued_recipient_count=total_enqueued,
            enqueued_at_ms=utc_now_ms(),
        )
        return {"ok": True, "ack": ack.to_json()}

    async def deregister_worker(self, request: JsonObject) -> JsonObject:
        """Remove one worker from the host."""
        worker_id = request.get("worker_id")
        if not isinstance(worker_id, str) or not worker_id:
            return {"ok": False, "error": "Deregistration requires a worker id."}

        await self._remove_worker(worker_id)
        return {"ok": True}

    async def _run_health_loop(self) -> None:
        """Poll every worker and evict workers that fail health checks repeatedly."""
        while self._is_running:
            await asyncio.sleep(self._health_check_interval_seconds)
            async with self._lock:
                # Snapshot ids first so health probes happen without holding the
                # main host state lock across network calls.
                worker_ids = list(self._workers)
            for worker_id in worker_ids:
                await self._check_worker_health(worker_id)

    async def _check_worker_health(self, worker_id: str) -> None:
        """Probe one worker and reset its failure counter when it responds."""
        async with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return

        try:
            response = await record.client.health_check({"ping": True})
        except grpc_aio.AioRpcError:
            await self._mark_worker_unhealthy(worker_id)
            return
        if response.get("ok") is not True:
            await self._mark_worker_unhealthy(worker_id)
            return

        async with self._lock:
            refreshed = self._workers.get(worker_id)
            if refreshed is not None:
                refreshed.failed_health_checks = 0

    async def _mark_worker_unhealthy(self, worker_id: str) -> None:
        """Advance health-failure tracking and remove the worker if needed."""
        async with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return
            record.failed_health_checks += 1
            # Require repeated failures before eviction so one transient transport
            # blip does not immediately reshuffle ownership.
            should_remove = (
                record.failed_health_checks >= self._health_check_failure_threshold
            )
        if should_remove:
            await self._remove_worker(worker_id)

    async def _remove_worker(self, worker_id: str) -> None:
        """Remove one worker and fail any RPCs that depended on it."""
        async with self._lock:
            record = self._workers.pop(worker_id, None)
            if record is None:
                return

            for agent_type in record.agent_types:
                current_owner = self._owners_by_agent_type.get(agent_type)
                if current_owner == worker_id:
                    del self._owners_by_agent_type[agent_type]

            for subscription in record.subscriptions.values():
                try:
                    self._routing.remove_subscription(subscription.id)
                except LookupError:
                    continue

            affected_sessions = [
                (message_id, session)
                for message_id, session in self._pending_sessions.items()
                if session.target_worker_id == worker_id
            ]

        for message_id, session in affected_sessions:
            if session.future.done():
                continue

            # Resolve outside the lock so callback chains on the session future do
            # not run while host state is still being mutated.
            session.future.set_result(
                _failed_outcome_json(
                    status=DeliveryStatus.UNDELIVERABLE,
                    message_id=message_id,
                    correlation_id=session.correlation_id,
                    message=(
                        f"Worker '{worker_id}' became unavailable before "
                        "RPC completion."
                    ),
                    retryable=True,
                )
            )

        await record.channel.close()
