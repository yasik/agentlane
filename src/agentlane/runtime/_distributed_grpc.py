"""Minimal gRPC transport glue for the distributed runtime.

The distributed runtime uses generic unary handlers plus JSON payloads instead of
generated protobuf services. That keeps the core package self-contained while still
using gRPC as the transport boundary between host and worker processes.
"""

import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, cast

import grpc
from grpc import aio as grpc_aio

type JsonObject = dict[str, object]
"""JSON-compatible object used by internal unary gRPC handlers."""

type UnaryServicerContext = grpc_aio.ServicerContext[Any, Any]
"""Concrete gRPC context shape for JSON unary handlers."""

type UnaryJsonMethod = Callable[
    [JsonObject, UnaryServicerContext], Awaitable[JsonObject]
]
"""Unary JSON handler signature exposed to gRPC."""

type UnaryJsonCall = Callable[[JsonObject], Awaitable[JsonObject]]
"""Unary JSON client call signature returned by gRPC channels."""

_HOST_SERVICE_NAME = "agentlane.runtime.host.HostService"
_WORKER_SERVICE_NAME = "agentlane.runtime.worker.WorkerService"


def _encode_json(value: JsonObject) -> bytes:
    """Encode a JSON-compatible mapping to UTF-8 bytes."""
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _decode_json(value: bytes) -> JsonObject:
    """Decode UTF-8 JSON bytes into a mapping."""
    decoded = json.loads(value.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise TypeError("Expected a JSON object for distributed runtime transport.")
    return cast(JsonObject, decoded)


class HostServiceHandler(Protocol):
    """Host-side unary gRPC handler contract."""

    async def register_worker(self, request: JsonObject) -> JsonObject:
        """Handle worker registration."""
        ...

    async def sync_catalog(self, request: JsonObject) -> JsonObject:
        """Handle worker catalog synchronization."""
        ...

    async def send_rpc(self, request: JsonObject) -> JsonObject:
        """Handle worker-originated direct RPC submission."""
        ...

    async def publish(self, request: JsonObject) -> JsonObject:
        """Handle worker-originated publish submission."""
        ...

    async def deregister_worker(self, request: JsonObject) -> JsonObject:
        """Handle worker deregistration."""
        ...


class WorkerServiceHandler(Protocol):
    """Worker-side unary gRPC handler contract."""

    async def deliver_rpc(self, request: JsonObject) -> JsonObject:
        """Handle host-originated direct delivery."""
        ...

    async def deliver_publish(self, request: JsonObject) -> JsonObject:
        """Handle host-originated publish fan-out delivery."""
        ...

    async def health_check(self, request: JsonObject) -> JsonObject:
        """Handle worker health checks."""
        ...


def create_host_generic_handler(
    handler: HostServiceHandler,
) -> grpc.GenericRpcHandler:
    """Build the host-side generic gRPC service definition."""

    async def register_worker(
        request: JsonObject, context: UnaryServicerContext
    ) -> JsonObject:
        del context
        return await handler.register_worker(request)

    async def sync_catalog(
        request: JsonObject, context: UnaryServicerContext
    ) -> JsonObject:
        del context
        return await handler.sync_catalog(request)

    async def send_rpc(
        request: JsonObject, context: UnaryServicerContext
    ) -> JsonObject:
        del context
        return await handler.send_rpc(request)

    async def publish(request: JsonObject, context: UnaryServicerContext) -> JsonObject:
        del context
        return await handler.publish(request)

    async def deregister_worker(
        request: JsonObject,
        context: UnaryServicerContext,
    ) -> JsonObject:
        del context
        return await handler.deregister_worker(request)

    return grpc.method_handlers_generic_handler(
        _HOST_SERVICE_NAME,
        {
            "RegisterWorker": grpc.unary_unary_rpc_method_handler(
                register_worker,
                request_deserializer=_decode_json,
                response_serializer=_encode_json,
            ),
            "SyncCatalog": grpc.unary_unary_rpc_method_handler(
                sync_catalog,
                request_deserializer=_decode_json,
                response_serializer=_encode_json,
            ),
            "SendRpc": grpc.unary_unary_rpc_method_handler(
                send_rpc,
                request_deserializer=_decode_json,
                response_serializer=_encode_json,
            ),
            "Publish": grpc.unary_unary_rpc_method_handler(
                publish,
                request_deserializer=_decode_json,
                response_serializer=_encode_json,
            ),
            "DeregisterWorker": grpc.unary_unary_rpc_method_handler(
                deregister_worker,
                request_deserializer=_decode_json,
                response_serializer=_encode_json,
            ),
        },
    )


def create_worker_generic_handler(
    handler: WorkerServiceHandler,
) -> grpc.GenericRpcHandler:
    """Build the worker-side generic gRPC service definition."""

    async def deliver_rpc(
        request: JsonObject, context: UnaryServicerContext
    ) -> JsonObject:
        del context
        return await handler.deliver_rpc(request)

    async def deliver_publish(
        request: JsonObject,
        context: UnaryServicerContext,
    ) -> JsonObject:
        del context
        return await handler.deliver_publish(request)

    async def health_check(
        request: JsonObject,
        context: UnaryServicerContext,
    ) -> JsonObject:
        del context
        return await handler.health_check(request)

    return grpc.method_handlers_generic_handler(
        _WORKER_SERVICE_NAME,
        {
            "DeliverRpc": grpc.unary_unary_rpc_method_handler(
                deliver_rpc,
                request_deserializer=_decode_json,
                response_serializer=_encode_json,
            ),
            "DeliverPublish": grpc.unary_unary_rpc_method_handler(
                deliver_publish,
                request_deserializer=_decode_json,
                response_serializer=_encode_json,
            ),
            "HealthCheck": grpc.unary_unary_rpc_method_handler(
                health_check,
                request_deserializer=_decode_json,
                response_serializer=_encode_json,
            ),
        },
    )


class HostServiceStub:
    """Small hand-written unary stub for worker -> host calls."""

    def __init__(self, channel: grpc_aio.Channel) -> None:
        self._register_worker: UnaryJsonCall = cast(
            UnaryJsonCall,
            channel.unary_unary(
                f"/{_HOST_SERVICE_NAME}/RegisterWorker",
                request_serializer=_encode_json,
                response_deserializer=_decode_json,
            ),
        )
        self._sync_catalog: UnaryJsonCall = cast(
            UnaryJsonCall,
            channel.unary_unary(
                f"/{_HOST_SERVICE_NAME}/SyncCatalog",
                request_serializer=_encode_json,
                response_deserializer=_decode_json,
            ),
        )
        self._send_rpc: UnaryJsonCall = cast(
            UnaryJsonCall,
            channel.unary_unary(
                f"/{_HOST_SERVICE_NAME}/SendRpc",
                request_serializer=_encode_json,
                response_deserializer=_decode_json,
            ),
        )
        self._publish: UnaryJsonCall = cast(
            UnaryJsonCall,
            channel.unary_unary(
                f"/{_HOST_SERVICE_NAME}/Publish",
                request_serializer=_encode_json,
                response_deserializer=_decode_json,
            ),
        )
        self._deregister_worker: UnaryJsonCall = cast(
            UnaryJsonCall,
            channel.unary_unary(
                f"/{_HOST_SERVICE_NAME}/DeregisterWorker",
                request_serializer=_encode_json,
                response_deserializer=_decode_json,
            ),
        )

    async def register_worker(self, request: JsonObject) -> JsonObject:
        """Register one worker with the host."""
        return await self._register_worker(request)

    async def sync_catalog(self, request: JsonObject) -> JsonObject:
        """Sync one worker catalog snapshot with the host."""
        return await self._sync_catalog(request)

    async def send_rpc(self, request: JsonObject) -> JsonObject:
        """Submit one direct RPC to the host."""
        return await self._send_rpc(request)

    async def publish(self, request: JsonObject) -> JsonObject:
        """Submit one publish request to the host."""
        return await self._publish(request)

    async def deregister_worker(self, request: JsonObject) -> JsonObject:
        """Deregister one worker from the host."""
        return await self._deregister_worker(request)


class WorkerServiceStub:
    """Small hand-written unary stub for host -> worker calls."""

    def __init__(self, channel: grpc_aio.Channel) -> None:
        self._deliver_rpc: UnaryJsonCall = cast(
            UnaryJsonCall,
            channel.unary_unary(
                f"/{_WORKER_SERVICE_NAME}/DeliverRpc",
                request_serializer=_encode_json,
                response_deserializer=_decode_json,
            ),
        )
        self._deliver_publish: UnaryJsonCall = cast(
            UnaryJsonCall,
            channel.unary_unary(
                f"/{_WORKER_SERVICE_NAME}/DeliverPublish",
                request_serializer=_encode_json,
                response_deserializer=_decode_json,
            ),
        )
        self._health_check: UnaryJsonCall = cast(
            UnaryJsonCall,
            channel.unary_unary(
                f"/{_WORKER_SERVICE_NAME}/HealthCheck",
                request_serializer=_encode_json,
                response_deserializer=_decode_json,
            ),
        )

    async def deliver_rpc(self, request: JsonObject) -> JsonObject:
        """Deliver one direct RPC to a worker."""
        return await self._deliver_rpc(request)

    async def deliver_publish(self, request: JsonObject) -> JsonObject:
        """Deliver one publish batch to a worker."""
        return await self._deliver_publish(request)

    async def health_check(self, request: JsonObject) -> JsonObject:
        """Run one health check against a worker."""
        return await self._health_check(request)
