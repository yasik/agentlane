# Runtime: Distributed Runtime Architecture

This document explains the current distributed runtime architecture in core v1:
what runs where, how messages flow, which responsibilities belong to the host
versus workers, and which simplifying decisions shape the implementation.

See also:

1. [Runtime: Engine and Execution](./engine-and-execution.md)
2. [Runtime: Distributed Runtime Usage](./distributed-runtime-usage.md)

## TL;DR

1. `DistributedRuntimeEngine` is the zero-config distributed entrypoint.
2. It manages one in-process `WorkerAgentRuntimeHost` plus one primary
   `WorkerAgentRuntime`.
3. `WorkerAgentRuntimeHost` is the routing and session service, not an agent
   execution runtime.
4. `WorkerAgentRuntime` executes agents locally and forwards distributed traffic
   through the host over gRPC.
5. v1 placement is intentionally simple: one worker owns a given `AgentType`.
6. Direct RPC returns terminal `DeliveryOutcome`; publish returns enqueue-only
   `PublishAck`.

## Topology

In distributed mode, caller-visible messaging still starts from a worker
runtime. The difference is that cross-worker routing moves through a host.

```text
caller / agent code
        |
        v
+-----------------------+
| WorkerAgentRuntime    |
| - public Engine API   |
| - local execution     |
| - host client         |
+-----------+-----------+
            |
            v
+-----------------------+
| WorkerAgentRuntimeHost|
| - worker directory    |
| - agent ownership     |
| - subscription index  |
| - RPC session table   |
+-----------+-----------+
            |
            v
+-----------------------+
| WorkerAgentRuntime    |
| - local execution     |
| - worker gRPC server  |
+-----------+-----------+
            |
            v
+-----------------------+
| AgentRegistry         |
| Dispatcher            |
| PerAgentMailboxSched. |
+-----------------------+
```

The important architectural boundary is that the host decides where work goes,
while workers decide how that work is executed locally.

## Core Components

### `DistributedRuntimeEngine`

`DistributedRuntimeEngine` is the convenience wrapper used by
`distributed_runtime()`.

Current behavior:

1. Start one in-process `WorkerAgentRuntimeHost`.
2. Start one primary `WorkerAgentRuntime`.
3. Expose the normal `RuntimeEngine` API from that worker.
4. Manage worker and host lifecycle together.

This keeps the common case zero-config without hiding the underlying host and
worker model.

### `WorkerAgentRuntimeHost`

The host is the control plane and routing plane.

Responsibilities:

1. Accept worker registrations.
2. Keep the live worker directory and gRPC channels.
3. Track exclusive `AgentType -> worker` ownership.
4. Mirror worker subscriptions for publish routing.
5. Keep pending direct RPC sessions until they resolve.
6. Health-check workers and remove unhealthy ones from future routing.

The host does not execute user agents.

### `WorkerAgentRuntime`

The worker is the execution node.

Responsibilities:

1. Expose `send_message` and `publish_message` to local callers and agents.
2. Own the local agent registry, scheduler, dispatcher, and serializer
   registry.
3. Start a worker gRPC server so the host can deliver work to it.
4. Register its address and current catalog with the host.
5. Execute inbound work locally once the host has chosen this worker.

### Shared Local Execution Path

Distributed workers deliberately reuse the same local execution machinery as the
in-process runtime:

1. `AgentRegistry`
2. `Dispatcher`
3. `PerAgentMailboxScheduler`

This keeps distributed mode as a transport and placement layer over the same
local execution semantics rather than creating a second execution model.

## Routing Model

### Direct RPC

For `send_message(...)`, the current flow is:

1. The origin worker builds a direct `MessageEnvelope`.
2. The worker serializes the payload and submits the RPC to the host.
3. The host creates a pending session keyed by message id.
4. The host resolves the destination worker from `recipient.type`.
5. The host forwards the request to that worker with `DeliverRpc`.
6. The destination worker submits the message through its normal local runtime
   path.
7. The destination worker returns a `DeliveryOutcome`.
8. The host resolves the pending session and returns the outcome to the origin
   worker.

If the target worker does not exist, disconnects, or times out, the host still
completes the RPC with a terminal failure outcome.

### Publish

For `publish_message(...)`, the current flow is:

1. The origin worker builds one publish `MessageEnvelope`.
2. The worker serializes the payload and submits the publish request to the
   host.
3. The host resolves matching routes from its mirrored global subscription
   index.
4. The host rewrites each route into a concrete recipient id.
5. The host batches those concrete deliveries by destination worker.
6. The host sends one `DeliverPublish` request per destination worker.
7. Each destination worker enqueues the concrete deliveries it receives.
8. The host sums enqueue counts and returns one `PublishAck`.

The destination worker does not rerun global publish routing. That decision has
already been made by the host.

## Placement and Catalog Sync

The current placement rule is intentionally narrow:

1. One worker owns a given `AgentType`.
2. Direct RPC routes by `recipient.type`.
3. A worker may host many concrete `AgentId` values for the same type.
4. Different keys of the same `AgentType` are not sharded across workers yet.

Workers advertise placement and publish-routing state through full catalog
snapshots.

Catalog sync properties:

1. A worker syncs agent types and subscriptions as a full-state replacement.
2. Registering factories or subscriptions before `start()` is fine because the
   worker pushes a full snapshot during startup.
3. Later local catalog changes are coalesced and resynced to the host.
4. Removing a subscription or agent type removes it from the host view on the
   next sync.

## Transport Boundary

The base transport is gRPC with generic unary request and response payloads.

Current host RPC surface:

1. `RegisterWorker`
2. `SyncCatalog`
3. `SendRpc`
4. `Publish`
5. `DeregisterWorker`

Current worker RPC surface:

1. `DeliverRpc`
2. `DeliverPublish`
3. `HealthCheck`

The transport stays intentionally small:

1. unary calls only
2. JSON-safe request and response payloads
3. dedicated wire models only where the transport shape differs from the
   canonical runtime shape

## Lifecycle

### Host Lifecycle

`WorkerAgentRuntimeHost.start()`:

1. Bind the gRPC server.
2. Resolve the final bound address when `:0` is used.
3. Begin accepting worker traffic.
4. Start the background health loop.

`WorkerAgentRuntimeHost.stop()`:

1. Stop accepting new traffic.
2. Resolve pending direct RPC sessions with terminal failure.
3. Close worker channels.
4. Stop the gRPC server.

`WorkerAgentRuntimeHost.stop_when_idle()`:

1. Stop accepting new traffic.
2. Wait for pending direct RPC sessions to drain.
3. Stop the host.

Current limitation:

1. Host idleness tracks direct RPC sessions only, not in-flight publish fan-out.

### Worker Lifecycle

`WorkerAgentRuntime.start()`:

1. Start local runtime resources.
2. Bind the worker gRPC server.
3. Connect to the host.
4. Register with the host using the resolved worker address.
5. Pass the host's registration-time health check.
6. Push the current catalog snapshot.

`WorkerAgentRuntime.stop()`:

1. Deregister from the host first so new cross-worker traffic stops.
2. Tear down transport resources.
3. Stop local runtime resources.

`WorkerAgentRuntime.stop_when_idle()`:

1. Deregister from the host first.
2. Drain local queued work.
3. Close transport resources.

## Concurrency and Failure Handling

The host is asynchronous and multiplexed. One long-running distributed RPC does
not block the entire host.

Current behavior:

1. The host records session state under a lock, then awaits worker delivery
   outside that lock.
2. While one RPC is waiting, the host can still route other RPCs, route
   publishes, and continue health checks.
3. Per-recipient ordering is still enforced by the destination worker's local
   mailbox scheduler.
4. Effective concurrency in distributed mode is per worker runtime: each worker
   applies its own local `worker_count`, and adding workers adds more execution
   capacity.

Failure handling rules:

1. Unknown direct targets return terminal failure, never silent drop.
2. Worker disconnects or timeouts resolve in-flight direct RPCs with terminal
   failure.
3. Unhealthy workers are removed from future routing after repeated failed
   health checks.
4. Publish remains best-effort and non-transactional. If one worker fails after
   another already enqueued locally, there is no rollback.

## Key Decisions

These are the main choices that define the current architecture:

1. Keep the host simple: routing, placement, health, and session ownership live
   there; agent execution does not.
2. Reuse the local runtime instead of creating a second distributed execution
   model.
3. Send all distributed direct RPC and publish traffic through the host so
   placement and routing stay centralized.
4. Use exclusive `AgentType -> worker` ownership as the deliberately simple
   placement rule.
5. Mirror subscriptions to the host and perform publish route resolution there.
6. Keep zero-config focused on topology only: managed host plus one primary
   worker.
7. Keep the transport minimal: gRPC, unary calls, JSON-safe payloads, and a
   small dedicated wire layer only where transport shape differs.

## Current Limitations

These are current implementation facts:

1. `distributed_runtime()` manages one in-process host plus one primary worker.
2. Additional workers must be created explicitly with `WorkerAgentRuntime`.
3. Placement is exclusive per `AgentType`.
4. `cancellation_token` is part of the API surface but is not propagated across
   the process boundary yet.
5. `PublishAck` confirms enqueue only, never downstream handler completion.

## Future Extensions

Likely future extension seams include:

1. richer placement strategies beyond `AgentType -> worker`
2. automatic worker provisioning from explicit bootstrap entrypoints
3. alternate worker directory or session stores
4. transport evolution beyond the current generic unary gRPC surface
