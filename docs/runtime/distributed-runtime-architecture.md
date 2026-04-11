# Runtime: Distributed Runtime Architecture

The distributed runtime keeps the same delivery model as the in-process runtime,
but separates routing from execution. A host knows which worker owns which
agent types. A worker executes the deliveries it receives. The same
[`MessageEnvelope`](../../src/agentlane/messaging/_envelope.py) model still
represents the work moving through the system.

In concrete terms, [`WorkerAgentRuntimeHost`](../../src/agentlane/runtime/_worker_runtime_host.py)
owns routing, [`WorkerAgentRuntime`](../../src/agentlane/runtime/_worker_runtime.py)
owns execution on each worker, and
[`DistributedRuntimeEngine`](../../src/agentlane/runtime/_worker_runtime.py) is
the convenience entrypoint that makes the whole arrangement still look like one
runtime from the caller's point of view.

## One Host, Many Workers

It helps to think about the system in terms of responsibilities rather than
types:

1. the host keeps placement, subscriptions, and pending RPC sessions
2. workers own local registries, schedulers, and agent execution
3. callers still talk to a runtime API rather than to workers directly

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

The architectural boundary is simple: the host decides where work goes, and the
worker decides how that work runs locally.

## Direct Send

For `send_message(...)`, the flow is:

1. the origin worker builds and serializes the request
2. the host records a pending RPC session
3. the host resolves the target worker from the recipient type
4. the host forwards the request to that worker
5. the destination worker runs the delivery through its normal local runtime
6. the resulting [`DeliveryOutcome`](../../src/agentlane/messaging/_outcome.py)
   is returned to the origin

The important point is that distributed send does not invent a new execution
model. It adds routing and transport around the local runtime path.

## Publish

For `publish_message(...)`, the host does more work up front:

1. the origin worker sends the publish request to the host
2. the host resolves all matching subscriptions
3. each route is rewritten into a concrete recipient
4. the concrete deliveries are grouped by destination worker
5. each worker receives only the deliveries it needs to enqueue

That means global publish matching happens once, at the host. Destination
workers do not repeat the full subscription lookup.

## Placement

The current placement rule is intentionally simple: one worker owns a given
`AgentType`.

This keeps routing predictable, but it also means keys of the same type are not
yet sharded across workers. Different workers may host different agent types.
One worker may also host many concrete `AgentId` values for the same type.

## Lifecycle Responsibilities

The host lifecycle is mostly about accepting or rejecting new cross-worker
traffic. The worker lifecycle is mostly about registering its local catalog,
keeping a connection to the host, and stopping new remote deliveries before
tearing down local resources.

That is why the shutdown order matters:

1. stop routing new traffic first
2. drain or fail pending work
3. close transport resources last

## Current Boundaries

The current implementation keeps the distributed surface small on purpose:

1. one managed host plus one primary worker in the convenience runtime
2. one owner per `AgentType`
3. unary RPC-style transport only
4. host idleness based on pending RPC sessions rather than full publish fan-out

Those boundaries are useful to know because they explain both the current
feature set and the current limitations.

## Related Docs

1. [Runtime: Engine and Execution](./engine-and-execution.md)
2. [Runtime: Distributed Runtime Usage](./distributed-runtime-usage.md)
