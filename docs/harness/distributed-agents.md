# Distributed Agents

AgentLane lets a high-level harness agent coordinate distributed workers without
giving up the simple `DefaultAgent` developer surface. The same application can
start in one Python process, then move specialist workers into separate
processes while preserving the same routing, messaging, and agent identities.

This is useful when an AI workflow needs to be inspectable and operable. A
clinical inbox assistant, for example, may need one model-backed copilot, several
deterministic or model-backed specialist agents, explicit fan-out and fan-in,
and a clear place for human review. AgentLane treats those pieces as an
application workflow instead of hiding them inside one prompt loop.

That emphasis comes from the kind of systems AgentLane was built for: clinical
AI pipelines where the workflow has to be composable, debuggable, and safe to
operate with humans in the loop.

The full runnable version of this pattern lives in
[`examples/harness/distributed_clinical_inbox_copilot`](../../examples/harness/distributed_clinical_inbox_copilot/).
This page explains the key building blocks behind that example.

## Why This Matters

Many agent systems begin as a prompt, a tool list, and a loop. That can be enough
for local demos. Production workflows often need stronger boundaries:

1. which agent received a task
2. which worker executed it
3. what state was local to that worker
4. how specialist work was fanned out
5. how results were gathered
6. how the same workflow can move from local development to distributed
   execution

AgentLane makes those boundaries explicit. The harness gives you model loops,
tools, streaming, and run state. The runtime gives you addressed delivery,
publish/subscribe, worker placement, and lifecycle control. Distributed agents
are what you get when those layers are used together.

## The Mental Model

A distributed harness workflow usually has two kinds of agents:

1. a top-level harness agent, often a `DefaultAgent`, that owns the user-facing
   model loop
2. worker agents, usually `BaseAgent` subclasses, that receive messages through
   the runtime and do focused work

The top-level agent can still expose normal harness tools. Those tools can use
runtime messaging to publish work to specialists, wait for a result receiver,
and return a compact tool result back into the model loop.

```text
DefaultAgent.run_stream(...)
        |
        v
top-level tool: launch_parallel_review(...)
        |
        v
WorkerAgentRuntime.publish_message(...)
        |
        v
WorkerAgentRuntimeHost
        |
        v
specialist workers -> result topic -> aggregator worker -> controller result receiver
```

The important point is that the model-facing agent and the distributed workers
use the same runtime concepts. Moving a worker to another process changes
deployment, not the workflow contract.

## Core Building Blocks

### `DefaultAgent(runtime=...)`

`DefaultAgent` is the small harness surface for model-backed agents. When you
pass a runtime explicitly, the agent uses that runtime instead of provisioning a
local default runtime.

```python
agent = DefaultAgent(
    descriptor=descriptor,
    runtime=copilot_worker,
)

stream = await agent.run_stream(user_prompt)
```

That is the bridge between the high-level harness and the distributed runtime.
The harness still owns the model loop. The supplied runtime owns delivery,
agent instance reuse, and worker routing.

### `WorkerAgentRuntimeHost`

The host is the distributed control plane. It tracks workers, agent-type
ownership, subscriptions, and pending direct RPC sessions.

```python
host = WorkerAgentRuntimeHost(address="127.0.0.1:0")
await host.start()
```

Binding to `127.0.0.1:0` is useful for examples and tests because the OS selects
an available local port.

### `WorkerAgentRuntime`

Each worker runtime owns local execution. It has its own registry, scheduler,
agent instances, serializer registry, and gRPC listener. It connects to a host
and advertises the agent types and subscriptions it owns.

```python
copilot_worker = WorkerAgentRuntime(host_address=host.address)
specialist_worker = WorkerAgentRuntime(host_address=host.address)
```

You can create several worker runtimes in one Python process. You can also run
the same worker setup in separate Python processes.

### Message Types

Distributed workers exchange serialized messages. Put message dataclasses in a
shared importable module and register the same message types on every runtime
that sends or receives them.

```python
@dataclass(slots=True)
class ClinicalReviewTask:
    session_id: str
    review_id: str
    patient_message: str
    chart_snapshot: str


@dataclass(slots=True)
class SpecialistFinding:
    session_id: str
    review_id: str
    agent_name: str
    headline: str
    detail: str
    urgent_flag: bool


def register_message_types(runtime: WorkerAgentRuntime) -> None:
    runtime.register_message_type(ClinicalReviewTask)
    runtime.register_message_type(SpecialistFinding)
```

This matters more in multi-process mode. A class defined in a script's
`__main__` module will not have the same import identity in a subprocess. A
shared module keeps serializer registration stable.

### Worker Agents

Worker agents are normal runtime agents. They receive a typed payload, do their
work, and publish or return results.

```python
class MedicationSafetyAgent(BaseAgent):
    @on_message
    async def handle(
        self,
        payload: ClinicalReviewTask,
        context: MessageContext,
    ) -> object:
        finding = SpecialistFinding(
            session_id=payload.session_id,
            review_id=payload.review_id,
            agent_name="med-safety-agent",
            headline="Medication safety",
            detail="Review found a likely medication-related safety signal.",
            urgent_flag=True,
        )
        await self.publish_message(
            finding,
            topic=TopicId.from_values(
                type_value="clinical.review_result",
                route_key=payload.review_id,
            ),
            correlation_id=context.correlation_id,
        )
        return finding
```

The specialist does not need to know who requested the review or where the
aggregator lives. It publishes the result to a topic. The host resolves the
matching subscribers.

### Subscriptions And Delivery Modes

Publish/subscribe is the natural fit for fan-out and fan-in workflows.

```python
specialist_worker.register_factory(
    "clinical.med_safety",
    MedicationSafetyAgent,
)
specialist_worker.subscribe_exact(
    topic_type="clinical.review_requested",
    agent_type="clinical.med_safety",
    delivery_mode=DeliveryMode.STATELESS,
)
```

`STATELESS` is useful for specialists that handle each published request
independently. `STATEFUL` is useful for aggregators keyed by a route key such as
`review_id`.

```python
aggregator_worker.subscribe_exact(
    topic_type="clinical.review_result",
    agent_type="clinical.aggregator",
    delivery_mode=DeliveryMode.STATEFUL,
)
```

With stateful delivery, the runtime creates or reuses the concrete subscriber
agent using the topic route key. That gives each review its own aggregation
state without adding a separate state store for the simple case.

## Single-Process Distributed Agents

Single-process distributed mode is often the easiest place to start. The host
and several workers run inside one interpreter, while the workflow already uses
distributed routing and explicit worker ownership.

```python
host = WorkerAgentRuntimeHost(address="127.0.0.1:0")
await host.start()

copilot_worker = WorkerAgentRuntime(host_address=host.address)
med_safety_worker = WorkerAgentRuntime(host_address=host.address)
aggregator_worker = WorkerAgentRuntime(host_address=host.address)

for worker in (copilot_worker, med_safety_worker, aggregator_worker):
    register_message_types(worker)

med_safety_worker.register_factory("clinical.med_safety", MedicationSafetyAgent)
med_safety_worker.subscribe_exact(
    topic_type="clinical.review_requested",
    agent_type="clinical.med_safety",
    delivery_mode=DeliveryMode.STATELESS,
)

aggregator_worker.register_factory("clinical.aggregator", ReviewAggregatorAgent)
aggregator_worker.subscribe_exact(
    topic_type="clinical.review_result",
    agent_type="clinical.aggregator",
    delivery_mode=DeliveryMode.STATEFUL,
)

await asyncio.gather(
    copilot_worker.start(),
    med_safety_worker.start(),
    aggregator_worker.start(),
)
```

At this point, the top-level harness agent can run on `copilot_worker`:

```python
agent = DefaultAgent(
    descriptor=descriptor,
    runtime=copilot_worker,
)
```

A tool exposed to that agent can publish a request through the same worker:

```python
@as_tool
async def launch_parallel_review(patient_message: str) -> str:
    request = ClinicalReviewTask(
        session_id=session_id,
        review_id=review_id,
        patient_message=patient_message,
        chart_snapshot=chart_snapshot,
    )
    ack = await copilot_worker.publish_message(
        request,
        topic=TopicId.from_values(
            type_value="clinical.review_requested",
            route_key=review_id,
        ),
    )
    if ack.enqueued_recipient_count != expected_specialists:
        raise RuntimeError("Review fan-out did not reach every specialist.")

    review = await review_tracker.wait_for_result(review_id)
    return format_review_for_model(review)
```

The model sees one ordinary tool result. The application gets an explicit
distributed workflow behind that tool.

## Multi-Process Distributed Agents

Multi-process mode keeps the same runtime shape and moves worker startup into a
separate entrypoint. The controller process starts the host and the top-level
copilot worker. Each specialist process creates its own `WorkerAgentRuntime`,
registers the same message types, registers one role, and connects to the
controller's host.

```python
async def run_worker(args: argparse.Namespace) -> None:
    worker = WorkerAgentRuntime(
        host_address=args.host_address,
        address=args.bind_address,
    )
    register_message_types(worker)
    register_worker_role(worker, args.role)

    await worker.start()
    print(
        json.dumps(
            {
                "type": "worker_ready",
                "role": args.role,
                "address": worker.address,
                "worker_id": worker.worker_id,
                "pid": os.getpid(),
            }
        ),
        flush=True,
    )
    await stop_event.wait()
    await worker.stop_when_idle()
```

The controller can launch workers with normal subprocess tools:

```python
process = await asyncio.create_subprocess_exec(
    sys.executable,
    "-m",
    "examples.harness.distributed_clinical_inbox_copilot.worker",
    "--role",
    "med-safety",
    "--host-address",
    host.address,
    "--bind-address",
    "127.0.0.1:0",
    stdout=asyncio.subprocess.PIPE,
)

ready = json.loads(await process.stdout.readline())
```

The readiness record is application-level metadata. AgentLane already knows the
worker through host registration. The record is useful for logs, dashboards,
tests, and operator visibility.

## Completion Is A Message Too

In single-process examples, it is tempting for an aggregator to complete an
in-memory future directly. That shortcut does not cross process boundaries.

The distributed pattern is to send the aggregate result back through the runtime:

```python
class ReviewAggregatorAgent(BaseAgent):
    @on_message
    async def handle(
        self,
        payload: SpecialistFinding,
        context: MessageContext,
    ) -> object:
        self._findings[payload.agent_name] = payload
        if len(self._findings) < expected_finding_count:
            return None

        review = AggregatedClinicalReview(
            session_id=payload.session_id,
            review_id=self.id.key.value,
            findings=serialize_findings(self._findings),
            urgent_flag=any(item.urgent_flag for item in self._findings.values()),
        )
        await self.publish_message(
            review,
            topic=TopicId.from_values(
                type_value="clinical.review_completed",
                route_key=payload.session_id,
            ),
            correlation_id=context.correlation_id,
        )
        return None
```

The controller owns a small result receiver agent that subscribes to
`clinical.review_completed` and resolves a local waiter. This keeps the boundary
clean: workers communicate by messages, and controller-local state stays in the
controller.

## Architecture Choices To Notice

### Runtime First

The distributed workflow is built from runtime primitives before the model loop
is involved. That is why a model-free smoke test can exercise fan-out, fan-in,
serialization, worker registration, and shutdown without calling a model.

### Logical Agent Identity

`AgentId` is the stable logical identity. A worker address is a network endpoint.
The host maps logical agent types to worker endpoints. Application code can
display both, but they represent different concerns.

### Host Routes, Workers Execute

The host owns placement and subscription matching. Workers own local execution,
agent state, and scheduling. This is the same boundary in single-process and
multi-process deployments.

### Serialization Is Part Of The Design

Cross-worker messages should be treated as serialized payloads. Prefer explicit
dataclasses for messages and JSON-shaped fields for nested aggregate data when
that data must cross process boundaries.

### Publish Acknowledgement Means Enqueued

`PublishAck.enqueued_recipient_count` tells you how many concrete deliveries were
enqueued. It does not mean every handler has completed. Use a result topic,
direct response, or workflow-specific completion message for completion.

### Streaming Stays With The Harness

`DefaultAgent.run_stream(...)` streams the top-level model run locally through
the harness. Distributed workers can still run behind tools and messages. The
current distributed runtime returns message outcomes rather than per-token
transport streams.

## When To Use This Pattern

Use distributed harness agents when a workflow has a top-level model agent and
one or more independently useful worker agents:

1. clinical triage with safety, guideline, chart, and communication specialists
2. financial review with market, risk, compliance, and summary workers
3. operations workflows with planner, executor, verifier, and escalation agents
4. any human-in-the-loop workflow where the system should show what work was
   delegated and where the result came from

Start with one process when you are designing the workflow. Move workers into
separate processes when placement, isolation, lifecycle, or scaling matters.
AgentLane keeps the communication model stable across both shapes.

## Related Reading

1. [Harness Architecture](./architecture.md)
2. [Harness Default Agents](./default-agents.md)
3. [Harness Runner](./runner.md)
4. [Runtime: Distributed Runtime Usage](../runtime/distributed-runtime-usage.md)
5. [Runtime: Distributed Runtime Architecture](../runtime/distributed-runtime-architecture.md)
6. [Distributed Clinical Inbox Copilot Example](../../examples/harness/distributed_clinical_inbox_copilot/)
