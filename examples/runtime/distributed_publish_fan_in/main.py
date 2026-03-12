"""Distributed publish fan-out / fan-in demo with a stateful aggregator."""

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentlane.messaging import (
    AgentId,
    DeliveryMode,
    DeliveryStatus,
    MessageContext,
    TopicId,
)
from agentlane.runtime import (
    BaseAgent,
    Engine,
    WorkerAgentRuntime,
    WorkerAgentRuntimeHost,
    on_message,
)

INGRESS_AGENT_TYPE = "demo.dist.ingress"
PLANNER_AGENT_TYPE = "demo.dist.planner"
INVENTORY_AGENT_TYPE = "demo.dist.inventory"
PRICING_AGENT_TYPE = "demo.dist.pricing"
AGGREGATOR_AGENT_TYPE = "demo.dist.aggregator"

PLAN_TOPIC_TYPE = "demo.dist.plan_ready"
RESULT_TOPIC_TYPE = "demo.dist.worker_result"

CONSOLE = Console()
COMPONENT_STYLES = {
    "host": "bold cyan",
    "ingress": "bold green",
    "planner": "bold magenta",
    "inventory": "bold blue",
    "pricing": "bold yellow",
    "aggregator": "bold white",
}


def log_line(component: str, message: str) -> None:
    """Print one timestamped demo log line."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    component_style = COMPONENT_STYLES.get(component, "white")
    CONSOLE.print(
        f"[dim][{timestamp}][/dim] "
        f"[{component_style}]{component:<18}[/{component_style}] "
        f"[dim]|[/dim] {message}"
    )


@dataclass(slots=True, frozen=True)
class DemoConfig:
    """Configuration for the distributed publish fan-in demo."""

    workflow_count: int
    """Number of workflows to run in parallel."""

    timeout_seconds: float
    """Timeout for waiting on aggregated workflow completion."""


@dataclass(slots=True)
class WorkflowSummary:
    """Final aggregated result for one workflow."""

    workflow_id: str
    """Unique workflow identifier."""

    merged_output: str
    """Deterministic merged output from all worker results."""

    source_count: int
    """Number of specialist results included in the merge."""


def expect_str(payload: dict[str, object], field_name: str) -> str:
    """Return one required string field from a JSON-like payload."""
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise TypeError(f"Expected string field '{field_name}'.")
    return value


class CompletionTracker:
    """Tracks workflow completion futures resolved by the aggregator agent."""

    def __init__(self) -> None:
        """Initialize completion tracker state."""
        self._futures: dict[str, asyncio.Future[WorkflowSummary]] = {}

    def register_workflow(self, workflow_id: str) -> None:
        """Register one workflow id with an incomplete future."""
        if workflow_id in self._futures:
            return
        self._futures[workflow_id] = asyncio.get_running_loop().create_future()

    def complete(self, summary: WorkflowSummary) -> None:
        """Resolve one workflow completion future."""
        self.register_workflow(summary.workflow_id)
        completion_future = self._futures[summary.workflow_id]
        if completion_future.done():
            return
        completion_future.set_result(summary)

    async def wait_for_result(
        self,
        workflow_id: str,
        timeout_seconds: float,
    ) -> WorkflowSummary:
        """Wait for the final workflow summary."""
        self.register_workflow(workflow_id)
        completion_future = self._futures[workflow_id]
        return await asyncio.wait_for(completion_future, timeout=timeout_seconds)


class IngressAgent(BaseAgent):
    """Entry agent that forwards workflow requests to planner."""

    def __init__(self, engine: Engine, planner_recipient: AgentId) -> None:
        """Initialize ingress agent with planner recipient identity."""
        super().__init__(engine)
        self._planner_recipient = planner_recipient

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Forward one workflow request to planner via direct RPC."""
        workflow_id = expect_str(payload, "workflow_id")
        prompt = expect_str(payload, "prompt")
        log_line(
            "ingress",
            f"workflow={workflow_id} received prompt='{prompt}'",
        )
        # Ingress only knows the planner's logical recipient id. The distributed
        # host resolves which worker currently owns that agent type.
        planner_outcome = await self.send_message(
            {"workflow_id": workflow_id, "prompt": prompt},
            recipient=self._planner_recipient,
            correlation_id=context.correlation_id,
        )
        log_line(
            "ingress",
            f"workflow={workflow_id} planner_status={planner_outcome.status.value}",
        )
        return {
            "workflow_id": workflow_id,
            "planner_status": planner_outcome.status.value,
        }


class PlannerAgent(BaseAgent):
    """Planner agent that fans one plan out through publish."""

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Build one plan and publish it to specialist workers."""
        workflow_id = expect_str(payload, "workflow_id")
        prompt = expect_str(payload, "prompt")
        plan_text = f"plan<{prompt}>"
        log_line(
            "planner",
            f"workflow={workflow_id} built plan='{plan_text}'",
        )
        # Publish decouples the planner from specialist placement. Reusing the
        # workflow id as the topic route key lets downstream agents correlate
        # every event that belongs to the same workflow.
        ack = await self.publish_message(
            {"workflow_id": workflow_id, "plan": plan_text},
            topic=TopicId.from_values(
                type_value=PLAN_TOPIC_TYPE,
                route_key=workflow_id,
            ),
            correlation_id=context.correlation_id,
        )
        log_line(
            "planner",
            f"workflow={workflow_id} fanout_enqueued={ack.enqueued_recipient_count}",
        )
        return {
            "workflow_id": workflow_id,
            "fanout_enqueued": ack.enqueued_recipient_count,
        }


class InventoryWorkerAgent(BaseAgent):
    """Specialist worker subscribed to plan events."""

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Process planner event and publish inventory result."""
        workflow_id = expect_str(payload, "workflow_id")
        plan = expect_str(payload, "plan")
        await asyncio.sleep(0.12)
        result_text = f"inventory-checked<{plan}>"
        log_line(
            "inventory",
            f"workflow={workflow_id} publishing result",
        )
        await self.publish_message(
            {
                "workflow_id": workflow_id,
                "worker_name": "inventory",
                "output": result_text,
            },
            topic=TopicId.from_values(
                type_value=RESULT_TOPIC_TYPE,
                route_key=workflow_id,
            ),
            correlation_id=context.correlation_id,
        )
        return None


class PricingWorkerAgent(BaseAgent):
    """Specialist worker subscribed to plan events."""

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Process planner event and publish pricing result."""
        workflow_id = expect_str(payload, "workflow_id")
        plan = expect_str(payload, "plan")
        await asyncio.sleep(0.05)
        result_text = f"price-estimated<{plan}>"
        log_line(
            "pricing",
            f"workflow={workflow_id} publishing result",
        )
        await self.publish_message(
            {
                "workflow_id": workflow_id,
                "worker_name": "pricing",
                "output": result_text,
            },
            topic=TopicId.from_values(
                type_value=RESULT_TOPIC_TYPE,
                route_key=workflow_id,
            ),
            correlation_id=context.correlation_id,
        )
        return None


class AggregatorAgent(BaseAgent):
    """Stateful fan-in agent keyed by workflow id."""

    def __init__(
        self,
        engine: Engine,
        tracker: CompletionTracker,
        expected_result_count: int,
    ) -> None:
        """Initialize aggregator with external completion tracker."""
        super().__init__(engine)
        self._tracker = tracker
        self._expected_result_count = expected_result_count
        self._partial_results: dict[str, str] = {}

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Collect worker results and resolve workflow completion when ready."""
        _ = context
        # For stateful topic delivery, the runtime materializes the subscriber
        # agent id from the topic route key. The aggregator can therefore use
        # its own id key as the workflow id it is collecting for.
        workflow_id = self.id.key.value
        worker_name = expect_str(payload, "worker_name")
        output = expect_str(payload, "output")
        # This state lives inside one logical aggregator agent instance, so each
        # workflow accumulates results independently even when many workflows are
        # active at the same time.
        self._partial_results[worker_name] = output
        log_line(
            "aggregator",
            (
                f"workflow={workflow_id} collected "
                f"{len(self._partial_results)}/{self._expected_result_count}"
            ),
        )
        if len(self._partial_results) < self._expected_result_count:
            return {
                "workflow_id": workflow_id,
                "pending_sources": self._expected_result_count
                - len(self._partial_results),
            }

        ordered_names = sorted(self._partial_results)
        merged_output = " | ".join(
            f"{name}:{self._partial_results[name]}" for name in ordered_names
        )
        summary = WorkflowSummary(
            workflow_id=workflow_id,
            merged_output=merged_output,
            source_count=len(self._partial_results),
        )
        self._tracker.complete(summary)
        log_line("aggregator", f"workflow={workflow_id} completed fan-in")
        return None


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the demo."""
    parser = argparse.ArgumentParser(
        description="Distributed publish fan-out / fan-in demo.",
    )
    parser.add_argument(
        "--workflow-count",
        type=int,
        default=3,
        help="Number of workflows to run in parallel.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help="Timeout when waiting for aggregated workflow completion.",
    )
    return parser


def print_cluster_layout(
    *,
    host: WorkerAgentRuntimeHost,
    ingress_worker: WorkerAgentRuntime,
    planner_worker: WorkerAgentRuntime,
    inventory_worker: WorkerAgentRuntime,
    pricing_worker: WorkerAgentRuntime,
    aggregator_worker: WorkerAgentRuntime,
) -> None:
    """Print resolved host and worker addresses."""
    table = Table(title="Distributed Runtime Layout", show_lines=False)
    table.add_column("Component", style="bold cyan")
    table.add_column("Address", style="white")
    table.add_row("host", host.address)
    table.add_row("ingress_worker", ingress_worker.address)
    table.add_row("planner_worker", planner_worker.address)
    table.add_row("inventory_worker", inventory_worker.address)
    table.add_row("pricing_worker", pricing_worker.address)
    table.add_row("aggregator_worker", aggregator_worker.address)
    CONSOLE.print(table)


def print_final_results(summaries: list[WorkflowSummary]) -> None:
    """Print final aggregated workflow summaries."""
    table = Table(title="Aggregated Workflow Results", show_lines=False)
    table.add_column("Workflow", style="bold green")
    table.add_column("Sources", justify="right")
    table.add_column("Merged Output", style="white")
    for summary in summaries:
        table.add_row(
            summary.workflow_id,
            str(summary.source_count),
            summary.merged_output,
        )
    CONSOLE.print(table)


async def run_demo(config: DemoConfig) -> None:
    """Run the distributed publish fan-out / fan-in demo."""
    # The host is the distributed control plane. Workers connect to it, advertise
    # their agent types/subscriptions, and send all cross-process traffic through it.
    host = WorkerAgentRuntimeHost(address="127.0.0.1:0")
    tracker = CompletionTracker()

    await host.start()
    log_line("host", f"started distributed host at {host.address}")

    ingress_worker = WorkerAgentRuntime(host_address=host.address)
    planner_worker = WorkerAgentRuntime(host_address=host.address)
    inventory_worker = WorkerAgentRuntime(host_address=host.address)
    pricing_worker = WorkerAgentRuntime(host_address=host.address)
    aggregator_worker = WorkerAgentRuntime(host_address=host.address)

    ingress_worker.register_factory(
        INGRESS_AGENT_TYPE,
        lambda engine: IngressAgent(
            engine,
            planner_recipient=AgentId.from_values(PLANNER_AGENT_TYPE, "planner"),
        ),
    )
    planner_worker.register_factory(PLANNER_AGENT_TYPE, PlannerAgent)

    inventory_worker.register_factory(INVENTORY_AGENT_TYPE, InventoryWorkerAgent)
    # Stateless subscribers receive one delivery per publish without keeping
    # per-workflow local state between messages.
    inventory_worker.subscribe_exact(
        topic_type=PLAN_TOPIC_TYPE,
        agent_type=INVENTORY_AGENT_TYPE,
        delivery_mode=DeliveryMode.STATELESS,
    )

    pricing_worker.register_factory(PRICING_AGENT_TYPE, PricingWorkerAgent)
    pricing_worker.subscribe_exact(
        topic_type=PLAN_TOPIC_TYPE,
        agent_type=PRICING_AGENT_TYPE,
        delivery_mode=DeliveryMode.STATELESS,
    )

    aggregator_worker.register_factory(
        AGGREGATOR_AGENT_TYPE,
        lambda engine: AggregatorAgent(
            engine,
            tracker=tracker,
            expected_result_count=2,
        ),
    )
    aggregator_worker.subscribe_exact(
        topic_type=RESULT_TOPIC_TYPE,
        agent_type=AGGREGATOR_AGENT_TYPE,
    )

    workers = [
        ingress_worker,
        planner_worker,
        inventory_worker,
        pricing_worker,
        aggregator_worker,
    ]
    # Starting a worker opens its gRPC listener and registers its factories and
    # subscriptions with the host before the demo starts sending traffic.
    await asyncio.gather(*(worker.start() for worker in workers))
    print_cluster_layout(
        host=host,
        ingress_worker=ingress_worker,
        planner_worker=planner_worker,
        inventory_worker=inventory_worker,
        pricing_worker=pricing_worker,
        aggregator_worker=aggregator_worker,
    )

    result_tasks: list[asyncio.Task[WorkflowSummary]] = []
    try:
        workflow_requests: list[dict[str, object]] = []
        for index in range(config.workflow_count):
            workflow_id = f"workflow-{index + 1}"
            tracker.register_workflow(workflow_id)
            workflow_requests.append(
                {
                    "workflow_id": workflow_id,
                    "prompt": f"launch-campaign-{index + 1}",
                }
            )
            result_tasks.append(
                asyncio.create_task(
                    tracker.wait_for_result(workflow_id, config.timeout_seconds)
                )
            )

        # The demo only performs direct RPCs into the ingress worker. Everything
        # after that point fans out and back in through the distributed runtime.
        ingress_outcomes = await asyncio.gather(
            *(
                ingress_worker.send_message(
                    request,
                    recipient=AgentId.from_values(
                        INGRESS_AGENT_TYPE,
                        expect_str(request, "workflow_id"),
                    ),
                )
                for request in workflow_requests
            )
        )
        for request, outcome in zip(workflow_requests, ingress_outcomes, strict=True):
            if outcome.status != DeliveryStatus.DELIVERED:
                raise RuntimeError(
                    "Ingress RPC failed for "
                    f"{expect_str(request, 'workflow_id')}: {outcome.status.value}"
                )

        summaries = await asyncio.gather(*result_tasks)
        print_final_results(summaries)
    finally:
        for task in result_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*(worker.stop_when_idle() for worker in workers))
        await host.stop_when_idle()


def main() -> None:
    """Parse CLI args and run the demo."""
    parser = build_parser()
    args = parser.parse_args()
    config = DemoConfig(
        workflow_count=args.workflow_count,
        timeout_seconds=args.timeout_seconds,
    )
    CONSOLE.print(
        Panel.fit(
            (
                "Distributed publish fan-out / fan-in demo\n"
                "- planner publishes one plan event\n"
                "- specialist workers handle the fan-out\n"
                "- one stateful aggregator agent fans results back in"
            ),
            border_style="cyan",
        )
    )
    asyncio.run(run_demo(config))


if __name__ == "__main__":
    main()
