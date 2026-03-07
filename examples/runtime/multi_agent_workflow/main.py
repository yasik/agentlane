"""Multi-agent runtime demo with direct send + publish fan-out patterns."""

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
    SingleThreadedRuntimeEngine,
    on_message,
    single_threaded_runtime,
)

INGRESS_AGENT_TYPE = "demo.ingress"
PLANNER_AGENT_TYPE = "demo.planner"
WORKER_A_AGENT_TYPE = "demo.worker_a"
WORKER_B_AGENT_TYPE = "demo.worker_b"
AGGREGATOR_AGENT_TYPE = "demo.aggregator"

PLAN_TOPIC_TYPE = "demo.workflow.plan_ready"
RESULT_TOPIC_TYPE = "demo.workflow.result"
CONSOLE = Console()
COMPONENT_STYLES = {
    "demo": "bold cyan",
    "ingress": "bold green",
    "planner": "bold magenta",
    "worker_a": "bold blue",
    "worker_b": "bold yellow",
    "aggregator": "bold white",
}


def log_line(component: str, message: str) -> None:
    """Print one timestamped demo log line."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    component_style = COMPONENT_STYLES.get(component, "white")
    CONSOLE.print(
        f"[dim][{timestamp}][/dim] "
        f"[{component_style}]{component:<20}[/{component_style}] "
        f"[dim]|[/dim] {message}"
    )


@dataclass(slots=True, frozen=True)
class DemoConfig:
    """Runtime configuration for the multi-agent workflow demo."""

    workflow_count: int
    """Number of workflows to start in parallel."""

    worker_count: int
    """Worker count for `SingleThreadedRuntimeEngine`."""

    timeout_seconds: float
    """Timeout waiting for aggregated workflow completion."""

    aggregator_route_key: str
    """Route key used to force a single stateful aggregator instance."""


@dataclass(slots=True)
class UserWorkflowRequest:
    """Ingress payload to start one workflow."""

    workflow_id: str
    """Unique workflow identifier."""

    prompt: str
    """Input task prompt for planner."""


@dataclass(slots=True)
class PlannerTask:
    """Planner payload sent via direct message from ingress."""

    workflow_id: str
    """Unique workflow identifier."""

    prompt: str
    """Workflow prompt forwarded from ingress."""


@dataclass(slots=True)
class PlanReadyEvent:
    """Publish payload emitted by planner for worker fan-out."""

    workflow_id: str
    """Unique workflow identifier."""

    plan: str
    """Planned instructions consumed by workers."""


@dataclass(slots=True)
class WorkerResultEvent:
    """Publish payload emitted by workers for aggregation."""

    workflow_id: str
    """Unique workflow identifier."""

    worker_name: str
    """Worker label (`worker_a` or `worker_b`)."""

    output: str
    """Worker output text."""


@dataclass(slots=True)
class AggregatedWorkflowResult:
    """Aggregated output tracked by the demo completion tracker."""

    workflow_id: str
    """Unique workflow identifier."""

    merged_output: str
    """Deterministic merged worker output string."""

    worker_count: int
    """Number of worker outputs included in merge."""


class CompletionTracker:
    """Tracks workflow completion futures resolved by aggregator agent."""

    def __init__(self, expected_worker_count: int) -> None:
        """Initialize tracker with expected worker result count per workflow."""
        self._expected_worker_count = expected_worker_count
        self._futures: dict[str, asyncio.Future[AggregatedWorkflowResult]] = {}
        self._results: dict[str, dict[str, str]] = {}

    def register_workflow(self, workflow_id: str) -> None:
        """Create completion future state for a new workflow id."""
        if workflow_id in self._futures:
            return
        self._futures[workflow_id] = asyncio.get_running_loop().create_future()
        self._results[workflow_id] = {}

    def record_worker_result(self, event: WorkerResultEvent) -> None:
        """Store one worker result and complete workflow future when ready."""
        if event.workflow_id not in self._futures:
            self.register_workflow(event.workflow_id)

        workflow_results = self._results[event.workflow_id]
        workflow_results[event.worker_name] = event.output
        if len(workflow_results) < self._expected_worker_count:
            return

        completion_future = self._futures[event.workflow_id]
        if completion_future.done():
            return

        ordered_names = sorted(workflow_results.keys())
        merged_output = " | ".join(
            f"{name}:{workflow_results[name]}" for name in ordered_names
        )
        completion_future.set_result(
            AggregatedWorkflowResult(
                workflow_id=event.workflow_id,
                merged_output=merged_output,
                worker_count=len(workflow_results),
            )
        )

    async def wait_for_result(
        self,
        workflow_id: str,
        timeout_seconds: float,
    ) -> AggregatedWorkflowResult:
        """Wait for aggregated result for one workflow id."""
        if workflow_id not in self._futures:
            self.register_workflow(workflow_id)
        completion_future = self._futures[workflow_id]
        return await asyncio.wait_for(completion_future, timeout=timeout_seconds)


class IngressAgent(BaseAgent):
    """Entry agent that receives user workflow requests."""

    def __init__(self, engine: Engine, planner_recipient: AgentId) -> None:
        """Initialize ingress agent with planner recipient identity."""
        super().__init__(engine)
        self._planner_recipient = planner_recipient

    @on_message
    async def handle(self, payload: UserWorkflowRequest, context: MessageContext) -> object:
        """Handle user request and forward to planner via direct send."""
        log_line(
            "ingress",
            f"received workflow={payload.workflow_id} prompt='{payload.prompt}'",
        )
        planner_outcome = await self.send_message(
            PlannerTask(workflow_id=payload.workflow_id, prompt=payload.prompt),
            recipient=self._planner_recipient,
            correlation_id=context.correlation_id,
        )
        log_line(
            "ingress",
            f"planner outcome workflow={payload.workflow_id} status={planner_outcome.status.value}",
        )
        return {
            "workflow_id": payload.workflow_id,
            "planner_status": planner_outcome.status.value,
        }


class PlannerAgent(BaseAgent):
    """Planner agent that publishes one plan event for worker fan-out."""

    @on_message
    async def handle(self, payload: PlannerTask, context: MessageContext) -> object:
        """Create one plan and publish it to worker subscribers."""
        plan_text = f"plan({payload.prompt})"
        log_line(
            "planner",
            f"workflow={payload.workflow_id} built plan='{plan_text}'",
        )
        await self.publish_message(
            PlanReadyEvent(workflow_id=payload.workflow_id, plan=plan_text),
            topic=TopicId.from_values(
                type_value=PLAN_TOPIC_TYPE,
                route_key=payload.workflow_id,
            ),
            correlation_id=context.correlation_id,
        )
        log_line(
            "planner",
            f"workflow={payload.workflow_id} published plan event",
        )
        return {"workflow_id": payload.workflow_id, "fanout_workers": 2}


class WorkerAAgent(BaseAgent):
    """First worker subscribed to planner publish events."""

    def __init__(self, engine: Engine, aggregator_route_key: str) -> None:
        """Initialize worker A with aggregator route key."""
        super().__init__(engine)
        self._aggregator_route_key = aggregator_route_key

    @on_message
    async def handle(self, payload: PlanReadyEvent, context: MessageContext) -> object:
        """Process planner event and publish worker result."""
        log_line(
            "worker_a",
            f"workflow={payload.workflow_id} received plan",
        )
        result_text = f"A-processed<{payload.plan}>"
        await asyncio.sleep(0.1)
        await self.publish_message(
            WorkerResultEvent(
                workflow_id=payload.workflow_id,
                worker_name="worker_a",
                output=result_text,
            ),
            topic=TopicId.from_values(
                type_value=RESULT_TOPIC_TYPE,
                route_key=self._aggregator_route_key,
            ),
            correlation_id=context.correlation_id,
        )
        log_line(
            "worker_a",
            f"workflow={payload.workflow_id} published result",
        )
        return None


class WorkerBAgent(BaseAgent):
    """Second worker subscribed to planner publish events."""

    def __init__(self, engine: Engine, aggregator_route_key: str) -> None:
        """Initialize worker B with aggregator route key."""
        super().__init__(engine)
        self._aggregator_route_key = aggregator_route_key

    @on_message
    async def handle(self, payload: PlanReadyEvent, context: MessageContext) -> object:
        """Process planner event and publish worker result."""
        log_line(
            "worker_b",
            f"workflow={payload.workflow_id} received plan",
        )
        result_text = f"B-processed<{payload.plan}>"
        await asyncio.sleep(0.2)
        await self.publish_message(
            WorkerResultEvent(
                workflow_id=payload.workflow_id,
                worker_name="worker_b",
                output=result_text,
            ),
            topic=TopicId.from_values(
                type_value=RESULT_TOPIC_TYPE,
                route_key=self._aggregator_route_key,
            ),
            correlation_id=context.correlation_id,
        )
        log_line(
            "worker_b",
            f"workflow={payload.workflow_id} published result",
        )
        return None


class AggregatorAgent(BaseAgent):
    """Single in-order stateful aggregator for all worker result events."""

    def __init__(self, engine: Engine, tracker: CompletionTracker) -> None:
        """Initialize aggregator with shared completion tracker."""
        super().__init__(engine)
        self._tracker = tracker
        self._message_index = 0

    @on_message
    async def handle(self, payload: WorkerResultEvent, context: MessageContext) -> object:
        """Aggregate worker results and complete workflows when both arrive."""
        _ = context
        self._message_index += 1
        log_line(
            "aggregator",
            (
                f"seq={self._message_index} workflow={payload.workflow_id} "
                f"received from={payload.worker_name}"
            ),
        )
        self._tracker.record_worker_result(payload)
        return {"accepted": True, "seq": self._message_index}


def print_banner(config: DemoConfig) -> None:
    """Print startup banner for the multi-agent workflow demo."""
    config_table = Table.grid(padding=(0, 2))
    config_table.add_row(
        "[bold]workflows[/bold]",
        str(config.workflow_count),
        "[bold]workers[/bold]",
        str(config.worker_count),
    )
    config_table.add_row(
        "[bold]timeout[/bold]",
        f"{config.timeout_seconds}s",
        "[bold]aggregator_route_key[/bold]",
        config.aggregator_route_key,
    )
    CONSOLE.print("")
    CONSOLE.print(
        Panel(
            config_table,
            title="[bold cyan]Runtime Demo: Multi-Agent Workflow[/bold cyan]",
            subtitle="Direct Send + Publish Fan-Out + In-Order Aggregator",
            border_style="cyan",
        )
    )


def print_final_results(results: list[AggregatedWorkflowResult]) -> None:
    """Print final aggregated workflow results."""
    result_table = Table(
        title="[bold green]Final Aggregated Results[/bold green]",
        header_style="bold cyan",
    )
    result_table.add_column("Workflow")
    result_table.add_column("Workers", justify="right")
    result_table.add_column("Merged Output")
    for result in results:
        result_table.add_row(
            result.workflow_id,
            str(result.worker_count),
            result.merged_output,
        )
    CONSOLE.print(result_table)


async def run_demo(config: DemoConfig) -> None:
    """Run multi-agent workflow demonstration."""
    print_banner(config)
    tracker = CompletionTracker(expected_worker_count=2)
    runtime = SingleThreadedRuntimeEngine(worker_count=config.worker_count)

    planner_recipient = AgentId.from_values(PLANNER_AGENT_TYPE, "default")
    runtime.register_factory(
        INGRESS_AGENT_TYPE,
        lambda engine: IngressAgent(engine=engine, planner_recipient=planner_recipient),
    )
    runtime.register_factory(
        PLANNER_AGENT_TYPE,
        PlannerAgent,
    )
    runtime.register_factory(
        WORKER_A_AGENT_TYPE,
        lambda engine: WorkerAAgent(
            engine=engine,
            aggregator_route_key=config.aggregator_route_key,
        ),
    )
    runtime.register_factory(
        WORKER_B_AGENT_TYPE,
        lambda engine: WorkerBAgent(
            engine=engine,
            aggregator_route_key=config.aggregator_route_key,
        ),
    )
    runtime.register_factory(
        AGGREGATOR_AGENT_TYPE,
        lambda engine: AggregatorAgent(engine=engine, tracker=tracker),
    )

    runtime.subscribe_exact(
        topic_type=PLAN_TOPIC_TYPE,
        agent_type=WORKER_A_AGENT_TYPE,
        delivery_mode=DeliveryMode.STATEFUL,
    )
    runtime.subscribe_exact(
        topic_type=PLAN_TOPIC_TYPE,
        agent_type=WORKER_B_AGENT_TYPE,
        delivery_mode=DeliveryMode.STATEFUL,
    )
    runtime.subscribe_exact(
        topic_type=RESULT_TOPIC_TYPE,
        agent_type=AGGREGATOR_AGENT_TYPE,
        delivery_mode=DeliveryMode.STATEFUL,
    )

    ingress_recipient = AgentId.from_values(INGRESS_AGENT_TYPE, "gateway")
    workflow_ids = [f"wf-{index + 1}" for index in range(config.workflow_count)]

    async with single_threaded_runtime(runtime):
        for workflow_id in workflow_ids:
            tracker.register_workflow(workflow_id)

        log_line("demo", "sending workflow requests to ingress")
        send_tasks = [
            asyncio.create_task(
                runtime.send_message(
                    UserWorkflowRequest(
                        workflow_id=workflow_id,
                        prompt=f"analyze-request-{workflow_id}",
                    ),
                    recipient=ingress_recipient,
                )
            )
            for workflow_id in workflow_ids
        ]
        send_outcomes = await asyncio.gather(*send_tasks)

        for workflow_id, outcome in zip(workflow_ids, send_outcomes, strict=True):
            status = outcome.status.value
            log_line("demo", f"ingress outcome workflow={workflow_id} status={status}")
            if outcome.status != DeliveryStatus.DELIVERED:
                raise RuntimeError(f"Workflow '{workflow_id}' failed to start: {status}")

        log_line("demo", "waiting for aggregator completions")
        result_tasks = [
            asyncio.create_task(
                tracker.wait_for_result(
                    workflow_id=workflow_id,
                    timeout_seconds=config.timeout_seconds,
                )
            )
            for workflow_id in workflow_ids
        ]
        aggregated_results = await asyncio.gather(*result_tasks)
        print_final_results(aggregated_results)


def parse_args() -> DemoConfig:
    """Parse command line arguments into demo configuration."""
    parser = argparse.ArgumentParser(
        description="Runtime demo: direct send + publish fan-out with in-order aggregator.",
    )
    parser.add_argument("--workflow-count", type=int, default=3)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--aggregator-route-key", type=str, default="global")
    args = parser.parse_args()
    return DemoConfig(
        workflow_count=args.workflow_count,
        worker_count=args.worker_count,
        timeout_seconds=args.timeout_seconds,
        aggregator_route_key=args.aggregator_route_key,
    )


def main() -> None:
    """CLI entry point for the multi-agent workflow runtime demo."""
    config = parse_args()
    asyncio.run(run_demo(config))


if __name__ == "__main__":
    main()
