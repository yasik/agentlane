"""Simple distributed portfolio analysis fan-out / fan-in starter example."""

import asyncio

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

PLANNER_AGENT_TYPE = "simple.publish.planner"
MARKET_DATA_AGENT_TYPE = "simple.publish.market_data"
RISK_AGENT_TYPE = "simple.publish.risk"
AGGREGATOR_AGENT_TYPE = "simple.publish.aggregator"

WORK_TOPIC_TYPE = "simple.publish.work"
RESULT_TOPIC_TYPE = "simple.publish.result"


def expect_str(payload: dict[str, object], field_name: str) -> str:
    """Return one required string field from a JSON-like payload."""
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise TypeError(f"Expected string field '{field_name}'.")
    return value


class JobTracker:
    """Tracks one final summary per job id."""

    def __init__(self) -> None:
        """Initialize tracker state."""
        self._futures: dict[str, asyncio.Future[str]] = {}
        self._results: dict[str, dict[str, str]] = {}

    def start(self, job_id: str) -> None:
        """Create storage for one job id if needed."""
        if job_id in self._futures:
            return
        self._futures[job_id] = asyncio.get_running_loop().create_future()
        self._results[job_id] = {}

    def record_result(self, job_id: str, worker_name: str, value: str) -> str | None:
        """Store one worker result and return the final summary when complete."""
        self.start(job_id)
        self._results[job_id][worker_name] = value
        if len(self._results[job_id]) < 2:
            return None

        ordered_names = sorted(self._results[job_id])
        summary = " | ".join(
            f"{name}={self._results[job_id][name]}" for name in ordered_names
        )
        future = self._futures[job_id]
        if not future.done():
            future.set_result(summary)
        return summary

    async def wait(self, job_id: str) -> str:
        """Wait for the final summary for one job id."""
        self.start(job_id)
        return await self._futures[job_id]


class PlannerAgent(BaseAgent):
    """Publishes one portfolio review to all subscribed specialist workers."""

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Publish one portfolio review item."""
        job_id = expect_str(payload, "job_id")
        text = expect_str(payload, "text")
        print(f"planner: publishing portfolio review for {job_id}")
        ack = await self.publish_message(
            {"job_id": job_id, "text": text},
            topic=TopicId.from_values(type_value=WORK_TOPIC_TYPE, route_key=job_id),
            correlation_id=context.correlation_id,
        )
        return {"enqueued": ack.enqueued_recipient_count}


class MarketDataAgent(BaseAgent):
    """Builds a compact market data view for the incoming portfolio review."""

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Process one published portfolio review item."""
        job_id = expect_str(payload, "job_id")
        text = expect_str(payload, "text")
        print(f"market_data: received portfolio review for {job_id}")
        await self.publish_message(
            {
                "worker_name": "market_data",
                "value": f"snapshot<{text}>",
            },
            topic=TopicId.from_values(type_value=RESULT_TOPIC_TYPE, route_key=job_id),
            correlation_id=context.correlation_id,
        )
        return None


class RiskAgent(BaseAgent):
    """Produces a compact risk summary for the incoming portfolio review."""

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Process one published portfolio review item."""
        job_id = expect_str(payload, "job_id")
        text = expect_str(payload, "text")
        print(f"risk: received portfolio review for {job_id}")
        await self.publish_message(
            {
                "worker_name": "risk",
                "value": f"limit_checks={len(text.split())}",
            },
            topic=TopicId.from_values(type_value=RESULT_TOPIC_TYPE, route_key=job_id),
            correlation_id=context.correlation_id,
        )
        return None


class AggregatorAgent(BaseAgent):
    """Collects specialist results for one job id."""

    def __init__(self, engine: Engine, tracker: JobTracker) -> None:
        """Initialize aggregator with shared completion tracking."""
        super().__init__(engine)
        self._tracker = tracker

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Record one specialist result."""
        _ = context
        job_id = self.id.key.value
        worker_name = expect_str(payload, "worker_name")
        value = expect_str(payload, "value")
        print(f"aggregator: received {worker_name} result for {job_id}")
        summary = self._tracker.record_result(job_id, worker_name, value)
        if summary is not None:
            print(f"aggregator: completed {job_id} -> {summary}")
        return None


async def run_example() -> None:
    """Run the simple distributed publish fan-in example."""
    host = WorkerAgentRuntimeHost(address="127.0.0.1:0")
    tracker = JobTracker()
    await host.start()

    planner_worker = WorkerAgentRuntime(host_address=host.address)
    market_data_worker = WorkerAgentRuntime(host_address=host.address)
    risk_worker = WorkerAgentRuntime(host_address=host.address)
    aggregator_worker = WorkerAgentRuntime(host_address=host.address)

    planner_worker.register_factory(PLANNER_AGENT_TYPE, PlannerAgent)
    market_data_worker.register_factory(MARKET_DATA_AGENT_TYPE, MarketDataAgent)
    market_data_worker.subscribe_exact(
        topic_type=WORK_TOPIC_TYPE,
        agent_type=MARKET_DATA_AGENT_TYPE,
        delivery_mode=DeliveryMode.STATELESS,
    )
    risk_worker.register_factory(RISK_AGENT_TYPE, RiskAgent)
    risk_worker.subscribe_exact(
        topic_type=WORK_TOPIC_TYPE,
        agent_type=RISK_AGENT_TYPE,
        delivery_mode=DeliveryMode.STATELESS,
    )
    aggregator_worker.register_factory(
        AGGREGATOR_AGENT_TYPE,
        lambda engine: AggregatorAgent(engine, tracker),
    )
    aggregator_worker.subscribe_exact(
        topic_type=RESULT_TOPIC_TYPE,
        agent_type=AGGREGATOR_AGENT_TYPE,
        delivery_mode=DeliveryMode.STATEFUL,
    )

    workers = [
        planner_worker,
        market_data_worker,
        risk_worker,
        aggregator_worker,
    ]
    await asyncio.gather(*(worker.start() for worker in workers))

    job_id = "job-1"
    tracker.start(job_id)
    print(f"main: sending job {job_id}")

    try:
        outcome = await planner_worker.send_message(
            {
                "job_id": job_id,
                "text": "portfolio rebalance in volatile market conditions",
            },
            recipient=AgentId.from_values(PLANNER_AGENT_TYPE, "planner"),
        )
        if outcome.status != DeliveryStatus.DELIVERED:
            raise RuntimeError(f"Planner RPC failed: {outcome.status.value}")

        summary = await tracker.wait(job_id)
        print(f"main: final summary -> {summary}")
    finally:
        await asyncio.gather(*(worker.stop_when_idle() for worker in workers))
        await host.stop_when_idle()


def main() -> None:
    """Run the example."""
    asyncio.run(run_example())


if __name__ == "__main__":
    main()
