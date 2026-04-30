"""Simple distributed direct trade analysis scatter-gather starter example."""

import asyncio
from collections.abc import Mapping
from typing import cast

from agentlane.messaging import AgentId, DeliveryOutcome, DeliveryStatus, MessageContext
from agentlane.runtime import (
    BaseAgent,
    Engine,
    WorkerAgentRuntime,
    WorkerAgentRuntimeHost,
    on_message,
)

COORDINATOR_AGENT_TYPE = "simple.scatter.coordinator"
EXECUTION_AGENT_TYPE = "simple.scatter.execution"
RISK_AGENT_TYPE = "simple.scatter.risk"


def expect_mapping(payload: object, context: str) -> dict[str, object]:
    """Return one JSON-like mapping payload."""
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping payload for {context}.")
    return cast(dict[str, object], payload)


def expect_str(payload: Mapping[str, object], field_name: str) -> str:
    """Return one required string field from a JSON-like payload."""
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise TypeError(f"Expected string field '{field_name}'.")
    return value


def expect_delivered(outcome: DeliveryOutcome, label: str) -> dict[str, object]:
    """Return the payload for one successful direct RPC."""
    if outcome.status != DeliveryStatus.DELIVERED:
        raise RuntimeError(f"{label} RPC failed: {outcome.status.value}")
    if outcome.response_payload is None:
        raise RuntimeError(f"{label} RPC returned no payload.")
    return expect_mapping(outcome.response_payload, label)


class CoordinatorAgent(BaseAgent):
    """Calls both specialists and merges their responses."""

    def __init__(
        self,
        engine: Engine,
        execution_recipient: AgentId,
        risk_recipient: AgentId,
    ) -> None:
        """Initialize coordinator with specialist recipients."""
        super().__init__(engine)
        self._execution_recipient = execution_recipient
        self._risk_recipient = risk_recipient

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Scatter direct RPCs to execution and risk specialists."""
        request_id = expect_str(payload, "request_id")
        text = expect_str(payload, "text")
        print(f"coordinator: sending trade review for {request_id}")

        execution_task = asyncio.create_task(
            self.send_message(
                {"request_id": request_id, "text": text},
                recipient=self._execution_recipient,
                correlation_id=context.correlation_id,
            )
        )
        risk_task = asyncio.create_task(
            self.send_message(
                {"request_id": request_id, "text": text},
                recipient=self._risk_recipient,
                correlation_id=context.correlation_id,
            )
        )

        execution_outcome, risk_outcome = await asyncio.gather(
            execution_task,
            risk_task,
        )
        execution = expect_delivered(execution_outcome, "execution")
        risk = expect_delivered(risk_outcome, "risk")
        result = {
            "request_id": request_id,
            "execution": expect_str(execution, "execution"),
            "risk": expect_str(risk, "risk"),
        }
        print(f"coordinator: merged result for {request_id}")
        return result


class ExecutionAgent(BaseAgent):
    """Returns an execution analysis for the trade text."""

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Handle one direct RPC."""
        _ = context
        request_id = expect_str(payload, "request_id")
        text = expect_str(payload, "text")
        print(f"execution: handling {request_id}")
        return {"execution": f"route-check<{text}>"}


class RiskAgent(BaseAgent):
    """Returns a risk analysis for the trade text."""

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Handle one direct RPC."""
        _ = context
        request_id = expect_str(payload, "request_id")
        text = expect_str(payload, "text")
        print(f"risk: handling {request_id}")
        return {"risk": f"limit-checks={len(text.split())}"}


async def run_example() -> None:
    """Run the simple distributed scatter-gather example."""
    host = WorkerAgentRuntimeHost(address="127.0.0.1:0")
    await host.start()

    coordinator_worker = WorkerAgentRuntime(host_address=host.address)
    execution_worker = WorkerAgentRuntime(host_address=host.address)
    risk_worker = WorkerAgentRuntime(host_address=host.address)

    coordinator_worker.register_factory(
        COORDINATOR_AGENT_TYPE,
        lambda engine: CoordinatorAgent(
            engine,
            execution_recipient=AgentId.from_values(
                EXECUTION_AGENT_TYPE,
                "execution",
            ),
            risk_recipient=AgentId.from_values(RISK_AGENT_TYPE, "risk"),
        ),
    )
    execution_worker.register_factory(EXECUTION_AGENT_TYPE, ExecutionAgent)
    risk_worker.register_factory(RISK_AGENT_TYPE, RiskAgent)

    workers = [coordinator_worker, execution_worker, risk_worker]
    await asyncio.gather(*(worker.start() for worker in workers))

    try:
        request = {
            "request_id": "trade-1",
            "text": "buy AAPL block order near market close",
        }
        print(f"main: sending {request['request_id']}")
        outcome = await coordinator_worker.send_message(
            request,
            recipient=AgentId.from_values(COORDINATOR_AGENT_TYPE, "trade-1"),
        )
        result = expect_delivered(outcome, "coordinator")
        print(f"main: final result -> {result}")
    finally:
        await asyncio.gather(*(worker.stop_when_idle() for worker in workers))
        await host.stop_when_idle()


def main() -> None:
    """Run the example."""
    asyncio.run(run_example())


if __name__ == "__main__":
    main()
