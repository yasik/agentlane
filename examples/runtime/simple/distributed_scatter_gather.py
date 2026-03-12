"""Simple distributed direct scatter-gather starter example."""

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
UPPERCASE_AGENT_TYPE = "simple.scatter.uppercase"
REVERSE_AGENT_TYPE = "simple.scatter.reverse"


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
        uppercase_recipient: AgentId,
        reverse_recipient: AgentId,
    ) -> None:
        """Initialize coordinator with specialist recipients."""
        super().__init__(engine)
        self._uppercase_recipient = uppercase_recipient
        self._reverse_recipient = reverse_recipient

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Scatter direct RPCs to both specialists and gather their results."""
        request_id = expect_str(payload, "request_id")
        text = expect_str(payload, "text")
        print(f"coordinator: sending work for {request_id}")

        uppercase_task = asyncio.create_task(
            self.send_message(
                {"request_id": request_id, "text": text},
                recipient=self._uppercase_recipient,
                correlation_id=context.correlation_id,
            )
        )
        reverse_task = asyncio.create_task(
            self.send_message(
                {"request_id": request_id, "text": text},
                recipient=self._reverse_recipient,
                correlation_id=context.correlation_id,
            )
        )

        uppercase_outcome, reverse_outcome = await asyncio.gather(
            uppercase_task,
            reverse_task,
        )
        uppercase = expect_delivered(uppercase_outcome, "uppercase")
        reverse = expect_delivered(reverse_outcome, "reverse")
        result = {
            "request_id": request_id,
            "uppercase": expect_str(uppercase, "uppercase"),
            "reversed": expect_str(reverse, "reversed"),
        }
        print(f"coordinator: merged result for {request_id}")
        return result


class UppercaseAgent(BaseAgent):
    """Returns an uppercase version of the text."""

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
        print(f"uppercase: handling {request_id}")
        return {"uppercase": text.upper()}


class ReverseAgent(BaseAgent):
    """Returns a reversed version of the text."""

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
        print(f"reverse: handling {request_id}")
        return {"reversed": text[::-1]}


async def run_example() -> None:
    """Run the simple distributed scatter-gather example."""
    host = WorkerAgentRuntimeHost(address="127.0.0.1:0")
    await host.start()

    coordinator_worker = WorkerAgentRuntime(host_address=host.address)
    uppercase_worker = WorkerAgentRuntime(host_address=host.address)
    reverse_worker = WorkerAgentRuntime(host_address=host.address)

    coordinator_worker.register_factory(
        COORDINATOR_AGENT_TYPE,
        lambda engine: CoordinatorAgent(
            engine,
            uppercase_recipient=AgentId.from_values(UPPERCASE_AGENT_TYPE, "uppercase"),
            reverse_recipient=AgentId.from_values(REVERSE_AGENT_TYPE, "reverse"),
        ),
    )
    uppercase_worker.register_factory(UPPERCASE_AGENT_TYPE, UppercaseAgent)
    reverse_worker.register_factory(REVERSE_AGENT_TYPE, ReverseAgent)

    workers = [coordinator_worker, uppercase_worker, reverse_worker]
    await asyncio.gather(*(worker.start() for worker in workers))

    try:
        request = {
            "request_id": "request-1",
            "text": "simple distributed runtime example",
        }
        print(f"main: sending {request['request_id']}")
        outcome = await coordinator_worker.send_message(
            request,
            recipient=AgentId.from_values(COORDINATOR_AGENT_TYPE, "request-1"),
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
