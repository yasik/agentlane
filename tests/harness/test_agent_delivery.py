"""Delivery aborts release child configuration before instance construction."""

import asyncio

import pytest

from agentlane.harness import Agent, AgentDescriptor, Runner, RunState
from agentlane.harness._agent_delivery import AgentDelivery
from agentlane.messaging import AgentId, DeliveryStatus
from agentlane.runtime import (
    AgentRegistry,
    CancellationToken,
    SingleThreadedRuntimeEngine,
)
from agentlane.runtime._types import DeliveryTask


class _PausedRuntime(SingleThreadedRuntimeEngine):
    def __init__(self, registry: AgentRegistry) -> None:
        super().__init__(registry=registry)
        self.submitted = asyncio.Event()
        self.release = asyncio.Event()

    async def _submit(self, task: DeliveryTask) -> None:
        self.submitted.set()
        await self.release.wait()
        await super()._submit(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
async def test_cancelled_delivery_registers_no_child(streamed: bool) -> None:
    registry = AgentRegistry()
    runtime = SingleThreadedRuntimeEngine(registry=registry)
    runner = Runner()
    parent = Agent(runtime, runner)
    cancellation = CancellationToken()
    cancellation.cancel()
    delivery = AgentDelivery(parent, runner, cancellation)
    child_id = AgentId.from_values("child", "cancelled")
    descriptor = AgentDescriptor(name="child")

    with pytest.raises(asyncio.CancelledError):
        if streamed:
            await delivery.stream_handoff(
                recipient=child_id,
                descriptor=descriptor,
                transferred_state=RunState(instructions=None, history=[], responses=[]),
                parent_tools=None,
                emit=lambda _event: None,
                control_stream=None,
            )
        else:
            await delivery.send(
                recipient=child_id,
                descriptor=descriptor,
                run_input="task",
                parent_tools=None,
            )

    assert not registry.has_instance(child_id)
    with pytest.raises(LookupError):
        await registry.get_or_create(child_id, engine=runtime)


@pytest.mark.asyncio
async def test_cancelled_delivery_discards_pending_child_before_dispatch() -> None:
    registry = AgentRegistry()
    runtime = _PausedRuntime(registry)
    runner = Runner()
    parent = Agent(runtime, runner)
    delivery = AgentDelivery(parent, runner, None)
    child_id = AgentId.from_values("child", "pending")
    task = asyncio.create_task(
        delivery.send(
            recipient=child_id,
            descriptor=AgentDescriptor(name="child"),
            run_input="task",
            parent_tools=None,
        )
    )

    try:
        await runtime.submitted.wait()
        assert not registry.has_instance(child_id)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        runtime.release.set()
        outcome = await runtime.send_message("later", child_id)
        assert outcome.status is DeliveryStatus.UNDELIVERABLE
        assert not registry.has_instance(child_id)
    finally:
        runtime.release.set()
        task.cancel()
        await runtime.stop_when_idle()
