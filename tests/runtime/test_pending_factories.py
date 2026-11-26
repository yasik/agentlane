"""Exact-id child factories preserve configuration through delayed dispatch."""

import asyncio

import pytest

from agentlane.messaging import AgentId, DeliveryStatus
from agentlane.runtime import (
    AgentRegistry,
    BaseAgent,
    CancellationToken,
    Engine,
    MessageContext,
    SingleThreadedRuntimeEngine,
    WorkerAgentRuntime,
    WorkerAgentRuntimeHost,
    on_message,
)


class _ConfiguredAgent(BaseAgent):
    def __init__(self, engine: Engine, label: str) -> None:
        super().__init__(engine)
        self.label = label
        self.count = 0

    @on_message
    async def handle(self, payload: str, context: MessageContext) -> object:
        del payload, context
        self.count += 1
        return {"label": self.label, "count": self.count}


@pytest.mark.asyncio
async def test_pending_factories_keep_same_type_instances_and_later_messages() -> None:
    runtime = SingleThreadedRuntimeEngine()
    first_id = AgentId.from_values("child", "first")
    second_id = AgentId.from_values("child", "second")
    runtime.register_factory(
        "child", lambda engine: _ConfiguredAgent(engine, "default")
    )
    runtime.register_pending_factory(
        first_id, lambda engine: _ConfiguredAgent(engine, "quick")
    )
    runtime.register_pending_factory(
        second_id, lambda engine: _ConfiguredAgent(engine, "reasoning")
    )

    # Both pending registrations exist before either request can dispatch.
    try:
        first, second = await asyncio.gather(
            runtime.send_message("first", first_id),
            runtime.send_message("second", second_id),
        )
        runtime.discard_pending_factory(first_id)
        runtime.discard_pending_factory(second_id)
        later = await runtime.send_message("later", first_id)

        assert first.response_payload == {"label": "quick", "count": 1}
        assert second.response_payload == {"label": "reasoning", "count": 1}
        assert later.response_payload == {"label": "quick", "count": 2}
    finally:
        await runtime.stop_when_idle()


@pytest.mark.asyncio
async def test_pending_factory_abort_does_not_construct_or_retain_child() -> None:
    registry = AgentRegistry()
    runtime = SingleThreadedRuntimeEngine(registry=registry)
    child_id = AgentId.from_values("child", "cancelled")
    constructed: list[_ConfiguredAgent] = []

    def factory(engine: Engine) -> _ConfiguredAgent:
        agent = _ConfiguredAgent(engine, "unused")
        constructed.append(agent)
        return agent

    runtime.register_pending_factory(child_id, factory)
    cancellation = CancellationToken()
    cancellation.cancel()
    try:
        outcome = await runtime.send_message(
            "cancelled", child_id, cancellation_token=cancellation
        )
        runtime.discard_pending_factory(child_id)

        assert outcome.status is DeliveryStatus.CANCELED
        assert constructed == []
        assert not registry.has_instance(child_id)
        # A removed pending registration cannot construct an instance later.
        later = await runtime.send_message("later", child_id)
        assert later.status is DeliveryStatus.UNDELIVERABLE
    finally:
        await runtime.stop_when_idle()


@pytest.mark.asyncio
async def test_pending_factory_cannot_replace_live_or_pending_child() -> None:
    runtime = SingleThreadedRuntimeEngine()
    child_id = AgentId.from_values("child", "stable")

    def factory(engine: Engine) -> _ConfiguredAgent:
        return _ConfiguredAgent(engine, "original")

    runtime.register_pending_factory(child_id, factory)
    with pytest.raises(ValueError, match="already registered"):
        runtime.register_pending_factory(child_id, factory)

    try:
        await runtime.send_message("start", child_id)
        with pytest.raises(ValueError, match="already registered"):
            runtime.register_pending_factory(child_id, factory)
    finally:
        await runtime.stop_when_idle()


@pytest.mark.asyncio
async def test_pending_factory_creation_is_shared_by_concurrent_lookups() -> None:
    registry = AgentRegistry()
    runtime = SingleThreadedRuntimeEngine(registry=registry)
    child_id = AgentId.from_values("child", "shared")
    started = asyncio.Event()
    release = asyncio.Event()
    count = 0

    async def factory(engine: Engine) -> _ConfiguredAgent:
        nonlocal count
        count += 1
        started.set()
        await release.wait()
        return _ConfiguredAgent(engine, "shared")

    runtime.register_pending_factory(child_id, factory)
    first = asyncio.create_task(registry.get_or_create(child_id, engine=runtime))
    await started.wait()
    second = asyncio.create_task(registry.get_or_create(child_id, engine=runtime))
    # Let the second lookup join the in-flight creation before it can finish.
    await asyncio.sleep(0)
    release.set()
    first_agent, second_agent = await asyncio.gather(first, second)

    assert first_agent is second_agent
    assert first_agent.id == child_id
    assert count == 1


@pytest.mark.asyncio
async def test_cancelled_pending_factory_clears_inflight_creation() -> None:
    registry = AgentRegistry()
    runtime = SingleThreadedRuntimeEngine(registry=registry)
    child_id = AgentId.from_values("child", "cancelled_creation")
    started = asyncio.Event()

    async def cancelled_factory(engine: Engine) -> _ConfiguredAgent:
        del engine
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("Cancelled factory must not finish")

    runtime.register_pending_factory(child_id, cancelled_factory)
    creation = asyncio.create_task(registry.get_or_create(child_id, engine=runtime))
    await started.wait()
    creation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creation

    # A cancelled creation leaves no stale future or retained instance.
    assert not registry.has_instance(child_id)
    runtime.register_pending_factory(
        child_id, lambda engine: _ConfiguredAgent(engine, "retry")
    )
    assert (await registry.get_or_create(child_id, engine=runtime)).id == child_id


@pytest.mark.asyncio
async def test_worker_routes_pending_exact_id_factory_through_host() -> None:
    host = WorkerAgentRuntimeHost(address="127.0.0.1:0")
    await host.start()
    sender = WorkerAgentRuntime(host_address=host.address)
    receiver = WorkerAgentRuntime(host_address=host.address)
    first_id = AgentId.from_values("pending-child", "first")
    second_id = AgentId.from_values("pending-child", "second")
    receiver.register_pending_factory(
        first_id, lambda engine: _ConfiguredAgent(engine, "quick")
    )
    receiver.register_pending_factory(
        second_id, lambda engine: _ConfiguredAgent(engine, "reasoning")
    )
    await receiver.start()
    await sender.start()

    try:
        first, second = await asyncio.gather(
            sender.send_message("first", first_id),
            sender.send_message("second", second_id),
        )
        receiver.discard_pending_factory(first_id)
        receiver.discard_pending_factory(second_id)
        later = await sender.send_message("later", first_id)

        assert first.response_payload == {"label": "quick", "count": 1}
        assert second.response_payload == {"label": "reasoning", "count": 1}
        assert later.response_payload == {"label": "quick", "count": 2}
    finally:
        await sender.stop_when_idle()
        await receiver.stop_when_idle()
        await host.stop_when_idle()
