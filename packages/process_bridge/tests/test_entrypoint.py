import asyncio
import sys
from types import ModuleType

import pytest
from agentlane_process_bridge import AgentBackend
from agentlane_process_bridge.__main__ import (
    load_backend_factory,
    parse_app_reference,
    resolve_agent_backend,
)

from .helpers import FakeAgent


def test_parse_app_reference_requires_module_and_attribute() -> None:
    assert parse_app_reference("demo.backend:create_backend") == (
        "demo.backend",
        "create_backend",
    )

    with pytest.raises(ValueError, match="module:attribute"):
        parse_app_reference("demo.backend")


def test_load_backend_factory_rejects_non_callable() -> None:
    module = ModuleType("test_process_bridge_app")
    module.__dict__["backend"] = object()
    sys.modules[module.__name__] = module
    try:
        with pytest.raises(TypeError, match="not callable"):
            load_backend_factory(f"{module.__name__}:backend")
    finally:
        sys.modules.pop(module.__name__, None)


def test_resolve_agent_backend_accepts_backend_and_bare_agent() -> None:
    async def scenario() -> None:
        agent = FakeAgent()
        backend = AgentBackend(agent=agent)

        assert await resolve_agent_backend(backend) is backend
        assert (await resolve_agent_backend(agent)).agent is agent

    asyncio.run(scenario())


def test_resolve_agent_backend_awaits_factory_result() -> None:
    async def factory() -> AgentBackend:
        return AgentBackend(agent=FakeAgent())

    async def scenario() -> None:
        backend = await resolve_agent_backend(factory())

        assert isinstance(backend.agent, FakeAgent)

    asyncio.run(scenario())
