import asyncio
import sys
from types import ModuleType

import pytest
from agentlane_process_bridge import AgentBackend
from agentlane_process_bridge.__main__ import (
    load_backend_factory,
    parse_app_reference,
    resolve_agent_backend,
    run_app_reference,
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


def test_run_app_reference_forwards_backend_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ConfigStore:
        def snapshot(self) -> dict[str, object]:
            return {"model": "openai/gpt-5.5"}

        def apply(self, patch: dict[str, object]) -> dict[str, object]:
            del patch

            return self.snapshot()

    module = ModuleType("test_process_bridge_config_app")
    config = ConfigStore()
    module.__dict__["create_backend"] = lambda: AgentBackend(
        agent=FakeAgent(),
        config=config,
    )
    sys.modules[module.__name__] = module
    captured: dict[str, object] = {}

    async def fake_run_stdio(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("agentlane_process_bridge.__main__.run_stdio", fake_run_stdio)

    try:
        asyncio.run(run_app_reference(f"{module.__name__}:create_backend"))
    finally:
        sys.modules.pop(module.__name__, None)

    assert captured["config"] is config
