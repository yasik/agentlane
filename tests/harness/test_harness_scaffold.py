from typing import cast

import agentlane.harness.context as harness_context
import agentlane.harness.memory as harness_memory
from agentlane.harness import Agent, Runner, RunnerHooks, Task
from agentlane.models import MessageDict
from agentlane.runtime import BaseAgent, CancellationToken, Engine


class _FakeTask(Task):
    pass


class _FakeAgent(Agent):
    pass


class _FakeRunner(Runner):
    async def run(
        self,
        agent: Agent,
        messages: list[MessageDict],
        *,
        hooks: RunnerHooks | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> list[object]:
        _ = agent
        _ = messages
        _ = hooks
        _ = cancellation_token
        return []


def test_harness_public_exports_are_defined() -> None:
    assert Task is not None
    assert Agent is not None
    assert Runner is not None
    assert RunnerHooks is not None


def test_task_and_agent_extend_base_agent() -> None:
    assert issubclass(Task, BaseAgent)
    assert issubclass(Agent, Task)
    assert issubclass(_FakeTask, BaseAgent)
    assert issubclass(_FakeAgent, Task)


def test_runner_can_be_specialized() -> None:
    _ = cast(Runner, _FakeRunner())


def test_harness_subpackages_import() -> None:
    assert harness_context.__all__ == []
    assert harness_memory.__all__ == []


def test_task_constructor_matches_runtime_engine_contract() -> None:
    _ = _FakeTask(cast(Engine, object()))
