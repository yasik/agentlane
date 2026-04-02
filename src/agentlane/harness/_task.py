"""Base task primitive for the harness."""

from agentlane.messaging import AgentId
from agentlane.runtime import BaseAgent, Engine


class Task(BaseAgent):
    """Base harness task primitive.

    Tasks are the top-level unit of work above LLM-driven agents. They reuse
    the runtime messaging contract directly and keep policy minimal so
    application-specific orchestration can stay open-ended.
    """

    def __init__(self, engine: Engine, bind_id: AgentId | None = None) -> None:
        """Initialize a task bound to one runtime engine capability.

        Args:
            engine: Runtime engine messaging capability exposed to this task.
            bind_id: Optional pre-bound task id, primarily for tests.
        """
        super().__init__(engine, bind_id=bind_id)
