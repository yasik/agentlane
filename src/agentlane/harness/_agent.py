"""Default harness agent primitive."""

from agentlane.messaging import AgentId
from agentlane.runtime import Engine

from ._task import Task


class Agent(Task):
    """Default harness agent primitive.

    The detailed agent lifecycle, message queueing, tool execution, and
    handoff behavior land in later phases. Phase 1 freezes only the base type
    and package boundary.
    """

    def __init__(self, engine: Engine, bind_id: AgentId | None = None) -> None:
        """Initialize an agent bound to one runtime engine capability.

        Args:
            engine: Runtime engine messaging capability exposed to this agent.
            bind_id: Optional pre-bound agent id, primarily for tests.
        """
        super().__init__(engine, bind_id=bind_id)
