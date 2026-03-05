"""Agent factory and instance registry."""

import inspect
from collections.abc import Awaitable, Callable

from agentlane.agents import Agent
from agentlane.messaging import AgentId, AgentKey, AgentType

AgentInstance = Agent
AgentFactory = Callable[[], AgentInstance | Awaitable[AgentInstance]]


class AgentRegistry:
    """Maps agent types to factories and caches live instances by AgentId."""

    def __init__(self) -> None:
        """Initialize empty registry state."""
        self._factories: dict[AgentType, AgentFactory] = {}
        self._instances: dict[AgentId, AgentInstance] = {}

    def register_factory(self, agent_type: AgentType, factory: AgentFactory) -> None:
        """Register a factory for a logical agent type."""
        self._factories[agent_type] = factory

    def register_instance(self, agent_id: AgentId, instance: AgentInstance) -> None:
        """Register a concrete instance for an agent id."""
        self._instances[agent_id] = instance

    def has_instance(self, agent_id: AgentId) -> bool:
        """Return whether an instance is active for the id."""
        return agent_id in self._instances

    async def get_or_create(self, agent_id: AgentId) -> AgentInstance:
        """Return an existing instance or lazily create one from factory."""
        if agent_id in self._instances:
            return self._instances[agent_id]

        if agent_id.type not in self._factories:
            raise LookupError(
                f"No factory registered for agent type '{agent_id.type.value}'."
            )

        instance_or_awaitable = self._factories[agent_id.type]()
        if inspect.isawaitable(instance_or_awaitable):
            instance = await instance_or_awaitable
        else:
            instance = instance_or_awaitable

        self._instances[agent_id] = instance
        return instance

    def resolve_agent_id(
        self, agent_type: AgentType, key: AgentKey | None = None
    ) -> AgentId:
        """Resolve a target id for type-only or explicit-key addressing."""
        if key is not None:
            return AgentId(type=agent_type, key=key)

        candidates = [
            agent_id for agent_id in self._instances if agent_id.type == agent_type
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 0:
            return AgentId(type=agent_type, key=AgentKey("default"))

        raise LookupError(
            f"Ambiguous keyless target for agent type '{agent_type.value}'. "
            f"Active keys: {[candidate.key.value for candidate in candidates]}"
        )
