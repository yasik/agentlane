"""Agent factory and instance registry."""

import inspect
from asyncio import Future, get_running_loop
from collections.abc import Awaitable, Callable
from threading import RLock

from agentlane.agents import Agent
from agentlane.messaging import AgentId, AgentKey, AgentType

from ._shared import Engine

type AgentFactory = Callable[[Engine], Agent | Awaitable[Agent]]
"""Factory signature for creating one agent bound to an engine."""


class AgentRegistry:
    """Maps agent types to factories and caches live instances by AgentId."""

    def __init__(self) -> None:
        """Initialize empty registry state."""
        self._lock = RLock()
        self._factories: dict[AgentType, AgentFactory] = {}
        self._instances: dict[AgentId, Agent] = {}
        self._creation_futures: dict[AgentId, Future[Agent]] = {}

    def register_factory(self, agent_type: AgentType, factory: AgentFactory) -> None:
        """Register a factory for a logical agent type.

        Factories must always accept the runtime engine capability.
        """
        with self._lock:
            self._factories[agent_type] = factory

    def register_instance(self, agent_id: AgentId, instance: Agent) -> None:
        """Register a concrete instance for an agent id."""
        with self._lock:
            self._instances[agent_id] = instance

    def has_instance(self, agent_id: AgentId) -> bool:
        """Return whether an instance is active for the id."""
        with self._lock:
            return agent_id in self._instances

    async def get_or_create(self, agent_id: AgentId, *, engine: Engine) -> Agent:
        """Return an existing instance or lazily create one from factory."""
        with self._lock:
            existing_instance = self._instances.get(agent_id)
            if existing_instance is not None:
                return existing_instance

            factory = self._factories.get(agent_id.type)
            if factory is None:
                raise LookupError(
                    f"No factory registered for agent type '{agent_id.type.value}'."
                )

            # If another caller is already creating this agent, we join that
            # in-flight creation instead of running the factory twice.
            creation_future = self._creation_futures.get(agent_id)
            if creation_future is None:
                creation_future = get_running_loop().create_future()
                self._creation_futures[agent_id] = creation_future
                # This caller owns creation and must complete the shared future.
                should_create = True
            else:
                # Another caller owns creation; wait for its outcome.
                should_create = False

        if not should_create:
            return await creation_future

        try:
            # Run factory outside lock to avoid blocking unrelated registry operations.
            instance_or_awaitable = factory(engine)
            if inspect.isawaitable(instance_or_awaitable):
                created_instance = await instance_or_awaitable
            else:
                created_instance = instance_or_awaitable
        except Exception as exc:
            with self._lock:
                # Publish the same exception to all waiters and clear in-flight state.
                pending_future = self._creation_futures.pop(agent_id, None)
                if pending_future is not None and not pending_future.done():
                    pending_future.set_exception(exc)
            raise

        with self._lock:
            self._instances[agent_id] = created_instance
            # Wake up all joiners with the single created instance.
            pending_future = self._creation_futures.pop(agent_id, None)
            if pending_future is not None and not pending_future.done():
                pending_future.set_result(created_instance)

        return created_instance

    def resolve_agent_id(self, agent_type: AgentType) -> AgentId:
        """Resolve a target id for type-only addressing."""
        with self._lock:
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
