"""Base task primitive for the harness."""

from collections.abc import Callable
from typing import Any, Self

from agentlane.messaging import AgentId, AgentType
from agentlane.runtime import BaseAgent, Engine, RuntimeEngine


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

    @property
    def task_id(self) -> AgentId:
        """Return the runtime-bound task identity."""
        return self.id

    @classmethod
    def create_factory(cls, *args: Any, **kwargs: Any) -> Callable[[Engine], BaseAgent]:
        """Create a runtime factory that instantiates this task type.

        Args:
            *args: Positional constructor arguments after ``engine``.
            **kwargs: Keyword constructor arguments after ``engine``.

        Returns:
            Callable[[Engine], BaseAgent]: Factory compatible with
            ``RuntimeEngine.register_factory``.
        """

        def factory(engine: Engine) -> Task:
            return cls(engine, *args, **kwargs)

        return factory

    @classmethod
    def register(
        cls,
        runtime: RuntimeEngine,
        agent_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> AgentType:
        """Register this task type as a lazy runtime factory.

        Args:
            runtime: Runtime used for delivery and instance activation.
            agent_type: Logical runtime type used for routing and addressing.
            *args: Positional constructor arguments after ``engine``.
            **kwargs: Keyword constructor arguments after ``engine``.

        Returns:
            AgentType: Normalized registered agent type.
        """
        return runtime.register_factory(
            agent_type,
            cls.create_factory(*args, **kwargs),
        )

    @classmethod
    def bind(
        cls,
        runtime: RuntimeEngine,
        agent_id: AgentId,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Instantiate and register one stateful task instance by id.

        Args:
            runtime: Runtime used for delivery and instance registration.
            agent_id: Concrete runtime identity for the stateful task instance.
            *args: Positional constructor arguments after ``engine``.
            **kwargs: Keyword constructor arguments after ``engine``.

        Returns:
            Self: The created and runtime-bound task instance.
        """
        instance = cls(runtime, *args, **kwargs)
        runtime.register_instance(agent_id, instance)
        return instance
