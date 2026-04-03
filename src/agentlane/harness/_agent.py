"""Default harness agent primitive."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentlane.messaging import AgentId, MessageContext
from agentlane.models import MessageDict
from agentlane.models import Tool as HarnessTool
from agentlane.runtime import CancellationToken, Engine, on_message

from ._hooks import RunnerHooks
from ._lifecycle import AgentDescriptor, AgentLifecycle
from ._runner import Runner
from ._task import Task


@dataclass(slots=True)
class UserMessage:
    """Public runtime payload for one user turn sent to a harness agent."""

    content: object
    """Opaque user content rendered into the conversation history."""


class Agent(Task):
    """Default harness agent primitive.

    This phase owns conversation lifecycle, queued user messages, and default
    runner entry. Tool execution and handoffs land in later phases.
    """

    def __init__(
        self,
        engine: Engine,
        runner: Runner,
        *,
        bind_id: AgentId | None = None,
        name: str | None = None,
        description: str | None = None,
        system_prompt: str | None = None,
        tools: Sequence[HarnessTool[Any, Any]] | None = None,
        skills: Sequence[object] | None = None,
        context: object | None = None,
        memory: object | None = None,
        hooks: RunnerHooks | None = None,
    ) -> None:
        """Initialize an agent bound to one runtime engine capability.

        Args:
            engine: Runtime engine messaging capability exposed to this agent.
            runner: Stateless runner responsible for each conversation turn.
            bind_id: Optional pre-bound agent id, primarily for tests.
            name: Optional human-readable agent name.
            description: Optional short description of the agent purpose.
            system_prompt: Optional system prompt used to seed new conversations.
            tools: Optional tool definitions reserved for later phases.
            skills: Optional skill descriptors reserved for later phases.
            context: Optional context reference reserved for later phases.
            memory: Optional memory reference reserved for later phases.
            hooks: Optional runner hooks for observability and tests.
        """
        super().__init__(engine, bind_id=bind_id)
        self._runner = runner
        self._hooks = hooks
        self._descriptor = AgentDescriptor(
            name=name or type(self).__name__,
            description=description,
            system_prompt=system_prompt,
            tools=tuple(tools) if tools is not None else None,
            skills=tuple(skills) if skills is not None else None,
            context=context,
            memory=memory,
        )
        self._lifecycle = AgentLifecycle(descriptor=self._descriptor)

    @property
    def name(self) -> str:
        """Return the human-readable agent name."""
        return self._descriptor.name

    @property
    def description(self) -> str | None:
        """Return the short agent description."""
        return self._descriptor.description

    @property
    def system_prompt(self) -> str | None:
        """Return the configured system prompt."""
        return self._descriptor.system_prompt

    @property
    def tools(self) -> tuple[HarnessTool[Any, Any], ...] | None:
        """Return the configured tools for this agent."""
        return self._descriptor.tools

    @property
    def skills(self) -> tuple[object, ...] | None:
        """Return the configured skills for this agent."""
        return self._descriptor.skills

    @property
    def context(self) -> object | None:
        """Return the opaque context reference for this agent."""
        return self._descriptor.context

    @property
    def memory(self) -> object | None:
        """Return the opaque memory reference for this agent."""
        return self._descriptor.memory

    @property
    def is_running(self) -> bool:
        """Return whether the agent is currently executing a runner turn."""
        return self._lifecycle.is_running

    @property
    def pending_user_message_count(self) -> int:
        """Return the number of user messages queued for later turns."""
        return self._lifecycle.pending_message_count

    @property
    def message_history(self) -> list[MessageDict]:
        """Return a snapshot of the current canonical conversation history."""
        return self._lifecycle.history_snapshot()

    @classmethod
    def user_message(cls, content: object) -> UserMessage:
        """Return one default harness payload for a user turn."""
        _ = cls
        return UserMessage(content=content)

    async def _enqueue_user_message(
        self,
        content: object,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> object:
        """Queue one internal user message and wait for the packaged result."""
        return await self._lifecycle.enqueue_user_message(
            agent=self,
            runner=self._runner,
            hooks=self._hooks,
            content=content,
            cancellation_token=cancellation_token,
        )

    @on_message
    async def handle(self, payload: UserMessage, context: MessageContext) -> object:
        """Handle one inbound runtime user message."""
        _ = context
        # The runtime-facing payload is public, but queueing stays inside the
        # agent so later phases can evolve lifecycle behavior behind one seam.
        return await self._enqueue_user_message(payload.content)
