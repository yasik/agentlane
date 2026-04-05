"""Default harness agent primitive."""

from typing import Any

from pydantic import BaseModel

from agentlane.messaging import AgentId, MessageContext
from agentlane.models import (
    Model,
    ModelResponse,
    OutputSchema,
    PromptSpec,
    Tools,
)
from agentlane.runtime import CancellationToken, Engine, on_message

from ._hooks import RunnerHooks
from ._lifecycle import AgentDescriptor, AgentLifecycle
from ._run import RunInput, RunResult, RunState
from ._runner import Runner
from ._task import Task


class Agent(Task):
    """Default harness agent primitive.

    This phase owns resumable run lifecycle, queued run inputs, and the default
    runner entry. Tool execution and handoffs land in later phases.
    """

    def __init__(
        self,
        engine: Engine,
        runner: Runner,
        *,
        bind_id: AgentId | None = None,
        descriptor: AgentDescriptor | None = None,
        run_state: RunState | None = None,
        hooks: RunnerHooks | None = None,
    ) -> None:
        """Initialize an agent bound to one runtime engine capability.

        Args:
            engine: Runtime engine messaging capability exposed to this agent.
            runner: Stateless runner responsible for each conversation turn.
            bind_id: Optional pre-bound agent id, primarily for tests.
            descriptor: Optional static agent descriptor. When omitted, the
                agent uses a default descriptor with its class name.
            run_state: Optional recovered run state for this concrete agent
                instance. When provided, later turns continue from that exact
                resumable state instead of starting a new run.
            hooks: Optional runner hooks for observability and tests.
        """
        super().__init__(engine, bind_id=bind_id)
        self._runner = runner
        self._hooks = hooks
        self._descriptor = descriptor or AgentDescriptor(name=type(self).__name__)
        self._lifecycle = AgentLifecycle(
            descriptor=self._descriptor,
            run_state=run_state,
        )

    @property
    def name(self) -> str:
        """Return the human-readable agent name."""
        return self._descriptor.name

    @property
    def description(self) -> str | None:
        """Return the short agent description."""
        return self._descriptor.description

    @property
    def model(self) -> Model[ModelResponse] | None:
        """Return the canonical model client for the default runner."""
        return self._descriptor.model

    @property
    def model_args(self) -> dict[str, Any] | None:
        """Return model request arguments forwarded to the model call."""
        return self._descriptor.model_args

    @property
    def schema(self) -> type[BaseModel] | OutputSchema[Any] | None:
        """Return the structured-output schema forwarded to the model."""
        return self._descriptor.schema

    @property
    def instructions(self) -> str | PromptSpec[Any] | None:
        """Return the configured instructions source."""
        return self._descriptor.instructions

    @property
    def tools(self) -> Tools | None:
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
    def pending_input_count(self) -> int:
        """Return the number of pending run inputs queued for later turns."""
        return self._lifecycle.pending_input_count

    @property
    def run_state(self) -> RunState | None:
        """Return a snapshot of the current resumable run state."""
        return self._lifecycle.run_state_snapshot()

    async def _enqueue_input(
        self,
        run_input: RunInput,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        """Queue one internal run input and wait for the final result."""
        return await self._lifecycle.enqueue_input(
            agent=self,
            runner=self._runner,
            hooks=self._hooks,
            run_input=run_input,
            cancellation_token=cancellation_token,
        )

    @on_message
    async def handle_str(self, payload: str, context: MessageContext) -> object:
        """Handle one inbound string run input."""
        _ = context
        return await self._enqueue_input(payload)

    @on_message
    async def handle_list(
        self,
        payload: list[object],
        context: MessageContext,
    ) -> object:
        """Handle one inbound list-based run input."""
        _ = context
        return await self._enqueue_input(list(payload))

    @on_message
    async def handle_run_state(
        self,
        payload: RunState,
        context: MessageContext,
    ) -> object:
        """Handle one inbound resumable run state."""
        _ = context
        return await self._enqueue_input(payload)
