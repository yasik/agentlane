"""Create child agents and deliver delegated input through the harness runtime."""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from agentlane.messaging import AgentId, DeliveryOutcome
from agentlane.models import ModelStreamEvent, Tools
from agentlane.runtime import CancellationToken, RuntimeEngine

from ._lifecycle import AgentDescriptor, LifecycleRunner
from ._run import RunResult, RunState
from ._snapshot import RunInput
from ._stream import RunStream
from ._stream_base import BaseRunStream
from ._task import Task


@runtime_checkable
class _StreamRunnableAgent(Protocol):
    """Structural protocol for local streamed harness delivery."""

    async def enqueue_input_stream(
        self,
        run_input: RunInput,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> RunStream: ...


class AgentDelivery:
    """Own child registration, input delivery, and child stream cleanup."""

    def __init__(
        self,
        agent: Task,
        runner: LifecycleRunner,
        cancellation_token: CancellationToken | None,
        *,
        child_turn_limits: dict[AgentId, int],
        max_turns: int,
    ) -> None:
        self.agent = agent
        self.runner = runner
        self.cancellation_token = cancellation_token
        self.child_turn_limits = child_turn_limits
        self.max_turns = max_turns

    async def send(
        self,
        *,
        recipient: AgentId,
        descriptor: AgentDescriptor,
        run_input: RunInput,
        parent_tools: Tools | None,
    ) -> DeliveryOutcome:
        """Register the child factory and deliver input through runtime messaging."""
        self._runtime().register_factory(
            recipient.type.value,
            type(self.agent).create_factory(
                runner=self.runner,
                descriptor=descriptor,
                parent_tools=parent_tools,
            ),
        )

        # Keep the parent's turn limit while the child runs.
        self.child_turn_limits[recipient] = self.max_turns
        try:
            return await self.agent.send_message(
                run_input,
                recipient=recipient,
                cancellation_token=self.cancellation_token,
            )
        finally:
            self.child_turn_limits.pop(recipient, None)

    async def stream_handoff(
        self,
        *,
        recipient: AgentId,
        parent_tools: Tools | None,
        descriptor: AgentDescriptor,
        transferred_state: RunState,
        emit: Callable[[ModelStreamEvent], None],
        control_stream: BaseRunStream[Any] | None,
    ) -> RunResult:
        """Transfer the run locally to another agent and stream its events."""
        runtime = self._runtime()
        bound_agent = type(self.agent).bind(
            runtime,
            recipient,
            runner=self.runner,
            descriptor=descriptor,
            parent_tools=parent_tools,
        )
        if not isinstance(bound_agent, _StreamRunnableAgent):
            raise RuntimeError(
                "Streamed handoff requires the child agent to support "
                "`enqueue_input_stream`."
            )

        # Start the child only after checking its streaming capability.
        self.child_turn_limits[recipient] = self.max_turns
        try:
            child_stream = await bound_agent.enqueue_input_stream(
                transferred_state,
                cancellation_token=self.cancellation_token,
            )

            # Route subsequent steering to the child that now owns the run.
            if control_stream is not None:
                control_stream.bind_steering(child_stream.steer)

            # Forward output and always close the child stream before returning.
            try:
                async for event in child_stream:
                    emit(event)

                return await child_stream.result()
            finally:
                await child_stream.aclose()
        finally:
            self.child_turn_limits.pop(recipient, None)

    def _runtime(self) -> RuntimeEngine:
        """Return the runtime engine required for delegated agent routing."""
        engine = getattr(self.agent, "_engine", None)
        if isinstance(engine, RuntimeEngine):
            return engine

        raise RuntimeError(
            "Harness sub-agent delegation requires an engine that supports "
            "runtime factory registration."
        )
