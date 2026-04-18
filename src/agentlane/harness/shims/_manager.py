"""Private shim-session manager for one bound harness agent."""

from collections.abc import Sequence
from typing import Any, Self

from agentlane.models import MessageDict, ModelResponse
from agentlane.models.run import RunContext

from .._run import RunResult, RunState
from ._base import BoundShim, Shim
from ._types import PreparedTurn, ShimBindingContext


class BoundShimManager:
    """Ordered bound shim sessions for one concrete agent instance."""

    def __init__(self, sessions: tuple[BoundShim, ...]) -> None:
        self._sessions = sessions

    @classmethod
    async def bind(
        cls,
        *,
        shims: Sequence[Shim] | None,
        context: ShimBindingContext,
    ) -> Self:
        """Bind all declared shim definitions in descriptor order."""
        if not shims:
            return cls(())

        sessions: list[BoundShim] = []
        for shim in shims:
            sessions.append(await shim.bind(context))
        return cls(tuple(sessions))

    @property
    def sessions(self) -> tuple[BoundShim, ...]:
        """Return the bound shim sessions in execution order."""
        return self._sessions

    async def on_run_start(
        self,
        state: RunState,
        transient_state: RunContext[Any],
    ) -> None:
        """Notify bound shims that one run has started."""
        for session in self._sessions:
            await session.on_run_start(state, transient_state)

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        """Let bound shims mutate the prepared turn in order."""
        for session in self._sessions:
            await session.prepare_turn(turn)

    async def transform_messages(
        self,
        turn: PreparedTurn,
        messages: list[MessageDict],
    ) -> list[MessageDict]:
        """Apply ordered optional message transformations."""
        transformed_messages = messages
        for session in self._sessions:
            replacement = await session.transform_messages(turn, transformed_messages)
            if replacement is not None:
                transformed_messages = replacement
        return transformed_messages

    async def on_model_response(
        self,
        turn: PreparedTurn,
        response: ModelResponse,
    ) -> None:
        """Notify bound shims after one model response completes."""
        for session in self._sessions:
            await session.on_model_response(turn, response)

    async def on_run_end(
        self,
        result: RunResult | None,
        transient_state: RunContext[Any],
    ) -> None:
        """Notify bound shims that one run has ended."""
        for session in self._sessions:
            await session.on_run_end(result, transient_state)
