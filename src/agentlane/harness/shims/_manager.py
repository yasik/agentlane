"""Private shim-session manager for one bound harness agent."""

from collections.abc import Sequence
from typing import Any, Self

from agentlane.models import MessageDict, ModelResponse
from agentlane.models.run import RunContext

from .._cancellation import raise_cleanup_errors
from .._hooks import RunnerHooks
from .._run import RunResult, RunState
from ._base import BoundShim, Shim
from ._types import PreparedTurn, ShimBindingContext


class BoundShimManager:
    """Ordered bound shim sessions for one concrete agent instance."""

    def __init__(self, sessions: tuple[BoundShim, ...]) -> None:
        self._sessions = sessions
        self._started_sessions: list[BoundShim] = []
        self._runner_hooks = _collect_runner_hooks(sessions)

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

    @property
    def runner_hooks(self) -> tuple[RunnerHooks, ...]:
        """Return shim-contributed runner hooks in descriptor order."""
        return self._runner_hooks

    async def on_run_start(
        self,
        state: RunState,
        transient_state: RunContext[Any],
    ) -> None:
        """Notify bound shims that one run has started."""
        self._started_sessions = []
        for session in self._sessions:
            self._started_sessions.append(session)
            await session.on_run_start(state, transient_state)

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        """Let bound shims mutate the prepared turn in order."""
        for session in self._sessions:
            await session.prepare_turn(turn)
        turn.apply_tool_exclusions()

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
        sessions, self._started_sessions = self._started_sessions, []
        errors: list[BaseException] = []
        for session in sessions:
            try:
                await session.on_run_end(result, transient_state)
            except BaseException as exc:
                errors.append(exc)

        if errors:
            raise_cleanup_errors("Shim cleanup failed.", errors)


def _collect_runner_hooks(
    sessions: tuple[BoundShim, ...],
) -> tuple[RunnerHooks, ...]:
    """Collect all shim-contributed runner hooks in session order."""
    hooks: list[RunnerHooks] = []
    for session in sessions:
        hooks.extend(session.runner_hooks())
    return tuple(hooks)
