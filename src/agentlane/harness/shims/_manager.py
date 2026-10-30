"""Private shim-session manager for one bound harness agent."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Self

from agentlane.models import MessageDict, ModelResponse
from agentlane.models.run import RunContext

from .._cancellation import raise_cleanup_errors
from .._hooks import RunnerHooks
from .._run import RunHistoryItem, RunResult, RunState, copy_history_item
from ._base import (
    BoundShim,
    DelegatingShim,
    Shim,
    ShimBindingContext,
    ToolSourceBinding,
)
from ._types import PreparedTurn


class _InheritedShim(DelegatingShim):
    """Bind the original wrapper chain with restricted source factories."""

    def __init__(
        self, definition: Shim, bindings: tuple[ToolSourceBinding, ...]
    ) -> None:
        # A grandchild must use its new limits, not an older adapter's limits.
        if isinstance(definition, _InheritedShim):
            definition = definition.inner
        super().__init__(definition)
        self._bindings = bindings

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        return await self.inner.bind(
            replace(context, tool_source_bindings=self._bindings)
        )


class BoundShimManager:
    """Ordered bound shim sessions for one concrete agent instance."""

    def __init__(
        self,
        sessions: tuple[BoundShim, ...],
        definitions: tuple[Shim, ...] = (),
    ) -> None:
        self._sessions = sessions
        self._definitions = definitions
        self._started_sessions: list[BoundShim] = []
        self._original_history: list[RunHistoryItem] | None = None
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
        return cls(tuple(sessions), tuple(shims))

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
        self._original_history = [
            copy_history_item(item) for item in turn.run_state.history
        ]
        try:
            for session in self._sessions:
                await session.prepare_turn(turn)
            turn.apply_tool_exclusions()
        except BaseException as error:
            turn.replace_history(self._original_history)
            self._original_history = None
            await self._notify_request_prepared(turn, [], error)
            raise

    async def transform_messages(
        self,
        turn: PreparedTurn,
        messages: list[MessageDict],
    ) -> list[MessageDict]:
        """Apply ordered optional message transformations."""
        transformed_messages = messages
        try:
            for session in self._sessions:
                replacement = await session.transform_messages(
                    turn, transformed_messages
                )
                if replacement is not None:
                    transformed_messages = replacement

            # Validation sees all injected context and the final tool catalog.
            for session in self._sessions:
                await session.validate_request(turn, transformed_messages)
        except BaseException as error:
            if self._original_history is not None:
                turn.replace_history(self._original_history)
            await self._notify_request_prepared(turn, transformed_messages, error)
            raise
        finally:
            self._original_history = None

        await self._notify_request_prepared(turn, transformed_messages, None)
        return transformed_messages

    async def _notify_request_prepared(
        self,
        turn: PreparedTurn,
        messages: list[MessageDict],
        error: BaseException | None,
    ) -> None:
        # These observers cannot change whether validation accepted the request.
        for session in self._sessions:
            try:
                await session.on_request_prepared(turn, messages, error)
            except Exception as observer_error:
                _ = observer_error

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

    def child_shims(
        self,
        names: frozenset[str],
        *,
        inherit_definitions: bool,
    ) -> tuple[Shim, ...]:
        """Rebind dynamic tool sources without copying executable closures."""
        shims: list[Shim] = []
        for definition, session in zip(self._definitions, self._sessions, strict=True):
            bindings = session.inherit_tools(names)
            if bindings:
                shims.append(_InheritedShim(definition, bindings))
            elif bindings is not None:
                continue
            elif inherit_definitions:
                shims.append(definition)
        return tuple(shims)


def _collect_runner_hooks(
    sessions: tuple[BoundShim, ...],
) -> tuple[RunnerHooks, ...]:
    """Collect all shim-contributed runner hooks in session order."""
    hooks: list[RunnerHooks] = []
    for session in sessions:
        hooks.extend(session.runner_hooks())
    return tuple(hooks)
