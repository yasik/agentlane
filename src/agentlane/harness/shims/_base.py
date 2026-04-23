"""Base shim contracts for the harness."""

import abc
from typing import Any

from agentlane.models import MessageDict, ModelResponse
from agentlane.models.run import RunContext

from .._hooks import RunnerHooks
from .._run import RunResult, RunState
from ._types import PreparedTurn, ShimBindingContext


class BoundShim:
    """Per-agent bound shim session.

    Concrete shims may override only the callbacks they need. The default
    implementations are no-ops so simple shims can stay compact.
    """

    async def on_run_start(
        self,
        state: RunState,
        transient_state: RunContext[Any],
    ) -> None:
        """Handle one run start and optionally mutate the working state."""
        _ = state
        _ = transient_state

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        """Mutate the prepared turn before one model request is built."""
        _ = turn

    async def transform_messages(
        self,
        turn: PreparedTurn,
        messages: list[MessageDict],
    ) -> list[MessageDict] | None:
        """Optionally replace the canonical message list for one turn."""
        _ = turn
        _ = messages
        return None

    async def on_model_response(
        self,
        turn: PreparedTurn,
        response: ModelResponse,
    ) -> None:
        """Handle one completed model response and update shim state."""
        _ = turn
        _ = response

    async def on_run_end(
        self,
        result: RunResult | None,
        transient_state: RunContext[Any],
    ) -> None:
        """Handle the end of one run."""
        _ = result
        _ = transient_state

    def runner_hooks(self) -> tuple[RunnerHooks, ...]:
        """Return additional hooks for this bound shim session."""
        return ()


class _ForwardingBoundShim(BoundShim):
    """Default bound adapter that forwards callbacks to one shim definition."""

    def __init__(self, shim: "Shim") -> None:
        self._shim = shim

    async def on_run_start(
        self,
        state: RunState,
        transient_state: RunContext[Any],
    ) -> None:
        await self._shim.on_run_start(state, transient_state)

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        await self._shim.prepare_turn(turn)

    async def transform_messages(
        self,
        turn: PreparedTurn,
        messages: list[MessageDict],
    ) -> list[MessageDict] | None:
        return await self._shim.transform_messages(turn, messages)

    async def on_model_response(
        self,
        turn: PreparedTurn,
        response: ModelResponse,
    ) -> None:
        await self._shim.on_model_response(turn, response)

    async def on_run_end(
        self,
        result: RunResult | None,
        transient_state: RunContext[Any],
    ) -> None:
        await self._shim.on_run_end(result, transient_state)

    def runner_hooks(self) -> tuple[RunnerHooks, ...]:
        return self._shim.runner_hooks()


class Shim(abc.ABC):
    """Definition-time contract for one harness shim.

    Most shims should subclass this one type only and override whichever
    lifecycle callbacks they need. The default `bind(...)` implementation
    creates a simple forwarding bound session automatically.

    Override `bind(...)` only when the shim needs private per-agent in-memory
    state or custom bind-time setup.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the stable shim name used for persisted state keys."""
        raise NotImplementedError

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        """Bind the shim to one concrete agent instance."""
        _ = context
        return _ForwardingBoundShim(self)

    async def on_run_start(
        self,
        state: RunState,
        transient_state: RunContext[Any],
    ) -> None:
        """Handle one run start and optionally mutate the working state."""
        _ = state
        _ = transient_state

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        """Mutate the prepared turn before one model request is built."""
        _ = turn

    async def transform_messages(
        self,
        turn: PreparedTurn,
        messages: list[MessageDict],
    ) -> list[MessageDict] | None:
        """Optionally replace the canonical message list for one turn."""
        _ = turn
        _ = messages
        return None

    async def on_model_response(
        self,
        turn: PreparedTurn,
        response: ModelResponse,
    ) -> None:
        """Handle one completed model response and update shim state."""
        _ = turn
        _ = response

    async def on_run_end(
        self,
        result: RunResult | None,
        transient_state: RunContext[Any],
    ) -> None:
        """Handle the end of one run."""
        _ = result
        _ = transient_state

    def runner_hooks(self) -> tuple[RunnerHooks, ...]:
        """Return additional runner hooks for this shim definition."""
        return ()
