"""Base shim contracts for the harness."""

import abc
from typing import Any

from agentlane.models import MessageDict, ModelResponse
from agentlane.models.run import RunContext

from .._run import RunResult, RunState
from ._types import PreparedTurn, ShimBindingContext


class BoundHarnessShim:
    """Per-agent bound shim session.

    Concrete shims may override only the callbacks they need. The default
    implementations are no-ops so simple shims can stay compact.
    """

    async def on_run_start(
        self,
        state: RunState,
        transient_state: RunContext[Any],
    ) -> None:
        """Observe one run start and optionally mutate the working state."""
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
        """Observe one completed model response and update shim state."""
        _ = turn
        _ = response

    async def on_run_end(
        self,
        result: RunResult | None,
        transient_state: RunContext[Any],
    ) -> None:
        """Observe the end of one run."""
        _ = result
        _ = transient_state


class HarnessShim(abc.ABC):
    """Definition-time contract for one harness shim."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the stable shim name used for persisted state keys."""
        raise NotImplementedError

    @abc.abstractmethod
    async def bind(self, context: ShimBindingContext) -> BoundHarnessShim:
        """Bind the shim to one concrete agent instance."""
        raise NotImplementedError
