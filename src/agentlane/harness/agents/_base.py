"""Shared contract for agent interfaces.

This module defines the minimal public execution surface for stateful agent
types that sit above the runtime-facing harness agent. Concrete agents may add
their own configuration or lifecycle conveniences, but they should preserve
the same two core execution paths:

1. continue the primary conversation line with ``run(...)``, and
2. create a non-mutating branch with ``fork(...)``.

Stateful agents also need one shared lifecycle control:

3. clear the persisted primary conversation line with ``reset()``.
"""

import abc

from agentlane.runtime import CancellationToken

from .._run import RunInput, RunResult


class AgentBase(abc.ABC):
    """Abstract base class for high-level harness agent interfaces.

    This contract is intentionally small. Implementations represent stateful
    conversation agents that preserve one primary execution line and may also
    expose one-off branches from that line.

    Future high-level agent types should expose at least two execution
    paths:

    1. ``run(...)`` for the main stateful conversation line, and
    2. ``fork(...)`` for a one-off branch that does not mutate that line.

    Stateful agents should also expose ``reset()`` so callers can discard
    the persisted primary state explicitly.
    """

    @abc.abstractmethod
    async def run(
        self,
        input: RunInput,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        """Run or continue the primary conversation line.

        Args:
            input: Raw run input or an explicit ``RunState`` resume payload.
            cancellation_token: Optional shared cancellation token for the run.

        Returns:
            RunResult: The final result for the primary conversation line.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def fork(
        self,
        input: RunInput,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        """Run one branch without mutating the primary conversation line.

        Args:
            input: Raw run input or an explicit ``RunState`` resume payload.
            cancellation_token: Optional shared cancellation token for the run.

        Returns:
            RunResult: The final result for the forked branch.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear the persisted primary conversation line for future runs."""
        raise NotImplementedError
