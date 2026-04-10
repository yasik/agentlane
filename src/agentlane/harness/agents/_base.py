"""Abstract base contract for developer-facing harness agents."""

import abc

from agentlane.runtime import CancellationToken

from .._run import RunInput, RunResult


class AgentBase(abc.ABC):
    """Abstract base class for ergonomic harness agent wrappers.

    This contract is intentionally small. Future high-level agent wrappers
    should expose at least two execution paths:

    1. ``run(...)`` for the main stateful conversation line
    2. ``fork(...)`` for a one-off branch that does not mutate that line
    """

    @abc.abstractmethod
    async def run(
        self,
        input: RunInput,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        """Run or continue the primary conversation line."""
        raise NotImplementedError

    @abc.abstractmethod
    async def fork(
        self,
        input: RunInput,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        """Run one branch without mutating the primary conversation line."""
        raise NotImplementedError
