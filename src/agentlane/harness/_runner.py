"""Runner contract for the harness agent loop."""

import abc

from agentlane.models import MessageDict
from agentlane.runtime import CancellationToken

from ._agent import Agent
from ._hooks import RunnerHooks


class Runner(abc.ABC):
    """Stateless runner contract for harness agents."""

    @abc.abstractmethod
    async def run(
        self,
        agent: Agent,
        messages: list[MessageDict],
        *,
        hooks: RunnerHooks | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> list[object]:
        """Execute one harness agent loop.

        Args:
            agent: Agent instance whose loop should run.
            messages: Canonical conversation input for the current turn.
            hooks: Optional lifecycle hooks for observability and testing.
            cancellation_token: Optional cooperative cancellation token.

        Returns:
            list[object]: Runner results to be repackaged by the agent.
        """
        raise NotImplementedError("Runner.run must be implemented by subclasses.")
