"""Runtime environment interfaces."""

from typing import Protocol

from ._types import DeliveryTask


class RuntimeEnvironment(Protocol):
    """Execution backend contract used by the facade to run scheduled delivery tasks."""

    async def start(self) -> None:
        """Start environment workers/resources."""

    async def stop(self) -> None:
        """Stop environment immediately."""

    async def stop_when_idle(self) -> None:
        """Drain pending work and stop."""

    async def submit(self, task: DeliveryTask) -> None:
        """Submit one delivery task."""
