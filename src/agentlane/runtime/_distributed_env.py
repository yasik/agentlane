"""Distributed runtime environment placeholder."""

from ._types import DeliveryTask


class DistributedEnvironment:
    """Stub for future multi-node execution; currently rejects delivery submission."""

    async def start(self) -> None:
        """Start distributed environment resources."""
        return

    async def stop(self) -> None:
        """Stop distributed environment resources."""
        return

    async def stop_when_idle(self) -> None:
        """Drain distributed environment work."""
        return

    async def submit(self, task: DeliveryTask) -> None:
        """Reject task submission until transport/placement is implemented."""
        raise NotImplementedError(
            "DistributedEnvironment is not implemented yet. "
            "Use SingleThreadedEnvironment for v1 execution."
        )
