"""Cancellation primitive shared between runtime and handlers."""

import asyncio


class CancellationToken:
    """Cooperative cancellation token exposed to handlers through MessageContext."""

    def __init__(self) -> None:
        """Create a token in non-cancelled state."""
        self._event = asyncio.Event()

    @property
    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()

    async def wait_cancelled(self) -> None:
        """Block until cancellation is requested."""
        await self._event.wait()

    def cancel(self) -> None:
        """Request cancellation."""
        self._event.set()
