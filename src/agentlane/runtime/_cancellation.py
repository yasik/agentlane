"""Cooperative cancellation primitive shared by runtime and extensions."""

import asyncio
from typing import Any


class CancellationToken:
    """Cooperative cancellation token for runtime, model, and tool execution."""

    def __init__(self) -> None:
        """Create a token in non-cancelled state."""
        self._event = asyncio.Event()
        self._linked_futures: set[asyncio.Future[Any]] = set()

    @property
    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()

    async def wait_cancelled(self) -> None:
        """Block until cancellation is requested."""
        await self._event.wait()

    def link_future(self, future: asyncio.Future[Any]) -> None:
        """Cancel the future when this token is cancelled."""
        if self.is_cancelled:
            future.cancel()
            return
        self._linked_futures.add(future)
        future.add_done_callback(self._linked_futures.discard)

    def cancel(self) -> None:
        """Request cancellation."""
        self._event.set()
        for future in tuple(self._linked_futures):
            future.cancel()
