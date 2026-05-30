"""Shared cancellation helpers for harness streaming and lifecycle."""

import asyncio
from collections.abc import Callable

from agentlane.runtime import CancellationToken


def cancellation_relay_task(
    *,
    source: CancellationToken | None,
    target: CancellationToken,
) -> asyncio.Task[None] | None:
    """Relay cancellation from one token into another when needed."""
    if source is None:
        return None

    return asyncio.create_task(
        _relay_cancellation(source=source, target=target),
    )


async def _relay_cancellation(
    *,
    source: CancellationToken,
    target: CancellationToken,
) -> None:
    """Wait for one token to cancel and propagate it to another."""
    await source.wait_cancelled()
    target.cancel()


def cancel_task_callback(task: asyncio.Task[None]) -> Callable[[], None]:
    """Return a cleanup callback that cancels the provided task."""

    def cancel_task() -> None:
        task.cancel()

    return cancel_task
