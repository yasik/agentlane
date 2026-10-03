"""Shared cancellation helpers for harness streaming and lifecycle."""

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any, Never, cast

from agentlane.runtime import CancellationToken


@asynccontextmanager
async def cancellation_scope(
    cancellation_token: CancellationToken | None,
) -> AsyncIterator[None]:
    """Cancel the current operation while preserving its task and context.

    The observer is stopped before leaving the scope so a cancelled token
    cannot interrupt later cleanup in the same task.
    """
    if cancellation_token is None:
        yield
        return

    if cancellation_token.is_cancelled:
        raise asyncio.CancelledError

    owner = asyncio.current_task()
    if owner is None:
        raise RuntimeError("A cancellation scope requires an asyncio task.")

    observer = asyncio.create_task(_cancel_on_token(cancellation_token, owner))
    try:
        yield
    finally:
        observer.cancel()
        # Join the observer without suppressing a new cancellation of the owner.
        await asyncio.gather(observer, return_exceptions=True)


async def _cancel_on_token(token: CancellationToken, owner: asyncio.Task[Any]) -> None:
    """Relay one token cancellation to the task within its active scope."""
    await token.wait_cancelled()
    owner.cancel()


def raise_cleanup_errors(message: str, errors: Sequence[BaseException]) -> Never:
    """Raise collected cleanup failures, preserving cancellation as cancellation."""
    for error in errors:
        cancelled = _find_cancellation(error)
        if cancelled is not None:
            raise cancelled

    raise BaseExceptionGroup(message, errors)


def _find_cancellation(error: BaseException) -> asyncio.CancelledError | None:
    """Return the first cancellation, including from a nested exception group."""
    if isinstance(error, asyncio.CancelledError):
        return error

    if isinstance(error, BaseExceptionGroup):
        group = cast(BaseExceptionGroup[BaseException], error)
        for nested in group.exceptions:
            cancelled = _find_cancellation(nested)
            if cancelled is not None:
                return cancelled

    return None


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
