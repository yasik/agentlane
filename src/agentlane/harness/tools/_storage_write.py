"""Wait for storage writes to settle before propagating cancellation."""

import asyncio

from agentlane.harness.filesystem import FileWriter
from agentlane.io import write_all


async def write_content(writer: FileWriter, path: str, content: bytes) -> None:
    """Wait for a started write to settle even when the handler is cancelled."""
    task = asyncio.create_task(asyncio.to_thread(_write, writer, path, content))
    cancelled = False

    # Cancelling the await cannot stop the worker's commit. Keep waiting, even
    # after repeated cancellation, so the caller cannot release a lock too early.
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            # Retrieve the failure below, after respecting any earlier cancellation.
            break

    if cancelled:
        # Consume any worker error before propagating cancellation to the caller.
        if not task.cancelled():
            task.exception()

        raise asyncio.CancelledError

    task.result()


def _write(writer: FileWriter, path: str, content: bytes) -> None:
    # The context exit may commit buffered data; completion includes that step.
    with writer.open_write(path) as stream:
        write_all(stream, content)
