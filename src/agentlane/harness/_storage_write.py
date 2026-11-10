"""Wait for storage writes to settle before propagating cancellation."""

import asyncio

from agentlane.harness.filesystem import FileRevisions, FileWriter
from agentlane.io import write_all


async def write_content(
    writer: FileWriter, path: str, content: bytes, *, revision: str | None = None
) -> None:
    """Wait for a started write to settle even when the handler is cancelled.

    With a revision, the write replaces the file only while it still has that
    revision; the writer must then implement `FileRevisions`.
    """
    task = asyncio.create_task(
        asyncio.to_thread(_write, writer, path, content, revision)
    )
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


def _write(writer: FileWriter, path: str, content: bytes, revision: str | None) -> None:
    if revision is None:
        opened = writer.open_write(path)
    else:
        assert isinstance(writer, FileRevisions)
        opened = writer.open_write_revision(path, revision)

    # The context exit may commit buffered data; completion includes that step.
    with opened as stream:
        write_all(stream, content)
