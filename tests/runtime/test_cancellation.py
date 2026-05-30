"""Tests for shared runtime cancellation semantics."""

import asyncio

from agentlane.runtime import CancellationToken


def test_cancellation_token_cancels_linked_future() -> None:
    """Cancelling the shared token should cancel futures linked by model clients."""
    event_loop = asyncio.new_event_loop()
    try:
        future = event_loop.create_future()
        token = CancellationToken()

        token.link_future(future)
        token.cancel()

        assert future.cancelled()
        assert token.is_cancelled
    finally:
        event_loop.close()
