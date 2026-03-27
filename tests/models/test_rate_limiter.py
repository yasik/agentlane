"""Tests for rate limiter implementations."""

import asyncio
import time
from collections.abc import Coroutine
from typing import Any

from agentlane.models import (
    CompositeRateLimiter,
    ConcurrentRequestLimiter,
    SlidingWindowRateLimiter,
)

def run_async[T](awaitable: Coroutine[Any, Any, T]) -> T:
    """Run an awaitable inside a fresh event loop for sync pytest tests."""
    return asyncio.run(awaitable)


class TestSlidingWindowRateLimiter:
    """Tests for SlidingWindowRateLimiter."""

    def test_allows_operations_within_limit(self) -> None:
        """Operations within the rate window should proceed immediately."""
        async def exercise() -> float:
            limiter = SlidingWindowRateLimiter(max_operations=5, window_seconds=1.0)
            start = time.monotonic()
            for _ in range(5):
                await limiter.acquire()
            return time.monotonic() - start

        elapsed = run_async(exercise())

        assert elapsed < 0.1

    def test_blocks_when_limit_exceeded(self) -> None:
        """Operations beyond the rate limit should wait for the window to slide."""
        async def exercise() -> float:
            limiter = SlidingWindowRateLimiter(max_operations=3, window_seconds=0.5)
            for _ in range(3):
                await limiter.acquire()
            start = time.monotonic()
            await limiter.acquire()
            return time.monotonic() - start

        elapsed = run_async(exercise())

        assert 0.4 < elapsed < 0.7

    def test_concurrent_acquires(self) -> None:
        """Concurrent acquires should synchronize correctly."""
        async def exercise() -> float:
            limiter = SlidingWindowRateLimiter(max_operations=5, window_seconds=1.0)
            tasks = [limiter.acquire() for _ in range(10)]
            start = time.monotonic()
            await asyncio.gather(*tasks)
            return time.monotonic() - start

        elapsed = run_async(exercise())

        assert 0.9 < elapsed < 1.3


class TestConcurrentRequestLimiter:
    """Tests for ConcurrentRequestLimiter."""

    def test_allows_concurrent_requests_within_limit(self) -> None:
        """Requests within the concurrency limit should proceed."""
        async def exercise() -> None:
            limiter = ConcurrentRequestLimiter(max_concurrent=3)
            await limiter.acquire()
            await limiter.acquire()
            await limiter.acquire()

        run_async(exercise())

        assert True

    def test_blocks_when_concurrent_limit_exceeded(self) -> None:
        """Requests beyond the concurrency limit should block."""
        async def exercise() -> bool:
            limiter = ConcurrentRequestLimiter(max_concurrent=2)
            await limiter.acquire()
            await limiter.acquire()
            acquired = False

            async def try_acquire() -> None:
                nonlocal acquired
                await limiter.acquire()
                acquired = True

            task = asyncio.create_task(try_acquire())
            await asyncio.sleep(0.1)
            assert not acquired
            limiter.release()
            await asyncio.sleep(0.1)
            assert acquired
            await task
            return acquired

        assert run_async(exercise()) is True

    def test_release_frees_slot(self) -> None:
        """Release should make a blocked slot available again."""
        async def exercise() -> float:
            limiter = ConcurrentRequestLimiter(max_concurrent=1)
            await limiter.acquire()
            limiter.release()
            start = time.monotonic()
            await limiter.acquire()
            return time.monotonic() - start

        elapsed = run_async(exercise())

        assert elapsed < 0.05


class TestCompositeRateLimiter:
    """Tests for CompositeRateLimiter."""

    def test_acquires_from_all_limiters(self) -> None:
        """Composite acquire should honor all child limiters."""
        async def exercise() -> bool:
            limiter1 = SlidingWindowRateLimiter(max_operations=2, window_seconds=1.0)
            limiter2 = ConcurrentRequestLimiter(max_concurrent=1)
            composite = CompositeRateLimiter([limiter1, limiter2])
            await composite.acquire()
            acquired = False

            async def try_acquire() -> None:
                nonlocal acquired
                await composite.acquire()
                acquired = True

            task = asyncio.create_task(try_acquire())
            await asyncio.sleep(0.1)
            assert not acquired
            limiter2.release()
            await asyncio.sleep(0.1)
            assert acquired
            await task
            return acquired

        assert run_async(exercise()) is True

    def test_empty_limiter_list(self) -> None:
        """An empty composite should allow immediate acquires."""
        async def exercise() -> float:
            composite = CompositeRateLimiter([])
            start = time.monotonic()
            await composite.acquire()
            return time.monotonic() - start

        elapsed = run_async(exercise())

        assert elapsed < 0.01

    def test_enforces_strictest_limit(self) -> None:
        """Composite should effectively enforce the strictest child limiter."""
        async def exercise() -> float:
            limiter1 = SlidingWindowRateLimiter(max_operations=10, window_seconds=1.0)
            limiter2 = SlidingWindowRateLimiter(max_operations=3, window_seconds=1.0)
            composite = CompositeRateLimiter([limiter1, limiter2])
            for _ in range(3):
                await composite.acquire()
            start = time.monotonic()
            await composite.acquire()
            return time.monotonic() - start

        elapsed = run_async(exercise())

        assert 0.9 < elapsed < 1.3

    def test_high_concurrency_no_hang(self) -> None:
        """High concurrency should not deadlock the sliding-window limiter."""
        async def exercise() -> float:
            limiter = SlidingWindowRateLimiter(max_operations=5, window_seconds=0.5)
            tasks = [limiter.acquire() for _ in range(50)]
            start = time.monotonic()
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=10.0)
            return time.monotonic() - start

        elapsed = run_async(exercise())

        assert elapsed < 8.0, (
            f"Test took {elapsed}s, likely hanging due to lock bug"
        )
        assert elapsed > 3.5, (
            f"Test completed too fast ({elapsed}s), rate limiting may be broken"
        )


class TestRateLimiterContextManager:
    """Tests for context-manager behavior across rate limiters."""

    def test_sliding_window_context_manager(self) -> None:
        """SlidingWindowRateLimiter should support async context-manager usage."""
        async def exercise() -> None:
            limiter = SlidingWindowRateLimiter(max_operations=2, window_seconds=1.0)
            async with limiter:
                pass
            async with limiter:
                pass

        run_async(exercise())

    def test_concurrent_limiter_context_manager_releases(self) -> None:
        """ConcurrentRequestLimiter should release on context-manager exit."""
        async def exercise() -> bool:
            limiter = ConcurrentRequestLimiter(max_concurrent=1)
            acquired = False
            async with limiter:
                async def try_acquire() -> None:
                    nonlocal acquired
                    async with limiter:
                        acquired = True

                task = asyncio.create_task(try_acquire())
                await asyncio.sleep(0.1)
                assert not acquired
            await asyncio.sleep(0.1)
            assert acquired
            await task
            return acquired

        assert run_async(exercise()) is True

    def test_concurrent_limiter_releases_on_exception(self) -> None:
        """ConcurrentRequestLimiter should release even if the block raises."""
        async def exercise() -> float:
            limiter = ConcurrentRequestLimiter(max_concurrent=1)
            try:
                async with limiter:
                    raise ValueError("Test exception")
            except ValueError:
                pass
            start = time.monotonic()
            async with limiter:
                pass
            return time.monotonic() - start

        elapsed = run_async(exercise())

        assert elapsed < 0.05

    def test_composite_context_manager(self) -> None:
        """CompositeRateLimiter should support nested async context-manager usage."""
        async def exercise() -> bool:
            limiter1 = SlidingWindowRateLimiter(max_operations=5, window_seconds=1.0)
            limiter2 = ConcurrentRequestLimiter(max_concurrent=2)
            composite = CompositeRateLimiter([limiter1, limiter2])
            acquired = False
            async with composite:
                async with composite:
                    async def try_acquire() -> None:
                        nonlocal acquired
                        async with composite:
                            acquired = True

                    task = asyncio.create_task(try_acquire())
                    await asyncio.sleep(0.1)
                    assert not acquired
                await asyncio.sleep(0.1)
                assert acquired
                await task
            return acquired

        assert run_async(exercise()) is True

    def test_composite_releases_on_exception(self) -> None:
        """CompositeRateLimiter should release child limiters on exception."""
        async def exercise() -> float:
            limiter1 = SlidingWindowRateLimiter(max_operations=5, window_seconds=1.0)
            limiter2 = ConcurrentRequestLimiter(max_concurrent=1)
            composite = CompositeRateLimiter([limiter1, limiter2])
            try:
                async with composite:
                    raise ValueError("Test exception")
            except ValueError:
                pass
            start = time.monotonic()
            async with composite:
                pass
            return time.monotonic() - start

        elapsed = run_async(exercise())

        assert elapsed < 0.05

    def test_composite_cleanup_on_partial_failure(self) -> None:
        """CompositeRateLimiter should clean up earlier limiters on enter failure."""
        class FailingLimiter:
            async def acquire(self) -> None:
                """Provide a no-op acquire implementation."""

            async def __aenter__(self) -> "FailingLimiter":
                """Raise when entering to simulate a downstream failure."""
                raise RuntimeError("Enter failed")

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: object | None,
            ) -> None:
                """Provide a no-op exit implementation."""

            def record_usage(self, tokens: int) -> None:
                """Match the RateLimiter protocol."""

        async def exercise() -> float:
            limiter1 = ConcurrentRequestLimiter(max_concurrent=1)
            limiter2 = FailingLimiter()
            composite = CompositeRateLimiter([limiter1, limiter2])
            try:
                async with composite:
                    pass
            except RuntimeError:
                pass
            start = time.monotonic()
            async with limiter1:
                pass
            return time.monotonic() - start

        elapsed = run_async(exercise())

        assert elapsed < 0.05
