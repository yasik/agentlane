"""Rate limiting utilities for LLM clients.

This module provides composable rate limiting implementations that can be
shared across multiple client instances to enforce global limits.
"""

import asyncio
import time
from collections import deque
from collections.abc import Sequence
from typing import Protocol, Self, runtime_checkable

_DEFAULT_TPM_HEADROOM = 0.7
"""Headroom percentage of the limit to allow for usage."""


@runtime_checkable
class RateLimiter(Protocol):
    """Protocol for rate limiting implementations.

    All rate limiters must implement an async context manager interface
    that acquires permission on enter and releases resources on exit.
    """

    async def acquire(self) -> None:
        """Acquire permission to proceed with an operation.

        This method blocks until the operation can proceed within the
        rate limit constraints. It never raises an exception.
        """

    async def __aenter__(self) -> Self:
        """Enter the rate limiter context.

        Acquires permission to proceed with an operation.
        """
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit the rate limiter context.

        Releases any resources acquired during the operation.
        """

    def record_usage(self, tokens: int) -> None:
        """Record tokens consumed by a completed operation."""


class SlidingWindowRateLimiter:
    """Rate limiter using a sliding window algorithm.

    Tracks timestamps of recent operations and blocks when the limit
    would be exceeded within the time window.

    Thread-safe for use across multiple asyncio tasks.
    """

    def __init__(self, max_operations: int, window_seconds: float) -> None:
        """Initialize the rate limiter.

        Args:
            max_operations: Maximum number of operations allowed in the time window.
            window_seconds: Size of the sliding time window in seconds.
        """
        self._max_operations = max_operations
        self._window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to proceed with an operation.

        Blocks until an operation slot becomes available within the rate limit.
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                cutoff = now - self._window_seconds

                # Remove timestamps outside the window
                while self._timestamps and self._timestamps[0] < cutoff:
                    self._timestamps.popleft()

                # Check if we can proceed
                if len(self._timestamps) < self._max_operations:
                    self._timestamps.append(now)
                    return

                # Calculate wait time until the oldest timestamp expires
                wait_time = self._timestamps[0] + self._window_seconds - now

            # Sleep outside the lock to allow other operations to proceed
            if wait_time > 0:
                await asyncio.sleep(wait_time)

    async def __aenter__(self) -> Self:
        """Enter the rate limiter context."""
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit the rate limiter context.

        For sliding window limiters, there are no resources to release.
        """

    def record_usage(self, tokens: int) -> None:
        """Record tokens consumed by a completed operation."""


class ConcurrentRequestLimiter:
    """Limits the number of concurrent in-flight operations.

    Uses an asyncio Semaphore to enforce a maximum number of
    simultaneous operations.
    """

    def __init__(self, max_concurrent: int) -> None:
        """Initialize the concurrent request limiter.

        Args:
            max_concurrent: Maximum number of concurrent operations allowed.
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def acquire(self) -> None:
        """Acquire permission for a concurrent operation.

        Blocks until a slot becomes available.
        """
        await self._semaphore.acquire()

    def release(self) -> None:
        """Release a concurrent operation slot.

        Must be called after the operation completes to free up capacity.
        """
        self._semaphore.release()

    async def __aenter__(self) -> Self:
        """Enter the rate limiter context."""
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit the rate limiter context.

        Releases the semaphore slot.
        """
        self.release()

    def record_usage(self, tokens: int) -> None:
        """Record tokens consumed by a completed operation."""


class TokenBucketRateLimiter:
    """Rate limiter based on token usage in a sliding window.

    Tracks tokens consumed and blocks when the limit would be exceeded.
    Unlike operation-based limiters, this requires recording usage after
    each request via `record_usage()`.

    Thread-safe for use across multiple asyncio tasks.
    """

    def __init__(self, max_tokens: int, window_seconds: float = 60.0) -> None:
        """Initialize the token rate limiter.

        Args:
            max_tokens: Maximum tokens allowed in the time window.
            window_seconds: Size of the sliding window (default 60s for TPM).
        """
        self._max_tokens = max_tokens
        self._window_seconds = window_seconds
        self._usage: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    def _current_usage(self, now: float) -> int:
        """Calculate current token usage in the window."""
        cutoff = now - self._window_seconds
        # Remove old entries
        while self._usage and self._usage[0][0] < cutoff:
            self._usage.popleft()
        return sum(tokens for _, tokens in self._usage)

    async def acquire(self) -> None:
        """Wait until there's capacity for more tokens.

        Blocks if current usage is at or near the limit (headroom% threshold).
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                current = self._current_usage(now)

                # Allow if we have headroom (headroom% of limit)
                if current < self._max_tokens * _DEFAULT_TPM_HEADROOM:
                    return

                # Calculate wait time until oldest usage expires
                if self._usage:
                    wait_time = self._usage[0][0] + self._window_seconds - now
                else:
                    wait_time = 0.1

            if wait_time > 0:
                await asyncio.sleep(wait_time)

    def record_usage(self, tokens: int) -> None:
        """Record tokens consumed by a completed request.

        Should be called after each LLM response with total_tokens.
        """
        self._usage.append((time.monotonic(), tokens))

    async def __aenter__(self) -> Self:
        """Enter the rate limiter context."""
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit the rate limiter context.

        Usage is recorded separately via record_usage().
        """


class CompositeRateLimiter:
    """Combines multiple rate limiters into a single limiter.

    Acquires from all underlying limiters sequentially, ensuring
    all rate limits are satisfied before allowing an operation to proceed.
    """

    def __init__(self, limiters: Sequence[RateLimiter]) -> None:
        """Initialize the composite rate limiter.

        Args:
            limiters: Sequence of rate limiters to combine.
        """
        self._limiters = list(limiters)

    async def acquire(self) -> None:
        """Acquire permission from all underlying rate limiters.

        Blocks until all limiters allow the operation to proceed.
        """
        for limiter in self._limiters:
            await limiter.acquire()

    async def __aenter__(self) -> Self:
        """Enter the rate limiter context.

        Enters all underlying rate limiters in order.
        """
        entered: list[RateLimiter] = []
        try:
            for limiter in self._limiters:
                await limiter.__aenter__()
                entered.append(limiter)
            return self
        except Exception:
            # If any limiter fails to enter, exit all that were entered
            for limiter in reversed(entered):
                await limiter.__aexit__(None, None, None)
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit the rate limiter context.

        Exits all underlying rate limiters in reverse order.
        """
        # Exit in reverse order (stack semantics)
        for limiter in reversed(self._limiters):
            await limiter.__aexit__(exc_type, exc_val, exc_tb)

    def record_usage(self, tokens: int) -> None:
        """Forward usage recording to all child limiters."""
        for limiter in self._limiters:
            limiter.record_usage(tokens)
