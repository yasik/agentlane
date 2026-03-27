"""Retry decorator for LLM clients.

This module provides reusable retry logic with exponential backoff,
Retry-After header support, and metrics tracking. The retry behavior
is configurable via a predicate function, allowing different clients
(LiteLLM, OpenAI, Anthropic) to define their own retryable conditions.
"""

import functools
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import structlog
import tenacity

T = TypeVar("T")

LOGGER = structlog.get_logger("agentlane.models")

# Default HTTP status codes that should trigger a retry
DEFAULT_RETRY_STATUS_CODES: list[int] = [429, 500, 502, 503, 504, 529]


@dataclass
class RetryMetrics:
    """Metrics captured during retry execution."""

    attempts: int
    """Number of attempts (1 = no retries)."""

    retry_wait: float
    """Total time spent sleeping between retries in seconds."""

    backend_latency: float
    """Actual LLM call time excluding retry waits in seconds."""


@dataclass
class RetryResult[T]:
    """Result wrapper containing both the response and retry metrics."""

    result: T
    """The actual result from the function."""

    metrics: RetryMetrics
    """Retry metrics captured during execution."""


def extract_retry_after(exception: BaseException) -> float | None:
    """Extract retry-after seconds from exception message or attributes.

    Parses retry delay from:
    1. response_headers["retry-after"] if available
    2. Exception message containing "retry after X second(s)"

    Args:
        exception: The exception to extract retry-after from.

    Returns:
        The retry delay in seconds, or None if not found.
    """
    # Try response_headers first (if the client exposes it)
    raw_headers = getattr(exception, "response_headers", None)
    headers: Mapping[str, Any]
    if isinstance(raw_headers, Mapping):
        headers = cast(Mapping[str, Any], raw_headers)
    else:
        headers = {}
    if "retry-after" in headers:
        try:
            return float(headers["retry-after"])
        except (ValueError, TypeError):
            pass

    # Parse from message: "Please retry after X second(s)"
    message = str(exception)
    match = re.search(r"retry after (\d+(?:\.\d+)?)\s*second", message, re.IGNORECASE)
    if match:
        return float(match.group(1))

    return None


def wait_with_retry_after(
    retry_state: tenacity.RetryCallState,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
) -> float:
    """Wait based on Retry-After header, falling back to exponential backoff.

    Args:
        retry_state: The current retry state from tenacity.
        min_wait: Minimum wait time for exponential backoff.
        max_wait: Maximum wait time (caps both Retry-After and backoff).

    Returns:
        The number of seconds to wait before the next retry.
    """
    if retry_state.outcome and retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        if exception is not None:
            retry_after = extract_retry_after(exception)
            if retry_after is not None:
                # Honor the server's request, but cap at max_wait
                return min(retry_after, max_wait)

    # Fallback to exponential backoff (capped at max_wait)
    exp_wait = tenacity.wait_random_exponential(min=min_wait, max=max_wait)
    return exp_wait(retry_state)


def is_retryable_by_status_code(
    exception: BaseException,
    retry_status_codes: list[int] | None = None,
) -> bool:
    """Check if exception has a retryable HTTP status code.

    Args:
        exception: The exception to check.
        retry_status_codes: List of HTTP status codes that should trigger a retry.
            Defaults to DEFAULT_RETRY_STATUS_CODES.

    Returns:
        True if the exception has a retryable status code.
    """
    if retry_status_codes is None:
        retry_status_codes = DEFAULT_RETRY_STATUS_CODES

    status_code = getattr(exception, "status_code", None)
    return status_code is not None and status_code in retry_status_codes


def retry_on_errors(
    max_retries: int,
    is_retryable: Callable[[BaseException], bool] | None = None,
    logger: structlog.typing.WrappedLogger | None = None,
) -> Callable[
    [Callable[..., Awaitable[T]]],
    Callable[..., Awaitable[RetryResult[T]]],
]:
    """Decorator to retry async functions on retryable errors.

    This decorator wraps an async function with retry logic using tenacity.
    It tracks retry metrics (attempts, wait time, backend latency) and returns
    them along with the result.

    Args:
        max_retries: Maximum number of retry attempts.
        is_retryable: Predicate function that determines if an exception should
            trigger a retry. Defaults to checking for retryable HTTP status codes.
        logger: Logger instance for logging retry attempts. Defaults to module logger.

    Returns:
        Decorated function that returns RetryResult[T] with the result and metrics.

    Example:
        ```python
        def is_retryable(exc: BaseException) -> bool:
            return isinstance(exc, (RateLimitError, APIConnectionError))

        @retry_on_errors(max_retries=3, is_retryable=is_retryable)
        async def call_llm():
            return await client.complete(...)

        result = await call_llm()
        print(f"Got response after {result.metrics.attempts} attempts")
        ```
    """
    resolved_logger = logger or LOGGER

    if is_retryable is None:
        is_retryable = is_retryable_by_status_code

    def _make_log_retry(
        metrics_state: dict[str, Any],
    ) -> Callable[[tenacity.RetryCallState], None]:
        """Create a retry logging callback that also tracks metrics.

        Args:
            metrics_state: Mutable dict to track retry metrics across attempts.

        Returns:
            Callback function for tenacity's before_sleep hook.
        """

        def _log_retry(retry_state: tenacity.RetryCallState) -> None:
            """Log retry attempts and update retry metrics."""
            if retry_state.outcome and retry_state.outcome.failed:
                exception = retry_state.outcome.exception()
                status_code = getattr(exception, "status_code", None)
                attempt = retry_state.attempt_number

                # Track retry metrics - next attempt will be attempt_number + 1
                metrics_state["attempts"] = attempt + 1

                # Accumulate wait time from the upcoming sleep
                if retry_state.next_action:
                    sleep = retry_state.next_action.sleep or 0
                    metrics_state["wait_time"] += sleep

                if attempt >= max_retries:
                    resolved_logger.error(
                        "max retries exceeded",
                        error=str(exception),
                        error_type=type(exception).__name__,
                        status_code=status_code,
                        attempts=attempt,
                    )
                else:
                    # Calculate the wait time that will be used
                    wait_time: float = (
                        retry_state.next_action.sleep if retry_state.next_action else 0
                    )
                    if wait_time > 0:
                        resolved_logger.warning(
                            "retrying llm call due to error",
                            error=str(exception),
                            error_type=type(exception).__name__,
                            status_code=status_code,
                            attempt=attempt,
                            next_retry_in=wait_time,
                        )

        return _log_retry

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[RetryResult[T]]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> RetryResult[T]:
            # Track metrics in closure - start at 1 for first attempt
            metrics_state: dict[str, Any] = {"wait_time": 0.0, "attempts": 1}

            retrying = tenacity.AsyncRetrying(
                retry=tenacity.retry_if_exception(is_retryable),
                stop=tenacity.stop_after_attempt(max_retries),
                wait=wait_with_retry_after,
                before_sleep=_make_log_retry(metrics_state),
                reraise=True,
            )

            call_start = time.perf_counter()
            result = cast(T, await retrying(func, *args, **kwargs))
            total_time_s = time.perf_counter() - call_start

            return RetryResult(
                result=result,
                metrics=RetryMetrics(
                    attempts=metrics_state["attempts"],
                    retry_wait=metrics_state["wait_time"],
                    backend_latency=total_time_s - metrics_state["wait_time"],
                ),
            )

        return wrapper

    return decorator
