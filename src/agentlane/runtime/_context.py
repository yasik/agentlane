"""Async context managers for scoped runtime lifecycle management."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from ._runtime import (
    DistributedRuntimeEngine,
    RuntimeEngine,
    SingleThreadedRuntimeEngine,
)


@asynccontextmanager
async def runtime_scope(
    runtime: RuntimeEngine | None = None,
    *,
    expected_type: type[RuntimeEngine] | None = None,
    runtime_factory: type[RuntimeEngine] = SingleThreadedRuntimeEngine,
) -> AsyncIterator[RuntimeEngine]:
    """Scope runtime lifecycle to an async context block.

    The scope starts the runtime on entry if it was not already running.
    On normal exit it drains pending work (`stop_when_idle`). On exceptional
    exit it cancels work immediately (`stop`).

    If `expected_type` is provided, the scope validates that the runtime
    implementation type matches and raises `ValueError` otherwise.

    If `runtime` is `None`, a new `RuntimeEngine` is created with
    reasonable defaults via `runtime_factory`.

    The context manager yields the `RuntimeEngine` instance itself.

    Example:
        ```python
        from agentlane.runtime import SingleThreadedRuntimeEngine, runtime_scope

        async with runtime_scope(
            expected_type=SingleThreadedRuntimeEngine
        ) as runtime:
            assert isinstance(runtime, SingleThreadedRuntimeEngine)
        ```

    Args:
        runtime: Optional pre-created runtime instance.
        expected_type: Optional runtime class constraint validated on entry.
        runtime_factory: Runtime class used when `runtime` is None.

    Returns:
        AsyncIterator[RuntimeEngine]: Async context yielding a running runtime.

    Raises:
        ValueError: If `runtime` is provided and does not match `expected_type`.
    """
    if runtime is None:
        runtime = runtime_factory()

    if expected_type is not None and not isinstance(runtime, expected_type):
        raise ValueError(
            "Runtime type mismatch. "
            f"Expected '{expected_type.__name__}', got '{type(runtime).__name__}'."
        )

    was_running = runtime.is_running
    if not was_running:
        await runtime.start()

    try:
        yield runtime
    except BaseException:
        # On abnormal exit, cancel in-flight and queued work eagerly.
        if not was_running:
            await runtime.stop()
        raise
    else:
        # On clean exit, allow queued work to drain before shutdown.
        if not was_running:
            await runtime.stop_when_idle()


@asynccontextmanager
async def single_threaded_runtime(
    runtime: RuntimeEngine | None = None,
) -> AsyncIterator[RuntimeEngine]:
    """Scope a single-threaded runtime lifecycle to one async context block.

    Use this for in-process message handling that should automatically
    initialize runtime workers and tear them down after execution.

    Example:
        ```python
        from agentlane.runtime import on_message
        from agentlane.messaging import AgentId, MessageContext
        from agentlane.runtime import single_threaded_runtime

        class WorkerAgent:
            @on_message
            async def handle(self, payload: dict, context: MessageContext) -> object:
                _ = context
                return {"echo": payload}

        async with single_threaded_runtime() as runtime:
            runtime.register_factory("worker", lambda _engine: WorkerAgent())
            outcome = await runtime.send_message(
                {"task": "ping"},
                recipient=AgentId.from_values("worker", "session-1"),
            )
            assert outcome.status.value == "delivered"
        ```

    Args:
        runtime: Optional runtime instance to scope. If None, creates one.

    Returns:
        AsyncIterator[RuntimeEngine]: Async context yielding single-threaded runtime.

    Raises:
        ValueError: If provided runtime is not `SingleThreadedRuntimeEngine`.
    """
    async with runtime_scope(
        runtime,
        expected_type=SingleThreadedRuntimeEngine,
        runtime_factory=SingleThreadedRuntimeEngine,
    ) as scoped_runtime:
        yield scoped_runtime


@asynccontextmanager
async def distributed_runtime(
    runtime: RuntimeEngine | None = None,
) -> AsyncIterator[RuntimeEngine]:
    """Scope a distributed runtime lifecycle to one async context block.

    Use this for workflows that should run against a distributed runtime
    implementation. The context manager validates runtime type before yielding.

    Example:
        ```python
        from agentlane.runtime import DistributedRuntimeEngine, distributed_runtime

        async with distributed_runtime() as runtime:
            # Configure distributed runtime-scoped resources here.
            # Message delivery is added when distributed submit is implemented.
            assert isinstance(runtime, DistributedRuntimeEngine)
        ```

    Args:
        runtime: Optional runtime instance to scope. If None, creates one.

    Returns:
        AsyncIterator[RuntimeEngine]: Async context yielding distributed runtime.

    Raises:
        ValueError: If provided runtime is not `DistributedRuntimeEngine`.
    """
    async with runtime_scope(
        runtime,
        expected_type=DistributedRuntimeEngine,
        runtime_factory=DistributedRuntimeEngine,
    ) as scoped_runtime:
        yield scoped_runtime
