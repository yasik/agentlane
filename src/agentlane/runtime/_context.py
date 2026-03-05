"""Async context managers for scoped runtime lifecycle management."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from ._engine import RuntimeEngine
from ._types import RuntimeMode


@asynccontextmanager
async def runtime_scope(
    runtime: RuntimeEngine | None = None,
    *,
    expected_mode: RuntimeMode | None = None,
) -> AsyncIterator[RuntimeEngine]:
    """Scope runtime lifecycle to an async context block.

    The scope starts the runtime on entry if it was not already running.
    On normal exit it drains pending work (`stop_when_idle`). On exceptional
    exit it cancels work immediately (`stop`).

    If `expected_mode` is provided, the scope validates that the runtime
    mode matches and raises `ValueError` otherwise.

    If `runtime` is `None`, a new `RuntimeEngine` is created with
    reasonable defaults. When `expected_mode` is provided, it is used as
    the mode for the created runtime.

    The context manager yields the `RuntimeEngine` instance itself.

    Example:
        ```python
        from agentlane.runtime import RuntimeEngine, runtime_scope, RuntimeMode

        async with runtime_scope(
            expected_mode=RuntimeMode.SINGLE_THREADED
        ) as runtime:
            outcome = await runtime.send_message("ping", recipient="worker", key="k")
        ```
    """
    if runtime is None:
        runtime = RuntimeEngine(
            mode=expected_mode or RuntimeMode.SINGLE_THREADED,
        )

    if expected_mode is not None and runtime.mode != expected_mode:
        raise ValueError(
            f"Runtime mode mismatch. Expected '{expected_mode}', got '{runtime.mode}'."
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
        from agentlane.agents import on_message
        from agentlane.messaging import MessageContext
        from agentlane.runtime import single_threaded_runtime

        class WorkerAgent:
            @on_message
            async def handle(self, payload: dict, context: MessageContext) -> object:
                _ = context
                return {"echo": payload}

        async with single_threaded_runtime() as runtime:
            runtime.register_factory("worker", WorkerAgent)
            outcome = await runtime.send_message(
                {"task": "ping"},
                recipient="worker",
                key="session-1",
            )
            assert outcome.status.value == "delivered"
        ```
    """
    async with runtime_scope(
        runtime,
        expected_mode=RuntimeMode.SINGLE_THREADED,
    ) as scoped_runtime:
        yield scoped_runtime


@asynccontextmanager
async def distributed_runtime(
    runtime: RuntimeEngine | None = None,
) -> AsyncIterator[RuntimeEngine]:
    """Scope a distributed runtime lifecycle to one async context block.

    Use this for workflows that should run against a runtime configured with
    `RuntimeMode.DISTRIBUTED`. The context manager validates the mode before
    yielding.

    Example:
        ```python
        from agentlane.runtime import RuntimeMode, distributed_runtime

        async with distributed_runtime() as runtime:
            # Configure distributed runtime-scoped resources here.
            # Message delivery is added when distributed submit is implemented.
            assert runtime.mode == RuntimeMode.DISTRIBUTED
        ```
    """
    async with runtime_scope(
        runtime,
        expected_mode=RuntimeMode.DISTRIBUTED,
    ) as scoped_runtime:
        yield scoped_runtime
