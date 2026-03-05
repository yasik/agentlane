"""Single-threaded runtime environment implementation."""

import asyncio

from agentlane.messaging import DeliveryOutcome, DeliveryStatus

from ._dispatcher import Dispatcher
from ._scheduler import PerAgentMailboxScheduler
from ._types import DeliveryTask

_IN_FLIGHT_CANCELED = "Runtime shutdown canceled an in-flight delivery."
_QUEUED_CANCELED = "Runtime shutdown canceled a queued delivery."


class SingleThreadedEnvironment:
    """Runs an in-process worker pool with per-recipient serialization via scheduler."""

    def __init__(
        self,
        *,
        scheduler: PerAgentMailboxScheduler,
        dispatcher: Dispatcher,
        worker_count: int = 10,
    ) -> None:
        """Create environment with scheduler and dispatcher."""
        if worker_count <= 0:
            raise ValueError("worker_count must be greater than zero.")

        # Scheduler owns queueing/ordering; dispatcher owns handler invocation.
        self._scheduler = scheduler
        self._dispatcher = dispatcher
        self._worker_count = worker_count
        # Worker pool executes tasks concurrently across different recipients.
        self._worker_pool: set[asyncio.Task[None]] = set()
        # Tracks currently dispatching tasks so stop() can propagate cancellation.
        self._inflight_by_worker: dict[asyncio.Task[None], DeliveryTask] = {}
        self._running = False

    async def start(self) -> None:
        """Start background worker if not already running."""
        if self._running:
            return
        self._running = True
        # Workers are intentionally detached; lifecycle is controlled via stop/stop_when_idle.
        for _ in range(self._worker_count):
            worker_task = asyncio.create_task(self._run_worker())
            self._worker_pool.add(worker_task)

    async def stop(self) -> None:
        """Stop immediately and cancel both in-flight and queued deliveries."""
        self._running = False
        # Ask currently running handlers to cooperate via context cancellation token.
        for inflight_task in self._inflight_by_worker.values():
            inflight_task.cancellation_token.cancel()

        if self._worker_pool:
            # Forcefully stop worker loops; in-flight tasks become CANCELED outcomes.
            for worker_task in self._worker_pool:
                worker_task.cancel()
            await asyncio.gather(*self._worker_pool, return_exceptions=True)
            self._worker_pool.clear()
            self._inflight_by_worker.clear()

        # Any task not yet dispatched is canceled eagerly during shutdown.
        queued_tasks = await self._scheduler.drain()
        for task in queued_tasks:
            self._cancel_task(task=task, message=_QUEUED_CANCELED)

    async def stop_when_idle(self) -> None:
        """Wait for all scheduled work to complete, then stop."""
        await self._scheduler.wait_idle()
        await self.stop()

    async def submit(self, task: DeliveryTask) -> None:
        """Submit one task for processing."""
        # Auto-start keeps API ergonomic for callers that publish/send first.
        if not self._running:
            await self.start()
        await self._scheduler.enqueue(task)

    async def _run_worker(self) -> None:
        """Worker loop processing tasks one-by-one for multiple recipients."""
        while self._running:
            task = await self._scheduler.pop_next()
            worker_task = asyncio.current_task()
            if worker_task is not None:
                # Mark inflight before dispatch so stop() can target active work.
                self._inflight_by_worker[worker_task] = task
            try:
                outcome = await self._dispatcher.dispatch(task)
                # RPC path: complete awaiting caller with terminal delivery outcome.
                if task.response_future is not None and not task.response_future.done():
                    task.response_future.set_result(outcome)
            except asyncio.CancelledError:
                # Worker cancellation during dispatch maps to a canceled delivery outcome.
                self._cancel_task(task=task, message=_IN_FLIGHT_CANCELED)
                raise
            finally:
                if worker_task is not None:
                    self._inflight_by_worker.pop(worker_task, None)
                # Every popped task must decrement pending count exactly once.
                await self._scheduler.mark_done(task.recipient)

    def _cancel_task(self, task: DeliveryTask, message: str) -> None:
        """Request cancellation and resolve pending response future if present."""
        # Token cancellation lets cooperative handlers terminate quickly.
        task.cancellation_token.cancel()
        if task.response_future is None or task.response_future.done():
            return

        # Normalize cancellation into DeliveryOutcome so callers get a terminal result.
        task.response_future.set_result(
            DeliveryOutcome.failed(
                status=DeliveryStatus.CANCELED,
                message_id=task.envelope.message_id,
                correlation_id=task.envelope.correlation_id,
                message=message,
                retryable=True,
            )
        )
