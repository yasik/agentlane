"""Single-threaded runtime environment implementation."""

import asyncio

from agentlane.messaging import DeliveryOutcome, DeliveryStatus

from ._dispatcher import Dispatcher
from ._scheduler import PerAgentMailboxScheduler
from ._types import DeliveryTask

_IN_FLIGHT_CANCELED = "Runtime shutdown canceled an in-flight delivery."
_QUEUED_CANCELED = "Runtime shutdown canceled a queued delivery."


class SingleThreadedEnvironment:
    """Runs a single worker loop over the scheduler for deterministic in-process execution."""

    def __init__(
        self,
        *,
        scheduler: PerAgentMailboxScheduler,
        dispatcher: Dispatcher,
    ) -> None:
        """Create environment with scheduler and dispatcher."""
        self._scheduler = scheduler
        self._dispatcher = dispatcher
        self._worker_task: asyncio.Task[None] | None = None
        self._inflight_task: DeliveryTask | None = None
        self._running = False

    async def start(self) -> None:
        """Start background worker if not already running."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop immediately and cancel both in-flight and queued deliveries."""
        self._running = False
        if self._inflight_task is not None:
            self._inflight_task.cancellation_token.cancel()

        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        queued_tasks = await self._scheduler.drain()
        for task in queued_tasks:
            self._cancel_task(task=task, message=_QUEUED_CANCELED)

    async def stop_when_idle(self) -> None:
        """Wait for all scheduled work to complete, then stop."""
        await self._scheduler.wait_idle()
        await self.stop()

    async def submit(self, task: DeliveryTask) -> None:
        """Submit one task for processing."""
        if not self._running:
            await self.start()
        await self._scheduler.enqueue(task)

    async def _run(self) -> None:
        """Worker loop processing one task at a time."""
        while self._running:
            task = await self._scheduler.pop_next()
            self._inflight_task = task
            try:
                outcome = await self._dispatcher.dispatch(task)
                if task.response_future is not None and not task.response_future.done():
                    task.response_future.set_result(outcome)
            except asyncio.CancelledError:
                self._cancel_task(task=task, message=_IN_FLIGHT_CANCELED)
                raise
            finally:
                self._inflight_task = None
                await self._scheduler.mark_done()

    @staticmethod
    def _cancel_task(task: DeliveryTask, message: str) -> None:
        """Request cancellation and resolve pending response future if present."""
        task.cancellation_token.cancel()
        if task.response_future is None or task.response_future.done():
            return

        task.response_future.set_result(
            DeliveryOutcome.failed(
                status=DeliveryStatus.CANCELED,
                message_id=task.envelope.message_id,
                correlation_id=task.envelope.correlation_id,
                message=message,
                retryable=True,
            )
        )
