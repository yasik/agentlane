"""Per-agent mailbox scheduler."""

import asyncio
from collections import defaultdict, deque

from agentlane.messaging import AgentId

from ._types import DeliveryTask


class SchedulerRejectedError(RuntimeError):
    """Raised when enqueue is rejected, usually due to mailbox capacity limits."""


class PerAgentMailboxScheduler:
    """Per-agent mailbox scheduler with in-order delivery and fair round-robin dispatch."""

    def __init__(self, *, mailbox_capacity: int = 2048) -> None:
        """Initialize scheduler structures."""
        self._mailboxes: dict[AgentId, deque[DeliveryTask]] = defaultdict(deque)
        self._ready_queue: deque[AgentId] = deque()
        self._ready_set: set[AgentId] = set()
        self._condition = asyncio.Condition()
        self._pending_task_count = 0
        self._mailbox_capacity = mailbox_capacity

    async def enqueue(self, task: DeliveryTask) -> None:
        """Enqueue a delivery task according to mailbox policy."""
        async with self._condition:
            mailbox = self._mailboxes[task.recipient]
            if len(mailbox) >= self._mailbox_capacity:
                raise SchedulerRejectedError(
                    f"Mailbox capacity exceeded for recipient '{task.recipient}'."
                )
            mailbox.append(task)
            self._pending_task_count += 1
            if task.recipient not in self._ready_set:
                self._ready_queue.append(task.recipient)
                self._ready_set.add(task.recipient)
            self._condition.notify_all()

    async def pop_next(self) -> DeliveryTask:
        """Pop the next task to dispatch using fair round-robin across mailboxes."""
        async with self._condition:
            while not self._ready_queue:
                await self._condition.wait()

            recipient = self._ready_queue.popleft()
            self._ready_set.remove(recipient)
            mailbox = self._mailboxes[recipient]
            task = mailbox.popleft()
            if mailbox:
                self._ready_queue.append(recipient)
                self._ready_set.add(recipient)
            return task

    async def mark_done(self) -> None:
        """Mark one previously enqueued task as completed."""
        async with self._condition:
            self._pending_task_count -= 1
            if self._pending_task_count <= 0:
                self._pending_task_count = 0
                self._condition.notify_all()

    async def wait_idle(self) -> None:
        """Block until all queued tasks are completed."""
        async with self._condition:
            while self._pending_task_count > 0:
                await self._condition.wait()

    async def drain(self) -> list[DeliveryTask]:
        """Remove and return all queued tasks without dispatching them."""
        async with self._condition:
            drained: list[DeliveryTask] = []
            for mailbox in self._mailboxes.values():
                drained.extend(mailbox)
                mailbox.clear()

            self._mailboxes.clear()
            self._ready_queue.clear()
            self._ready_set.clear()

            self._pending_task_count -= len(drained)
            if self._pending_task_count <= 0:
                self._pending_task_count = 0
                self._condition.notify_all()

            return drained
