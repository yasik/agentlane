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
        """Initialize scheduler structures.

        Args:
            mailbox_capacity: Max queued tasks allowed per recipient mailbox.
        """
        # One FIFO mailbox per recipient preserves per-agent ordering.
        self._mailboxes: dict[AgentId, deque[DeliveryTask]] = defaultdict(deque)
        # Round-robin queue of recipients that currently have pending work.
        self._ready_queue: deque[AgentId] = deque()
        # Membership guard so a recipient appears in ready_queue at most once.
        self._ready_set: set[AgentId] = set()
        # Recipients currently being processed by a worker.
        # A recipient is re-queued only after mark_done(recipient).
        self._active_recipients: set[AgentId] = set()
        # Single condition protects scheduler state and coordinates waiters/workers.
        self._condition = asyncio.Condition()
        # Tracks enqueued-but-not-completed tasks for wait_idle/drain semantics.
        self._pending_task_count = 0
        self._mailbox_capacity = mailbox_capacity

    async def enqueue(self, task: DeliveryTask) -> None:
        """Enqueue a delivery task according to mailbox policy.

        Args:
            task: Delivery task to enqueue.

        Returns:
            None: Always returns after scheduler state update.

        Raises:
            SchedulerRejectedError: If recipient mailbox is at capacity.
        """
        async with self._condition:
            mailbox = self._mailboxes[task.recipient]
            if len(mailbox) >= self._mailbox_capacity:
                raise SchedulerRejectedError(
                    f"Mailbox capacity exceeded for recipient '{task.recipient}'."
                )

            # Append to recipient-local FIFO mailbox.
            mailbox.append(task)
            self._pending_task_count += 1

            # Enqueue recipient once when it transitions from empty->non-empty.
            # Active recipients are re-queued by mark_done(recipient).
            if (
                task.recipient not in self._ready_set
                and task.recipient not in self._active_recipients
            ):
                self._ready_queue.append(task.recipient)
                self._ready_set.add(task.recipient)

            # Wake workers waiting for new schedulable recipients.
            self._condition.notify_all()

    async def pop_next(self) -> DeliveryTask:
        """Pop the next task to dispatch using fair round-robin across mailboxes.

        Returns:
            DeliveryTask: Next schedulable task.
        """
        async with self._condition:
            while not self._ready_queue:
                await self._condition.wait()

            # Pick next recipient in round-robin order.
            recipient = self._ready_queue.popleft()
            self._ready_set.remove(recipient)
            mailbox = self._mailboxes[recipient]

            # Pop exactly one item to preserve per-recipient FIFO semantics.
            task = mailbox.popleft()
            # Recipient stays active until mark_done(recipient), preventing
            # overlapping execution for the same AgentId.
            self._active_recipients.add(recipient)

            return task

    async def mark_done(self, recipient: AgentId) -> None:
        """Mark one previously enqueued task as completed for a recipient.

        Args:
            recipient: Recipient whose currently active task completed.

        Returns:
            None: Always returns after scheduler accounting updates.
        """
        async with self._condition:
            self._active_recipients.discard(recipient)
            self._pending_task_count -= 1
            mailbox = self._mailboxes.get(recipient)
            if (
                mailbox
                and recipient not in self._ready_set
                and recipient not in self._active_recipients
            ):
                # Recipient still has backlog and is no longer active:
                # re-queue once for fair round-robin scheduling.
                self._ready_queue.append(recipient)
                self._ready_set.add(recipient)
                self._condition.notify_all()
            if self._pending_task_count <= 0:
                self._pending_task_count = 0
                # Unblock wait_idle/drain waiters once no work remains.
                self._condition.notify_all()

    async def wait_idle(self) -> None:
        """Block until all queued tasks are completed.

        Returns:
            None: Returns only when pending task count reaches zero.
        """
        async with self._condition:
            while self._pending_task_count > 0:
                await self._condition.wait()

    async def drain(self) -> list[DeliveryTask]:
        """Remove and return all queued tasks without dispatching them.

        Returns:
            list[DeliveryTask]: Tasks removed from queue and not dispatched.
        """
        async with self._condition:
            drained: list[DeliveryTask] = []
            for mailbox in self._mailboxes.values():
                # Preserve FIFO order inside each mailbox while flattening.
                drained.extend(mailbox)
                mailbox.clear()

            # Reset scheduler readiness state after force-drain.
            self._mailboxes.clear()
            self._ready_queue.clear()
            self._ready_set.clear()

            # Adjust pending count for all tasks removed from dispatch path.
            self._pending_task_count -= len(drained)
            if self._pending_task_count <= 0:
                self._pending_task_count = 0
                self._condition.notify_all()

            return drained
