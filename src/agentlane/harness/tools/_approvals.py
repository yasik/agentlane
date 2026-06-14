"""Approval lifecycle helpers for first-party harness tools."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import count
from typing import cast

from ._permissions import (
    ToolPermissionDecision,
    ToolPermissionOutcome,
    ToolPermissionRequest,
)


@dataclass(frozen=True, slots=True)
class _ToolApprovalEventStreamEnd:
    """Sentinel used to close broker event subscribers."""


_TOOL_APPROVAL_EVENT_STREAM_END = _ToolApprovalEventStreamEnd()
type _ToolApprovalEventQueueItem = ToolApprovalEvent | _ToolApprovalEventStreamEnd


class ToolApprovalStatus(StrEnum):
    """Lifecycle status for one host approval request."""

    PENDING = "pending"
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class ToolApprovalRecord:
    """Snapshot of one approval request tracked by a broker."""

    request_id: str
    request: ToolPermissionRequest
    approval_required_decision: ToolPermissionDecision
    status: ToolApprovalStatus
    final_decision: ToolPermissionDecision | None = None


@dataclass(frozen=True, slots=True)
class ToolApprovalEvent:
    """Event emitted when an approval request changes status."""

    record: ToolApprovalRecord

    @property
    def request_id(self) -> str:
        """Return the stable request id for this event."""
        return self.record.request_id

    @property
    def status(self) -> ToolApprovalStatus:
        """Return the status snapshot carried by this event."""
        return self.record.status


class ToolApprovalBroker:
    """Approval callback helper for host applications.

    The broker keeps request tracking and host resolution separate from any
    CLI, desktop, or web approval UI.

    By default the broker only handles `REQUIRE_APPROVAL` decisions and raises
    on anything else, because an `ALLOW`/`DENY` decision needs no host
    resolution round-trip. Pass `record_immediate_decisions=True` to instead
    record an already-decided `ALLOW`/`DENY` as a request that is resolved in
    the same step: it emits the `PENDING` then `RESOLVED` events and returns the
    decision unchanged. Hosts use this to route auto-allow and auto-deny modes
    through the same observable event stream as interactive approvals instead of
    maintaining separate shadow counters.
    """

    def __init__(self, *, record_immediate_decisions: bool = False) -> None:
        """Create an approval broker.

        Args:
            record_immediate_decisions: When `True`, `callback` records an
                already-decided `ALLOW`/`DENY` decision as an immediately
                resolved request instead of raising. Defaults to `False` so the
                broker only brokers `REQUIRE_APPROVAL` decisions.
        """
        self._record_immediate_decisions = record_immediate_decisions
        self._request_ids = count(1)
        self._records: dict[str, ToolApprovalRecord] = {}
        self._waiters: dict[str, asyncio.Future[ToolPermissionDecision]] = {}
        self._event_queues: set[asyncio.Queue[_ToolApprovalEventQueueItem]] = set()
        self._events_closed = False

    async def callback(
        self,
        request: ToolPermissionRequest,
        decision: ToolPermissionDecision,
    ) -> ToolPermissionDecision:
        """ToolApprovalCallback-compatible approval boundary."""
        if decision.outcome != ToolPermissionOutcome.REQUIRE_APPROVAL:
            if self._record_immediate_decisions:
                return self._record_immediate(request, decision)
            raise ValueError("ToolApprovalBroker can only broker approvals.")

        request_id = self._next_request_id()
        record = ToolApprovalRecord(
            request_id=request_id,
            request=request,
            approval_required_decision=decision,
            status=ToolApprovalStatus.PENDING,
        )
        future: asyncio.Future[ToolPermissionDecision] = (
            asyncio.get_running_loop().create_future()
        )
        self._records[request_id] = record
        self._waiters[request_id] = future
        self._emit(record)

        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            self._discard(request_id)
            raise

    def _record_immediate(
        self,
        request: ToolPermissionRequest,
        decision: ToolPermissionDecision,
    ) -> ToolPermissionDecision:
        """Record an already-decided allow/deny as a resolved request.

        The pending then resolved events are emitted so subscribers observe the
        same lifecycle they see for brokered approvals, without ever exposing a
        pending record (the request is resolved before this method returns).
        """
        request_id = self._next_request_id()
        pending_record = ToolApprovalRecord(
            request_id=request_id,
            request=request,
            approval_required_decision=decision,
            status=ToolApprovalStatus.PENDING,
        )
        self._emit(pending_record)
        self._emit(
            replace(
                pending_record,
                status=ToolApprovalStatus.RESOLVED,
                final_decision=decision,
            )
        )
        return decision

    def pending(self) -> tuple[ToolApprovalRecord, ...]:
        """Return a read-only snapshot of currently pending approvals."""
        return tuple(
            record
            for record in self._records.values()
            if record.status == ToolApprovalStatus.PENDING
        )

    async def resolve(
        self,
        request_id: str,
        decision: ToolPermissionDecision,
    ) -> bool:
        """Resolve one pending approval request.

        Returns:
            bool: `True` when a pending request was resolved, or `False` when
            the request id was unknown or no longer pending.
        """
        return self._complete(
            request_id,
            decision=decision,
        )

    async def events(self) -> AsyncIterator[ToolApprovalEvent]:
        """Yield approval lifecycle events emitted after subscription."""
        if self._events_closed:
            return

        queue: asyncio.Queue[_ToolApprovalEventQueueItem] = asyncio.Queue()
        self._event_queues.add(queue)
        try:
            while True:
                item = await queue.get()
                if item is _TOOL_APPROVAL_EVENT_STREAM_END:
                    return
                yield cast(ToolApprovalEvent, item)
        finally:
            self._event_queues.discard(queue)

    def close(self) -> None:
        """Close all event subscribers and prevent future event subscriptions."""
        if self._events_closed:
            return

        self._events_closed = True
        queues = tuple(self._event_queues)
        self._event_queues.clear()
        for queue in queues:
            queue.put_nowait(_TOOL_APPROVAL_EVENT_STREAM_END)

    def _complete(
        self,
        request_id: str,
        *,
        decision: ToolPermissionDecision,
    ) -> bool:
        record = self._records.get(request_id)
        if record is None or record.status != ToolApprovalStatus.PENDING:
            return False

        completed_record = replace(
            record,
            status=ToolApprovalStatus.RESOLVED,
            final_decision=decision,
        )

        self._records.pop(request_id, None)
        waiter = self._waiters.pop(request_id, None)
        if waiter is not None and not waiter.done():
            waiter.set_result(decision)

        self._emit(completed_record)
        return True

    def _discard(self, request_id: str) -> None:
        record = self._records.get(request_id)
        if record is None or record.status != ToolApprovalStatus.PENDING:
            return

        self._records.pop(request_id, None)
        waiter = self._waiters.pop(request_id, None)
        if waiter is not None and not waiter.done():
            waiter.cancel()

    def _emit(self, record: ToolApprovalRecord) -> None:
        event = ToolApprovalEvent(record=record)
        for queue in tuple(self._event_queues):
            queue.put_nowait(event)

    def _next_request_id(self) -> str:
        return f"tool-approval-{next(self._request_ids)}"
