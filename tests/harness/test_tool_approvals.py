"""Tests for host-facing tool approval broker helpers."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

from pydantic import BaseModel

from agentlane.harness.tools import (
    ToolApprovalBroker,
    ToolApprovalEvent,
    ToolApprovalStatus,
    ToolOperation,
    ToolPermissionDecision,
    ToolPermissionRequest,
    read_tool,
)
from agentlane.models import Tool
from agentlane.runtime import CancellationToken


def _request(tmp_path: Path) -> ToolPermissionRequest:
    return ToolPermissionRequest(
        tool_name="read",
        operation=ToolOperation.READ_FILE,
        cwd=tmp_path,
        path=tmp_path / "notes.txt",
    )


async def _next_event(
    events: AsyncIterator[ToolApprovalEvent],
) -> ToolApprovalEvent:
    return await asyncio.wait_for(anext(events), timeout=1.0)


async def _assert_events_closed(events: AsyncIterator[ToolApprovalEvent]) -> None:
    try:
        await asyncio.wait_for(anext(events), timeout=1.0)
    except StopAsyncIteration:
        return
    raise AssertionError("Expected approval events iterator to close.")


def test_tool_approval_broker_resolves_pending_request(tmp_path: Path) -> None:
    async def scenario() -> None:
        broker = ToolApprovalBroker()
        events = broker.events()
        requested_event_task = asyncio.create_task(_next_event(events))
        approval_required = ToolPermissionDecision.require_approval()
        callback_task = asyncio.create_task(
            broker.callback(_request(tmp_path), approval_required)
        )

        requested_event = await requested_event_task
        pending = broker.pending()

        assert requested_event.request_id == "tool-approval-1"
        assert requested_event.status == ToolApprovalStatus.PENDING
        assert len(pending) == 1
        assert pending[0].request_id == requested_event.request_id
        assert pending[0].request == _request(tmp_path)
        assert pending[0].approval_required_decision == approval_required
        assert pending[0].final_decision is None
        await asyncio.sleep(0)
        assert not callback_task.done()

        allowed = ToolPermissionDecision.allow()
        resolved = await broker.resolve(requested_event.request_id, allowed)
        result = await asyncio.wait_for(callback_task, timeout=1.0)
        resolved_event = await _next_event(events)

        assert resolved is True
        assert result == allowed
        assert broker.pending() == ()
        assert broker.__dict__["_records"] == {}
        assert resolved_event.request_id == requested_event.request_id
        assert resolved_event.status == ToolApprovalStatus.RESOLVED
        assert resolved_event.record.final_decision == allowed

    asyncio.run(scenario())


def test_tool_approval_broker_can_resolve_with_original_decision(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        broker = ToolApprovalBroker()
        events = broker.events()
        requested_event_task = asyncio.create_task(_next_event(events))
        approval_required = ToolPermissionDecision.require_approval("approval needed")
        callback_task = asyncio.create_task(
            broker.callback(_request(tmp_path), approval_required)
        )

        requested_event = await requested_event_task
        resolved = await broker.resolve(requested_event.request_id, approval_required)
        result = await asyncio.wait_for(callback_task, timeout=1.0)
        resolved_event = await _next_event(events)

        assert resolved is True
        assert requested_event.status == ToolApprovalStatus.PENDING
        assert result == approval_required
        assert broker.pending() == ()
        assert resolved_event.request_id == requested_event.request_id
        assert resolved_event.status == ToolApprovalStatus.RESOLVED
        assert resolved_event.record.final_decision == approval_required

    asyncio.run(scenario())


def test_tool_approval_broker_can_resolve_with_deny_decision(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        broker = ToolApprovalBroker()
        events = broker.events()
        requested_event_task = asyncio.create_task(_next_event(events))
        callback_task = asyncio.create_task(
            broker.callback(
                _request(tmp_path),
                ToolPermissionDecision.require_approval(),
            )
        )

        requested_event = await requested_event_task
        denied = ToolPermissionDecision.deny("not approved")
        resolved = await broker.resolve(requested_event.request_id, denied)
        result = await asyncio.wait_for(callback_task, timeout=1.0)
        resolved_event = await _next_event(events)

        assert resolved is True
        assert result == denied
        assert broker.pending() == ()
        assert resolved_event.request_id == requested_event.request_id
        assert resolved_event.status == ToolApprovalStatus.RESOLVED
        assert resolved_event.record.final_decision == result

    asyncio.run(scenario())


def test_tool_approval_broker_ignores_completed_requests(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        broker = ToolApprovalBroker()
        events = broker.events()
        requested_event_task = asyncio.create_task(_next_event(events))
        callback_task = asyncio.create_task(
            broker.callback(
                _request(tmp_path),
                ToolPermissionDecision.require_approval(),
            )
        )

        requested_event = await requested_event_task
        assert await broker.resolve(
            requested_event.request_id,
            ToolPermissionDecision.allow(),
        )
        assert await callback_task == ToolPermissionDecision.allow()
        assert not await broker.resolve(
            requested_event.request_id,
            ToolPermissionDecision.deny(),
        )
        assert not await broker.resolve(
            "unknown-request", ToolPermissionDecision.deny()
        )

    asyncio.run(scenario())


def test_tool_approval_broker_rejects_non_approval_decisions(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        broker = ToolApprovalBroker()

        try:
            await broker.callback(_request(tmp_path), ToolPermissionDecision.allow())
        except ValueError as exc:
            assert str(exc) == "ToolApprovalBroker can only broker approvals."
            return
        raise AssertionError("Expected non-approval decision to be rejected.")

    asyncio.run(scenario())


def test_tool_approval_broker_close_stops_event_subscribers(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        broker = ToolApprovalBroker()
        events = broker.events()
        requested_event_task = asyncio.create_task(_next_event(events))
        approval_required = ToolPermissionDecision.require_approval()
        callback_task = asyncio.create_task(
            broker.callback(_request(tmp_path), approval_required)
        )

        requested_event = await requested_event_task
        close_task = asyncio.create_task(_assert_events_closed(events))
        await asyncio.sleep(0)
        broker.close()
        await close_task

        pending = broker.pending()
        assert len(pending) == 1
        assert pending[0].request_id == requested_event.request_id
        assert not callback_task.done()

        allowed = ToolPermissionDecision.allow()
        assert await broker.resolve(requested_event.request_id, allowed)
        assert await asyncio.wait_for(callback_task, timeout=1.0) == allowed

        await _assert_events_closed(broker.events())
        broker.close()

    asyncio.run(scenario())


def test_tool_approval_broker_callback_can_allow_read_tool(
    tmp_path: Path,
) -> None:
    class RequireApprovalPolicy:
        def check(
            self,
            request: ToolPermissionRequest,
        ) -> ToolPermissionDecision:
            del request
            return ToolPermissionDecision.require_approval()

    async def scenario() -> None:
        broker = ToolApprovalBroker()
        events = broker.events()
        requested_event_task = asyncio.create_task(_next_event(events))
        target = tmp_path / "notes.txt"
        target.write_text("approved\n", encoding="utf-8")
        definition = read_tool(
            cwd=tmp_path,
            permissions=RequireApprovalPolicy(),
            approval_callback=broker.callback,
        )
        tool_spec = definition.tool
        if not hasattr(tool_spec, "run"):
            raise AssertionError("read tool should be executable.")
        tool = cast(Tool[BaseModel, str], tool_spec)
        args_model = tool.args_type()
        tool_task = asyncio.create_task(
            tool.run(args_model(path="notes.txt"), CancellationToken())
        )

        requested_event = await requested_event_task
        assert requested_event.record.request.path == target

        assert await broker.resolve(
            requested_event.request_id,
            ToolPermissionDecision.allow(),
        )
        output = await asyncio.wait_for(tool_task, timeout=1.0)

        assert output == "approved"

    asyncio.run(scenario())


async def _collect_immediate_events(
    broker: ToolApprovalBroker,
) -> list[ToolApprovalEvent]:
    """Subscribe, then return the pending then resolved immediate events."""
    collected: list[ToolApprovalEvent] = []
    async for event in broker.events():
        collected.append(event)
        if len(collected) == 2:
            return collected
    return collected


def test_tool_approval_broker_records_immediate_allow_when_enabled(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        broker = ToolApprovalBroker(record_immediate_decisions=True)
        collector = asyncio.create_task(_collect_immediate_events(broker))
        # Yield so the collector subscribes before the immediate decision emits.
        await asyncio.sleep(0)

        allow = ToolPermissionDecision.allow()
        returned = await broker.callback(_request(tmp_path), allow)

        pending_event, resolved_event = await asyncio.wait_for(collector, timeout=1.0)

        assert returned == allow
        assert pending_event.status == ToolApprovalStatus.PENDING
        assert resolved_event.status == ToolApprovalStatus.RESOLVED
        assert resolved_event.request_id == pending_event.request_id
        assert resolved_event.record.final_decision == allow
        # The decision resolved in the same step, so it never lingers pending.
        assert broker.pending() == ()

    asyncio.run(scenario())


def test_tool_approval_broker_records_immediate_deny_when_enabled(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        broker = ToolApprovalBroker(record_immediate_decisions=True)
        collector = asyncio.create_task(_collect_immediate_events(broker))
        await asyncio.sleep(0)

        deny = ToolPermissionDecision.deny("not allowed in this mode")
        returned = await broker.callback(_request(tmp_path), deny)

        pending_event, resolved_event = await asyncio.wait_for(collector, timeout=1.0)

        assert returned == deny
        assert pending_event.status == ToolApprovalStatus.PENDING
        assert resolved_event.status == ToolApprovalStatus.RESOLVED
        assert resolved_event.record.final_decision == deny
        assert broker.pending() == ()

    asyncio.run(scenario())


def test_tool_approval_broker_immediate_disabled_still_brokers_approvals(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        broker = ToolApprovalBroker(record_immediate_decisions=True)
        events = broker.events()
        requested_event_task = asyncio.create_task(_next_event(events))
        callback_task = asyncio.create_task(
            broker.callback(
                _request(tmp_path), ToolPermissionDecision.require_approval()
            )
        )

        requested_event = await requested_event_task

        assert requested_event.status == ToolApprovalStatus.PENDING
        assert broker.pending()[0].request_id == requested_event.request_id

        allow = ToolPermissionDecision.allow()
        assert await broker.resolve(requested_event.request_id, allow)
        assert await asyncio.wait_for(callback_task, timeout=1.0) == allow

    asyncio.run(scenario())
