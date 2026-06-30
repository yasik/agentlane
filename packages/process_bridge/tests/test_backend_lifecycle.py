import asyncio
from io import StringIO
from pathlib import Path

from agentlane_process_bridge import (
    BRIDGE_COMMAND_HANDLERS,
    ApproveCommand,
    BridgeBackend,
    BridgeCommandBackend,
    BridgeCommandHandler,
    BridgeEventType,
    CancelCommand,
    EventWriter,
    PromptCommand,
    ResetCommand,
    ShutdownCommand,
)

from agentlane.harness import RunResult, RunState
from agentlane.harness.tools import (
    ToolApprovalBroker,
    ToolApprovalRecord,
    ToolOperation,
    ToolPermissionDecision,
    ToolPermissionRequest,
)

from .helpers import FakeAgent, emitted_events, wait_for_event_count, wait_for_stream


def test_default_command_registry_covers_known_commands() -> None:
    command_types = [handler.command_type for handler in BRIDGE_COMMAND_HANDLERS]

    assert command_types == [
        PromptCommand,
        ApproveCommand,
        CancelCommand,
        ResetCommand,
        ShutdownCommand,
    ]
    assert len(command_types) == len(set(command_types))


def test_backend_accepts_explicit_command_handler_registry() -> None:
    class CustomPromptHandler(BridgeCommandHandler):
        def __init__(self) -> None:
            super().__init__(PromptCommand)

        async def handle(
            self,
            backend: BridgeCommandBackend,
            command: object,
        ) -> None:
            del command

            await backend.events.emit(
                BridgeEventType.RUN_EVENT,
                run_event_type="custom_prompt",
            )

    async def scenario() -> None:
        output = StringIO()
        backend = BridgeBackend(
            agent=FakeAgent(),
            events=EventWriter(output),
            command_handlers=(CustomPromptHandler(),),
        )

        await backend.handle_command(PromptCommand(text="go"))

        [event] = emitted_events(output)
        assert event["type"] == "run_event"
        assert event["run_event_type"] == "custom_prompt"
        await backend.close()

    asyncio.run(scenario())


def test_backend_start_emits_ready_with_metadata() -> None:
    def ready_metadata() -> dict[str, object]:
        return {"app": "demo"}

    async def scenario() -> None:
        output = StringIO()
        backend = BridgeBackend(
            agent=FakeAgent(),
            events=EventWriter(output),
            ready_metadata=ready_metadata,
        )

        await backend.start()

        [event] = emitted_events(output)
        assert event["type"] == "ready"
        assert event["protocol_version"] == "1.0"
        assert event["package"] == "agentlane-process-bridge"
        assert event["metadata"] == {"app": "demo"}
        await backend.close()

    asyncio.run(scenario())


def test_backend_rejects_bad_prompt_without_ending_active_run() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))

        await backend.handle_command(PromptCommand(text="go"))
        stream = await wait_for_stream(agent)
        await wait_for_event_count(output, 1)

        await backend.handle_command(PromptCommand(text="second"))
        await backend.handle_command(PromptCommand(text="   "))

        events = emitted_events(output)
        assert [event["type"] for event in events] == ["run_start", "error", "error"]
        assert events[1]["scope"] == "command"
        assert events[2]["message"] == "Prompt must not be empty."

        stream.finish(RunResult(final_output="done", responses=[], turn_count=1))
        events = await wait_for_event_count(output, 4)
        assert events[3]["type"] == "run_complete"
        assert events[3]["final_output"] == "done"
        assert events[3]["turn_count"] == 1
        assert events[3]["response_count"] == 0
        assert events[3]["shim_state"] == {}
        await backend.close()

    asyncio.run(scenario())


def test_backend_cancel_closes_stream_drains_result_and_emits_terminal() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))

        await backend.handle_command(PromptCommand(text="go"))
        stream = await wait_for_stream(agent)
        await backend.handle_command(CancelCommand())
        await wait_for_event_count(output, 3)

        events = emitted_events(output)
        assert [event["type"] for event in events] == [
            "run_start",
            "cancel_requested",
            "run_cancelled",
        ]
        assert stream.aclose_calls == 1
        assert stream.result_awaits == 1
        assert agent.cancellation_tokens[0] is not None
        assert agent.cancellation_tokens[0].is_cancelled
        await backend.close()

    asyncio.run(scenario())


def test_backend_reset_cancels_run_then_accepts_new_prompt() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))

        await backend.handle_command(PromptCommand(text="first"))
        await wait_for_stream(agent)
        await backend.handle_command(ResetCommand())
        await wait_for_event_count(output, 3)
        await backend.handle_command(PromptCommand(text="second"))
        await wait_for_stream(agent, 1)

        events = emitted_events(output)
        assert [event["type"] for event in events[:4]] == [
            "run_start",
            "run_cancelled",
            "reset",
            "run_start",
        ]
        assert agent.reset_calls == 1
        assert agent.prompts == ["first", "second"]
        await backend.close()

    asyncio.run(scenario())


def test_backend_shutdown_orders_run_cancelled_before_shutdown() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))

        await backend.handle_command(PromptCommand(text="go"))
        await wait_for_stream(agent)
        await backend.handle_command(ShutdownCommand())

        events = await wait_for_event_count(output, 3)
        assert [event["type"] for event in events] == [
            "run_start",
            "run_cancelled",
            "shutdown",
        ]

    asyncio.run(scenario())


def test_backend_shutdown_closes_when_terminal_cancel_emit_fails() -> None:
    class OutputClosedAfterFirstWrite(StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.write_count = 0

        def write(self, value: str) -> int:
            if self.write_count > 0:
                raise BrokenPipeError("stdout closed")

            self.write_count += 1
            return super().write(value)

    async def scenario() -> None:
        output = OutputClosedAfterFirstWrite()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))

        await backend.handle_command(PromptCommand(text="go"))
        await wait_for_stream(agent)

        await backend.handle_command(ShutdownCommand())

        assert not backend.has_active_run()
        assert emitted_events(output)[0]["type"] == "run_start"

    asyncio.run(scenario())


def test_backend_run_failure_denies_pending_approval_and_clears_run() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))

        await backend.handle_command(PromptCommand(text="go"))
        stream = await wait_for_stream(agent)
        approval_task = asyncio.create_task(
            backend.approvals.callback(
                _approval_request(),
                ToolPermissionDecision.require_approval("needs review"),
            )
        )
        await _wait_for_pending_approval(backend)

        stream.fail(RuntimeError("boom"))
        events = await wait_for_event_count(output, 2)
        decision = await approval_task

        assert [event["type"] for event in events] == ["run_start", "error"]
        assert events[1]["scope"] == "run"
        assert events[1]["message"] == "boom"
        assert not decision.allowed
        assert backend.approvals.pending() == ()

        await _wait_for_no_active_run(backend)
        await backend.handle_command(PromptCommand(text="next"))
        await wait_for_stream(agent, 1)
        assert agent.prompts == ["go", "next"]
        await backend.close()

    asyncio.run(scenario())


def test_backend_run_failure_reports_stream_cleanup_error() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))

        await backend.handle_command(PromptCommand(text="go"))
        stream = await wait_for_stream(agent)
        stream.close_error = RuntimeError("close failed")

        stream.fail(RuntimeError("boom"))
        events = await wait_for_event_count(output, 2)

        assert [event["type"] for event in events] == ["run_start", "error"]
        assert events[1]["scope"] == "run"
        assert events[1]["message"] == "boom; cleanup failed: close failed"
        await backend.close()

    asyncio.run(scenario())


def test_backend_successful_run_emits_complete_payload_and_clears_run() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))

        await backend.handle_command(PromptCommand(text="go"))
        stream = await wait_for_stream(agent)
        run_state = RunState(
            instructions=None,
            history=[],
            responses=[],
            turn_count=3,
        )
        run_state.shim_state["mode"] = "demo"

        stream.finish(
            RunResult(
                final_output="done",
                responses=[],
                turn_count=3,
                run_state=run_state,
            )
        )
        events = await wait_for_event_count(output, 2)

        assert [event["type"] for event in events] == ["run_start", "run_complete"]
        assert events[1]["final_output"] == "done"
        assert events[1]["turn_count"] == 3
        assert events[1]["response_count"] == 0
        assert events[1]["shim_state"] == {"mode": "demo"}

        await _wait_for_no_active_run(backend)
        await backend.close()

    asyncio.run(scenario())


def test_backend_cancel_denies_pending_approval() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))

        await backend.handle_command(PromptCommand(text="go"))
        await wait_for_stream(agent)
        approval_task = asyncio.create_task(
            backend.approvals.callback(
                _approval_request(),
                ToolPermissionDecision.require_approval("needs review"),
            )
        )
        await _wait_for_pending_approval(backend)

        await backend.handle_command(CancelCommand())
        decision = await approval_task

        assert not decision.allowed
        assert backend.approvals.pending() == ()
        await backend.close()

    asyncio.run(scenario())


def test_backend_cancel_reports_stream_cleanup_error() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))

        task = asyncio.create_task(backend.run_prompt("go"))
        stream = await wait_for_stream(agent)
        stream.close_error = RuntimeError("close failed")

        task.cancel()
        await task

        events = emitted_events(output)
        assert [event["type"] for event in events] == ["run_start", "error"]
        assert events[1]["scope"] == "run"
        assert events[1]["message"] == "Run cancellation cleanup failed: close failed"

        await backend.close()

    asyncio.run(scenario())


def test_approval_command_grants_only_strict_boolean_true() -> None:
    async def scenario() -> None:
        backend = BridgeBackend(agent=FakeAgent(), events=EventWriter(StringIO()))
        request = ToolPermissionRequest(
            tool_name="write",
            operation=ToolOperation.CREATE_FILE,
            cwd=Path("/workspace"),
        )
        task = asyncio.create_task(
            backend.approvals.callback(
                request,
                ToolPermissionDecision.require_approval("needs review"),
            )
        )
        for _ in range(100):
            pending = backend.approvals.pending()
            if pending:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("Expected pending approval.")

        await backend.handle_command(
            ApproveCommand(request_id=pending[0].request_id, allowed=False)
        )
        decision = await task

        assert not decision.allowed
        await backend.close()

    asyncio.run(scenario())


def test_approval_command_allows_pending_request() -> None:
    async def scenario() -> None:
        backend = BridgeBackend(agent=FakeAgent(), events=EventWriter(StringIO()))
        task = asyncio.create_task(
            backend.approvals.callback(
                _approval_request(),
                ToolPermissionDecision.require_approval("needs review"),
            )
        )
        [pending] = await _wait_for_pending_approval(backend)

        await backend.handle_command(
            ApproveCommand(request_id=pending.request_id, allowed=True)
        )
        decision = await task

        assert decision.allowed
        assert backend.approvals.pending() == ()
        await backend.close()

    asyncio.run(scenario())


def test_backend_uses_injected_approval_broker() -> None:
    async def scenario() -> None:
        # An app wires its agent's tool approval_callback to this broker, then
        # hands the SAME broker to the backend; an approve command must resolve
        # the request the agent's callback is waiting on.
        broker = ToolApprovalBroker(record_immediate_decisions=True)
        backend = BridgeBackend(
            agent=FakeAgent(),
            events=EventWriter(StringIO()),
            approvals=broker,
        )
        assert backend.approvals is broker

        task = asyncio.create_task(
            broker.callback(
                _approval_request(),
                ToolPermissionDecision.require_approval("needs review"),
            )
        )
        [pending] = await _wait_for_pending_approval(backend)

        await backend.handle_command(
            ApproveCommand(request_id=pending.request_id, allowed=True)
        )
        decision = await task

        assert decision.allowed
        assert broker.pending() == ()
        await backend.close()

    asyncio.run(scenario())


def test_approval_command_reports_unknown_request_id() -> None:
    async def scenario() -> None:
        output = StringIO()
        backend = BridgeBackend(agent=FakeAgent(), events=EventWriter(output))

        await backend.handle_command(ApproveCommand(request_id="missing", allowed=True))

        [event] = emitted_events(output)
        assert event["type"] == "error"
        assert event["scope"] == "command"
        assert event["message"] == "No pending approval request for id missing."
        await backend.close()

    asyncio.run(scenario())


def _approval_request() -> ToolPermissionRequest:
    return ToolPermissionRequest(
        tool_name="write",
        operation=ToolOperation.CREATE_FILE,
        cwd=Path("/workspace"),
    )


async def _wait_for_pending_approval(
    backend: BridgeBackend,
) -> tuple[ToolApprovalRecord, ...]:
    for _ in range(100):
        pending = backend.approvals.pending()
        if pending:
            return pending
        await asyncio.sleep(0.01)
    raise AssertionError("Expected pending approval.")


async def _wait_for_no_active_run(backend: BridgeBackend) -> None:
    for _ in range(100):
        if not backend.has_active_run():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Expected active run to clear.")
