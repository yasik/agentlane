import asyncio
from io import StringIO
from pathlib import Path

import pytest
from agentlane_process_bridge import (
    BRIDGE_COMMAND_HANDLERS,
    ApproveCommand,
    BridgeBackend,
    BridgeCommandBackend,
    BridgeCommandHandler,
    BridgeEventType,
    CancelCommand,
    ConfigRejectedError,
    ConfigureCommand,
    ContractPayloadError,
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


class _ConfigStore:
    def __init__(self, document: dict[str, object] | None = None) -> None:
        self.document = document or {"model": "openai/gpt-5.5"}
        self.apply_calls: list[dict[str, object]] = []
        self.reject_message: str | None = None
        self.internal_document: dict[str, object] | None = None

    def snapshot(self) -> dict[str, object]:
        return dict(self.document)

    def apply(self, patch: dict[str, object]) -> dict[str, object]:
        self.apply_calls.append(patch)

        if self.reject_message is not None:
            raise ConfigRejectedError(self.reject_message)

        if self.internal_document is not None:
            self.document = dict(self.internal_document)
            raise RuntimeError("store bug")

        self.document = {**self.document, **patch}
        return self.snapshot()


def test_default_command_registry_covers_known_commands() -> None:
    command_types = [handler.command_type for handler in BRIDGE_COMMAND_HANDLERS]

    assert command_types == [
        PromptCommand,
        ApproveCommand,
        ConfigureCommand,
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
        assert "config" not in event
        await backend.close()

    asyncio.run(scenario())


def test_backend_start_emits_ready_with_config_snapshot() -> None:
    async def scenario() -> None:
        output = StringIO()
        store = _ConfigStore({"model": "openai/gpt-5.5"})
        backend = BridgeBackend(
            agent=FakeAgent(),
            events=EventWriter(output),
            config=store,
        )

        await backend.start()

        [event] = emitted_events(output)
        assert event["type"] == "ready"
        assert event["config"] == {"model": "openai/gpt-5.5"}
        await backend.close()

    asyncio.run(scenario())


def test_configure_reports_unsupported_without_store() -> None:
    async def scenario() -> None:
        output = StringIO()
        backend = BridgeBackend(agent=FakeAgent(), events=EventWriter(output))

        await backend.handle_command(ConfigureCommand(patch={"model": "x"}))

        [event] = emitted_events(output)
        assert event["type"] == "config"
        assert event["ok"] is False
        assert event["config"] is None
        assert event["error"]["code"] == "unsupported"
        await backend.close()

    asyncio.run(scenario())


def test_prompt_immediate_cancel_still_emits_terminal_event() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))

        await backend.handle_command(PromptCommand(text="go"))
        await backend.handle_command(CancelCommand())
        await backend.cancel_active_run(emit_terminal=True)

        event_types = [event["type"] for event in emitted_events(output)]
        assert "run_start" in event_types
        assert "cancel_requested" in event_types
        assert "run_cancelled" in event_types
        await backend.close()

    asyncio.run(scenario())


def test_prompt_immediate_reset_still_emits_terminal_event() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))

        await backend.handle_command(PromptCommand(text="go"))
        await backend.handle_command(ResetCommand())

        event_types = [event["type"] for event in emitted_events(output)]
        assert "run_start" in event_types
        assert "run_cancelled" in event_types
        assert event_types[-1] == "reset"
        assert agent.reset_calls == 1
        await backend.close()

    asyncio.run(scenario())


def test_configure_reports_invalid_patch_shape_with_snapshot() -> None:
    async def scenario() -> None:
        output = StringIO()
        store = _ConfigStore({"model": "openai/gpt-5.5"})
        backend = BridgeBackend(
            agent=FakeAgent(),
            events=EventWriter(output),
            config=store,
        )

        await backend.handle_command(ConfigureCommand(patch=None))

        [event] = emitted_events(output)
        assert event["type"] == "config"
        assert event["ok"] is False
        assert event["config"] == {"model": "openai/gpt-5.5"}
        assert event["error"]["code"] == "invalid"
        assert store.apply_calls == []
        await backend.close()

    asyncio.run(scenario())


def test_configure_reports_store_rejection_with_snapshot() -> None:
    async def scenario() -> None:
        output = StringIO()
        store = _ConfigStore({"model": "openai/gpt-5.5"})
        store.reject_message = "Unknown model: openai/gpt-9"
        backend = BridgeBackend(
            agent=FakeAgent(),
            events=EventWriter(output),
            config=store,
        )

        await backend.handle_command(ConfigureCommand(patch={"model": "openai/gpt-9"}))

        [event] = emitted_events(output)
        assert event["type"] == "config"
        assert event["ok"] is False
        assert event["config"] == {"model": "openai/gpt-5.5"}
        assert event["error"] == {
            "code": "rejected",
            "message": "Unknown model: openai/gpt-9",
        }
        await backend.close()

    asyncio.run(scenario())


def test_configure_reports_internal_error_with_truth_snapshot() -> None:
    async def scenario() -> None:
        output = StringIO()
        store = _ConfigStore({"model": "openai/gpt-5.5"})
        store.internal_document = {"model": "anthropic/claude-opus-4-8"}
        backend = BridgeBackend(
            agent=FakeAgent(),
            events=EventWriter(output),
            config=store,
        )

        await backend.handle_command(
            ConfigureCommand(patch={"model": "anthropic/claude-opus-4-8"})
        )

        [event] = emitted_events(output)
        assert event["type"] == "config"
        assert event["ok"] is False
        assert event["config"] == {"model": "anthropic/claude-opus-4-8"}
        assert event["error"]["code"] == "internal"
        assert (
            event["error"]["message"]
            == "Runtime configuration failed inside the backend."
        )
        await backend.close()

    asyncio.run(scenario())


def test_configure_oversize_document_fails_loudly() -> None:
    async def scenario() -> None:
        output = StringIO()
        store = _ConfigStore({"model": "openai/gpt-5.5"})
        backend = BridgeBackend(
            agent=FakeAgent(),
            events=EventWriter(output),
            config=store,
        )

        with pytest.raises(ContractPayloadError):
            await backend.handle_command(
                ConfigureCommand(patch={"catalog": "x" * 40_000})
            )

        assert emitted_events(output) == []
        await backend.close()

    asyncio.run(scenario())


def test_configure_settles_before_next_prompt_run() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        store = _ConfigStore({"model": "openai/gpt-5.5"})
        backend = BridgeBackend(
            agent=agent,
            events=EventWriter(output),
            config=store,
        )

        await backend.handle_command(
            ConfigureCommand(patch={"model": "anthropic/claude-opus-4-8"})
        )
        await backend.handle_command(PromptCommand(text="go"))
        await wait_for_stream(agent)

        events = emitted_events(output)
        assert [event["type"] for event in events] == ["config", "run_start"]
        assert events[0]["config"] == {"model": "anthropic/claude-opus-4-8"}
        assert store.apply_calls == [{"model": "anthropic/claude-opus-4-8"}]
        assert agent.prompts == ["go"]
        await backend.close()

    asyncio.run(scenario())


def test_configure_is_accepted_while_run_is_active() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        store = _ConfigStore({"model": "openai/gpt-5.5"})
        backend = BridgeBackend(
            agent=agent,
            events=EventWriter(output),
            config=store,
        )

        await backend.handle_command(PromptCommand(text="go"))
        await wait_for_stream(agent)
        await backend.handle_command(
            ConfigureCommand(patch={"model": "anthropic/claude-opus-4-8"})
        )

        events = emitted_events(output)
        assert [event["type"] for event in events] == ["run_start", "config"]
        assert events[1]["ok"] is True
        assert events[1]["config"] == {"model": "anthropic/claude-opus-4-8"}
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
        assert "config" not in events[2]
        assert agent.reset_calls == 1
        assert agent.prompts == ["first", "second"]
        await backend.close()

    asyncio.run(scenario())


def test_backend_reset_reannounces_config_snapshot() -> None:
    class ResetAwareConfigStore:
        def __init__(self, agent: FakeAgent) -> None:
            self.agent = agent

        def snapshot(self) -> dict[str, object]:
            return {"reset_calls": self.agent.reset_calls}

        def apply(self, patch: dict[str, object]) -> dict[str, object]:
            del patch

            return self.snapshot()

    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        store = ResetAwareConfigStore(agent)
        backend = BridgeBackend(
            agent=agent,
            events=EventWriter(output),
            config=store,
        )

        await backend.handle_command(ResetCommand())

        [event] = emitted_events(output)
        assert event["type"] == "reset"
        assert event["config"] == {"reset_calls": 1}
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

        await backend.handle_command(PromptCommand(text="go"))
        stream = await wait_for_stream(agent)
        stream.close_error = RuntimeError("close failed")

        await backend.cancel_active_run(emit_terminal=True)

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
