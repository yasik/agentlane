"""Reusable backend controller for AgentLane stdio process bridges.

The backend owns one local bridge session: command validation, active-run
lifecycle, approval resolution, and run-event serialization. Transport parsing
and raw I/O stay outside this module so hosts can reuse the same lifecycle
rules with stdio, tests, or another process boundary.
"""

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal, Protocol, cast, runtime_checkable

import structlog

from agentlane.harness import RunEvent, RunResult, RunState
from agentlane.harness.tools import (
    ToolApprovalBroker,
    ToolApprovalEvent,
    ToolPermissionDecision,
)
from agentlane.runtime import CancellationToken

from ._events import RunEventEncoder
from ._protocol import (
    ERROR_SCOPE_COMMAND,
    ERROR_SCOPE_RUN,
    ApproveCommand,
    BridgeCommand,
    BridgeEventType,
    CancelCommand,
    ConfigureCommand,
    ContractPayloadError,
    EventWriter,
    PromptCommand,
    ResetCommand,
    ShutdownCommand,
)

_logger = structlog.get_logger(__name__)

type ConfigErrorCode = Literal["invalid", "unsupported", "rejected", "internal"]

_CONFIG_INTERNAL_ERROR_MESSAGE = "Runtime configuration failed inside the backend."
"""Non-leaking client message used when app-owned config code raises unexpectedly."""


class ConfigRejectedError(Exception):
    """Runtime config patch was rejected with a user-presentable message."""


class RuntimeConfigStore(Protocol):
    """App-owned runtime configuration document synchronized with the client.

    The bridge transports desired-state patches in and authoritative full
    documents out. It never interprets config keys or values.
    """

    def snapshot(self) -> dict[str, object]:
        """Return the complete current JSON-serializable config document."""
        ...

    def apply(self, patch: dict[str, object]) -> dict[str, object]:
        """Validate and apply a patch, then return the full applied document.

        Implementations should validate the whole patch before mutating state.
        Reject unknown keys and invalid values at the app boundary; config
        values should come from a closed vocabulary such as a model catalog, not
        arbitrary user strings. Raise ``ConfigRejectedError`` for user-fixable
        rejections; any other exception is treated as an app bug and surfaced as
        an internal failure.
        """
        ...


class RunEventStreamLike(Protocol):
    """Run-event stream contract returned by an AgentLane runtime.

    The bridge consumes events through iteration and then awaits ``result()`` to
    observe the final output or the terminal exception raised by the run.
    """

    def __aiter__(self) -> AsyncIterator[RunEvent]:
        """Yield run events in the order produced by the AgentLane runtime."""
        ...

    async def result(self) -> RunResult:
        """Return the completed run result or raise the terminal run error."""
        ...


@runtime_checkable
class _ClosableStream(Protocol):
    """Optional cooperative close surface implemented by live run streams."""

    async def aclose(self) -> None:
        """Ask the stream to stop producing events and release run resources."""
        ...


class AgentRuntime(Protocol):
    """Minimal agent surface needed by the bridge backend.

    App-specific agents can satisfy this protocol without inheriting from a
    bridge base class, which keeps the bridge package independent of host code.
    """

    @property
    def run_state(self) -> RunState | None:
        """Return the current conversation state, if the runtime tracks one."""
        ...

    def reset(self) -> None:
        """Reset conversation state after active bridge work has been cancelled."""
        ...

    async def run_events(
        self,
        input: str,
        /,
        *,
        approval_events: AsyncIterator[ToolApprovalEvent],
        cancellation_token: CancellationToken | None = None,
    ) -> RunEventStreamLike:
        """Start a run and return its event stream.

        The runtime must consume approval decisions from ``approval_events`` and
        observe ``cancellation_token`` when provided by the bridge.
        """
        ...


class ReadyMetadataProvider(Protocol):
    """Supplies optional metadata for the initial bridge ``ready`` event.

    Hosts can use this callback to expose app/runtime facts that help the
    TypeScript side configure itself before the first command is sent. Returned
    values must be JSON-serializable because they are written directly into the
    NDJSON protocol payload.
    """

    def __call__(self) -> dict[str, object] | Awaitable[dict[str, object]]:
        """Return ready-event metadata synchronously or asynchronously."""
        ...


class BridgeCommandBackend(Protocol):
    """Narrow backend surface consumed by command handlers.

    Handlers depend on behavior instead of the concrete ``BridgeBackend`` so new
    command types can stay small and independently testable.
    """

    agent: AgentRuntime
    events: EventWriter
    approvals: ToolApprovalBroker
    config: RuntimeConfigStore | None

    def clear_completed_run(self) -> None:
        """Release a finished active-run task before command validation.

        Implementations must not cancel running work here; this is only lazy
        cleanup for tasks that already emitted their terminal outcome.
        """
        ...

    def has_active_run(self) -> bool:
        """Return whether a prompt run is currently owned by the backend.

        Implementations may clear completed tasks before answering so command
        handlers can immediately accept a new prompt after a terminal event.
        """
        ...

    def active_run_is_cancelling(self) -> bool:
        """Return whether the active run has already been asked to cancel."""
        ...

    async def start_prompt_run(self, prompt: str) -> None:
        """Start ``prompt`` asynchronously and mark it as the active run.

        Callers are responsible for checking ``has_active_run()`` first; this
        method assumes the prompt command has already been validated. It returns
        only after the run has emitted its acknowledgement and entered the
        cancellation guard, so a following cancel/reset command cannot strand
        the client without a terminal event.
        """
        ...

    def request_active_run_cancel(self, *, emit_terminal: bool) -> None:
        """Ask the active run to cancel without waiting for cleanup.

        ``emit_terminal`` controls whether the run task should still send a
        terminal cancellation/error event. Use ``False`` only when the transport
        is already closing and additional protocol output would be unsafe.
        """
        ...

    def reset_encoder_turns(self) -> None:
        """Clear per-turn run-event encoder state after an agent reset."""
        ...

    async def cancel_active_run(self, *, emit_terminal: bool) -> None:
        """Cancel the active run and wait until its teardown has completed.

        This stronger form is for reset, shutdown, and close paths that need
        approval waiters and terminal run events settled before continuing.
        """
        ...

    async def deny_pending_approvals(self, reason: str) -> None:
        """Resolve every currently pending approval request as denied."""
        ...

    def config_snapshot_payload(self) -> dict[str, object]:
        """Return ``{"config": snapshot}`` when a store is registered."""
        ...


@dataclass(frozen=True, slots=True)
class BridgeCommandHandler(ABC):
    """One extension point for a bridge command type."""

    command_type: type[object]

    @abstractmethod
    async def handle(self, backend: BridgeCommandBackend, command: object) -> None:
        """Validate and apply one parsed command to the bridge backend."""


class BridgeBackend:
    """Stateful bridge backend serving one TypeScript client process.

    The backend permits one active prompt run at a time. The run task owns its
    cancellation token and streams encoded AgentLane events through the writer
    while command handlers remain free to accept approvals, cancellation, reset,
    and shutdown commands.
    """

    def __init__(
        self,
        *,
        agent: AgentRuntime,
        events: EventWriter,
        ready_metadata: ReadyMetadataProvider | None = None,
        command_handlers: tuple[BridgeCommandHandler, ...] | None = None,
        approvals: ToolApprovalBroker | None = None,
        config: RuntimeConfigStore | None = None,
    ) -> None:
        self.agent = agent
        self.events = events
        self.ready_metadata = ready_metadata
        self.config = config
        # Approval callbacks and bridge commands must share the same broker
        # instance. Otherwise tool calls wait on one broker while client
        # decisions resolve against another one.
        self.approvals = ToolApprovalBroker() if approvals is None else approvals
        self._encoder = RunEventEncoder()
        self._command_handlers = (
            BRIDGE_COMMAND_HANDLERS if command_handlers is None else command_handlers
        )
        # The task is the single source of truth for active-run ownership.
        # Completed tasks are cleared lazily before accepting commands so the
        # terminal event has a chance to flush before the next prompt starts.
        self._active_run: asyncio.Task[None] | None = None
        self._cancel_terminal_event = True

    async def start(self) -> None:
        """Emit the initial ready event."""
        metadata = await self._resolve_ready_metadata()
        verbatim_payload = {
            "metadata": metadata,
            **self.config_snapshot_payload(),
        }
        await self.events.emit(
            BridgeEventType.READY,
            version=_package_version(),
            package="agentlane-process-bridge",
            verbatim_payload=verbatim_payload,
        )

    async def close(self, *, emit_terminal: bool = False) -> None:
        """Close active work and approval streams.

        ``emit_terminal=False`` is used for EOF/dead-client paths where stdout
        should not receive more protocol traffic.
        """
        await self.cancel_active_run(emit_terminal=emit_terminal)
        await self.deny_pending_approvals("Bridge backend closed.")
        self.approvals.close()
        await self.events.aclose()

    async def handle_command(self, command: BridgeCommand) -> None:
        """Dispatch one parsed inbound command to its registered handler."""
        for handler in self._command_handlers:
            if isinstance(command, handler.command_type):
                await handler.handle(self, command)
                return

        # Unknown commands are still well-formed protocol values, so report a
        # command-scoped error instead of raising out of the transport loop.
        await self.events.emit(
            BridgeEventType.ERROR,
            message=f"Unknown command: {command.type}",
            scope=ERROR_SCOPE_COMMAND,
        )

    async def run_prompt(
        self,
        prompt: str,
        *,
        started: asyncio.Future[None],
    ) -> None:
        """Run one prompt and stream AgentLane run events to the client.

        This task owns the run cancellation token. It emits exactly one terminal
        run outcome unless cancellation or failure handling has already emitted
        one while unwinding the stream.
        """
        stream: RunEventStreamLike | None = None
        token = CancellationToken()

        try:
            started.set_result(None)

            stream = await self.agent.run_events(
                prompt,
                approval_events=self.approvals.events(),
                cancellation_token=token,
            )

            async for event in stream:
                await self._emit_run_event(event)

            # The stream can finish before the final result is available. Await
            # it here so failures are reported at the run boundary, not leaked
            # as unobserved task exceptions.
            result = await stream.result()
        except asyncio.CancelledError:
            token.cancel()
            await self._handle_cancelled_run(stream)
            return
        except Exception as exc:
            _logger.exception("bridge_run_failed")
            await self._handle_failed_run(stream, exc)
            return
        else:
            await self.events.emit(
                BridgeEventType.RUN_COMPLETE,
                final_output=str(result.final_output),
                turn_count=result.turn_count,
                response_count=len(result.responses),
                shim_state=_run_shim_state(result.run_state or self.agent.run_state),
            )
        finally:
            # A stale run task can finish after a new run has started. Only the
            # task that still owns the active slot may clear it.
            if self._active_run is asyncio.current_task():
                self._active_run = None

            self._cancel_terminal_event = True

    def clear_completed_run(self) -> None:
        """Forget a completed run task before accepting the next command."""
        if self._active_run is not None and self._active_run.done():
            self._active_run = None

    def has_active_run(self) -> bool:
        """Return whether a run is currently active."""
        self.clear_completed_run()
        return self._active_run is not None

    def active_run_is_cancelling(self) -> bool:
        """Return whether the active run is already cancelling."""
        self.clear_completed_run()
        return self._active_run is not None and self._active_run.cancelling() > 0

    async def start_prompt_run(self, prompt: str) -> None:
        """Start one prompt run in the background."""
        await self.events.emit(BridgeEventType.RUN_START, prompt=prompt)

        # The started future is set as the first statement in `run_prompt`, inside
        # its cancellation guard. Waiting for it avoids the race where a following
        # cancel/reset command cancels the task before it can emit run_cancelled.
        started = asyncio.get_running_loop().create_future()
        self._active_run = asyncio.create_task(
            self.run_prompt(prompt, started=started),
        )
        await started

    def request_active_run_cancel(self, *, emit_terminal: bool) -> None:
        """Request cancellation without waiting for run teardown.

        Cancel commands use this non-blocking path so the client receives an
        immediate acknowledgement while the run task performs stream cleanup.
        """
        if self._active_run is None:
            return

        self._cancel_terminal_event = emit_terminal
        self._active_run.cancel()

    def reset_encoder_turns(self) -> None:
        """Reset encoder state for the next conversation."""
        self._encoder.reset_turns()

    async def cancel_active_run(self, *, emit_terminal: bool) -> None:
        """Cancel an active run and wait for its teardown.

        Reset, shutdown, and backend close use this path when they must know the
        run has finished draining approvals and terminal events before moving on.
        """
        self.clear_completed_run()
        if self._active_run is None:
            return

        self.request_active_run_cancel(emit_terminal=emit_terminal)
        try:
            await self._active_run
        except asyncio.CancelledError:
            pass
        except (BrokenPipeError, OSError):
            # A dying client can close stdout while the terminal cancellation
            # event is being emitted. Treat that as transport teardown rather
            # than a command-validation failure.
            _logger.exception("bridge_cancel_terminal_emit_failed")
        finally:
            self._active_run = None

    async def _handle_cancelled_run(self, stream: RunEventStreamLike | None) -> None:
        try:
            await _close_run_stream(stream)
        except Exception as exc:
            _logger.exception("run_stream_close_failed")
            await self.deny_pending_approvals("Run cancelled.")

            if self._cancel_terminal_event:
                await self.events.emit(
                    BridgeEventType.ERROR,
                    message=f"Run cancellation cleanup failed: {exc}",
                    scope=ERROR_SCOPE_RUN,
                )
            return

        # Approval waiters must be released even when the model/tool run exits
        # through cooperative cancellation rather than a normal result.
        await self.deny_pending_approvals("Run cancelled.")

        if self._cancel_terminal_event:
            await self.events.emit(BridgeEventType.RUN_CANCELLED)

    async def _handle_failed_run(
        self,
        stream: RunEventStreamLike | None,
        error: Exception,
    ) -> None:
        cleanup_error: Exception | None = None

        try:
            await _close_run_stream(stream)
        except Exception as exc:
            cleanup_error = exc
            _logger.exception("run_stream_close_failed")

        # Keep the original run failure as the primary user-facing error. If
        # cleanup also failed, append it instead of replacing the root cause.
        await self.deny_pending_approvals("Run failed.")
        await self.events.emit(
            BridgeEventType.ERROR,
            message=_run_error_message(error, cleanup_error),
            scope=ERROR_SCOPE_RUN,
        )

    async def deny_pending_approvals(self, reason: str) -> None:
        """Resolve currently pending approvals as denied.

        The broker owns waiter notification; resolving here is enough to unblock
        any tool call that is waiting on a client decision.
        """
        for record in self.approvals.pending():
            await self.approvals.resolve(
                record.request_id,
                ToolPermissionDecision.deny(reason),
            )

    def config_snapshot_payload(self) -> dict[str, object]:
        """Return a ready/reset payload fragment containing current config."""
        if self.config is None:
            return {}

        return {"config": _snapshot_config(self.config)}

    async def _emit_run_event(self, event: RunEvent) -> None:
        encoded = self._encoder.encode(event)
        if encoded is None:
            return

        await self.events.emit_payload(encoded.type, encoded.payload)

    async def _resolve_ready_metadata(self) -> dict[str, object]:
        if self.ready_metadata is None:
            return {}

        result = self.ready_metadata()
        if inspect.isawaitable(result):
            return await result

        return result


class PromptCommandHandler(BridgeCommandHandler):
    """Handle prompt commands by starting a new AgentLane run."""

    def __init__(self) -> None:
        super().__init__(PromptCommand)

    async def handle(self, backend: BridgeCommandBackend, command: object) -> None:
        prompt_command = cast(PromptCommand, command)
        # A completed run may still occupy the active slot until the next
        # command arrives. Clear it before validating whether a new prompt can
        # start.
        backend.clear_completed_run()

        prompt = prompt_command.text.strip()
        if not prompt:
            await backend.events.emit(
                BridgeEventType.ERROR,
                message="Prompt must not be empty.",
                scope=ERROR_SCOPE_COMMAND,
            )
            return

        if backend.has_active_run():
            await backend.events.emit(
                BridgeEventType.ERROR,
                message="A run is already active.",
                scope=ERROR_SCOPE_COMMAND,
            )
            return

        await backend.start_prompt_run(prompt)


class ApprovalCommandHandler(BridgeCommandHandler):
    """Handle approval decisions for pending tool permission requests."""

    def __init__(self) -> None:
        super().__init__(ApproveCommand)

    async def handle(self, backend: BridgeCommandBackend, command: object) -> None:
        approval_command = cast(ApproveCommand, command)
        request_id = approval_command.request_id

        decision = (
            ToolPermissionDecision.allow()
            if approval_command.allowed
            else ToolPermissionDecision.deny(
                approval_command.reason or "Denied by bridge client.",
            )
        )
        resolved = await backend.approvals.resolve(request_id, decision)

        if not resolved:
            await backend.events.emit(
                BridgeEventType.ERROR,
                message=f"No pending approval request for id {request_id}.",
                scope=ERROR_SCOPE_COMMAND,
            )


class ConfigureCommandHandler(BridgeCommandHandler):
    """Handle runtime configuration patches through the app-owned store."""

    def __init__(self) -> None:
        super().__init__(ConfigureCommand)

    async def handle(self, backend: BridgeCommandBackend, command: object) -> None:
        configure_command = cast(ConfigureCommand, command)
        store = backend.config

        if store is None:
            await _emit_config_result(
                backend,
                ok=False,
                config=None,
                code="unsupported",
                message="Runtime configuration is not supported by this backend.",
            )
            return

        if configure_command.patch is None:
            await _emit_config_result(
                backend,
                ok=False,
                config=_snapshot_config(store),
                code="invalid",
                message="Configure command patch must be a JSON object.",
            )
            return

        try:
            config = store.apply(configure_command.patch)
        except ConfigRejectedError as exc:
            await _emit_config_result(
                backend,
                ok=False,
                config=_snapshot_config(store),
                code="rejected",
                message=str(exc) or "Runtime configuration was rejected.",
            )
            return
        except Exception:
            _logger.exception("bridge_config_apply_failed")
            await _emit_config_result(
                backend,
                ok=False,
                config=_snapshot_config(store),
                code="internal",
                message=_CONFIG_INTERNAL_ERROR_MESSAGE,
            )
            return

        await _emit_config_result(
            backend,
            ok=True,
            config=config,
            code=None,
            message=None,
        )


class CancelCommandHandler(BridgeCommandHandler):
    """Handle cooperative cancellation for the active run."""

    def __init__(self) -> None:
        super().__init__(CancelCommand)

    async def handle(self, backend: BridgeCommandBackend, command: object) -> None:
        del command

        if not backend.has_active_run():
            await backend.events.emit(
                BridgeEventType.CANCEL_IGNORED,
                reason="no active run",
            )
            return

        if backend.active_run_is_cancelling():
            await backend.events.emit(
                BridgeEventType.CANCEL_IGNORED,
                reason="cancellation already in progress",
            )
            return

        backend.request_active_run_cancel(emit_terminal=True)
        # The run task emits the terminal run_cancelled event after stream
        # cleanup. This acknowledgement only confirms that cancellation was
        # accepted.
        await backend.deny_pending_approvals("Run cancelled.")
        await backend.events.emit(BridgeEventType.CANCEL_REQUESTED)


class ResetCommandHandler(BridgeCommandHandler):
    """Handle run cancellation plus agent conversation reset."""

    def __init__(self) -> None:
        super().__init__(ResetCommand)

    async def handle(self, backend: BridgeCommandBackend, command: object) -> None:
        del command

        # Reset is sequenced after run cancellation so the client does not see a
        # fresh conversation state before the previous run's terminal event.
        await backend.cancel_active_run(emit_terminal=True)
        await backend.deny_pending_approvals("Run reset.")
        backend.agent.reset()
        backend.reset_encoder_turns()
        await backend.events.emit(
            BridgeEventType.RESET,
            verbatim_payload=backend.config_snapshot_payload(),
        )


class ShutdownCommandHandler(BridgeCommandHandler):
    """Handle graceful process-bridge shutdown.

    Shutdown is best-effort over a transport that may already be closing. We
    still close approval and writer state locally even when stdout is gone.
    """

    def __init__(self) -> None:
        super().__init__(ShutdownCommand)

    async def handle(self, backend: BridgeCommandBackend, command: object) -> None:
        del command

        try:
            await backend.cancel_active_run(emit_terminal=True)
            await backend.deny_pending_approvals("Bridge shutdown.")
            await backend.events.emit(BridgeEventType.SHUTDOWN)
        except (BrokenPipeError, OSError):
            _logger.exception("bridge_shutdown_transport_failed")
        finally:
            # No more approval events should be accepted after shutdown, even
            # if notifying the client failed because the process pipe closed.
            backend.approvals.close()

            try:
                await backend.events.aclose()
            except (BrokenPipeError, OSError):
                _logger.exception("bridge_shutdown_writer_close_failed")


BRIDGE_COMMAND_HANDLERS: tuple[BridgeCommandHandler, ...] = (
    PromptCommandHandler(),
    ApprovalCommandHandler(),
    ConfigureCommandHandler(),
    CancelCommandHandler(),
    ResetCommandHandler(),
    ShutdownCommandHandler(),
)
"""Default ordered command handlers used by ``BridgeBackend``."""


async def _close_run_stream(stream: RunEventStreamLike | None) -> None:
    """Close a live run stream and drain its result future.

    ``aclose()`` failures are real cleanup failures and propagate to the run
    boundary. ``result()`` failures after close usually contain the original run
    exception or cancellation, so this helper only logs them after cleanup has
    succeeded.
    """
    if stream is None or not isinstance(stream, _ClosableStream):
        return

    await stream.aclose()

    try:
        await stream.result()
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        # After cancellation or a stream failure, the result future often holds
        # the original run error. The caller already reports that error; only
        # aclose() failures are cleanup failures.
        _logger.debug("run_stream_result_drain_failed", error=str(exc))


def _run_error_message(error: Exception, cleanup_error: Exception | None) -> str:
    if cleanup_error is None:
        return str(error)

    return f"{error}; cleanup failed: {cleanup_error}"


def _run_shim_state(state: RunState | None) -> dict[str, object]:
    if state is None:
        return {}

    return dict(state.shim_state)


def _snapshot_config(store: RuntimeConfigStore) -> dict[str, object]:
    try:
        return store.snapshot()
    except Exception as exc:
        raise ContractPayloadError("Runtime config snapshot failed.") from exc


async def _emit_config_result(
    backend: BridgeCommandBackend,
    *,
    ok: bool,
    config: dict[str, object] | None,
    code: ConfigErrorCode | None,
    message: str | None,
) -> None:
    """Emit the sole settlement event for one configure command."""
    error: dict[str, object] | None
    if code is None:
        error = None
    else:
        error = {"code": code, "message": message or ""}

    await backend.events.emit(
        BridgeEventType.CONFIG,
        ok=ok,
        error=error,
        verbatim_payload={"config": config},
    )


def _package_version() -> str:
    try:
        return version("agentlane")
    except PackageNotFoundError:
        return "0.0.0"
