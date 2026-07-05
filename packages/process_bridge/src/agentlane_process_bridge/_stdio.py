"""Stdio entrypoints for the AgentLane process bridge."""

import asyncio
import logging
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import TextIO

import structlog

from agentlane.harness.tools import ToolApprovalBroker

from ._backend import (
    AgentRuntime,
    BridgeBackend,
    ReadyMetadataProvider,
    RuntimeConfigStore,
)
from ._protocol import (
    ERROR_SCOPE_COMMAND,
    BridgeEventType,
    ContractPayloadError,
    EventWriter,
    ProtocolError,
    ShutdownCommand,
    parse_command_line,
)

_logger = structlog.get_logger(__name__)

MAX_COMMAND_LINE_CHARS = 1_000_000
"""Maximum inbound NDJSON command line length accepted from stdin."""


@dataclass(frozen=True, slots=True)
class AgentBackend:
    """Application backend returned by `--app` factories.

    The factory owns Python-side agent construction. Returning this typed object
    makes broker sharing explicit when an app wires tool approvals into its
    agent and the process bridge.
    """

    agent: AgentRuntime
    approvals: ToolApprovalBroker | None = None
    ready_metadata: ReadyMetadataProvider | None = None
    config: RuntimeConfigStore | None = None


async def serve_stdio(
    backend: BridgeBackend,
    *,
    readline: Callable[[int], str],
    max_command_line_chars: int = MAX_COMMAND_LINE_CHARS,
) -> None:
    """Serve one NDJSON command loop until shutdown, EOF, or a dead pipe."""
    should_close = True
    try:
        while True:
            try:
                # Read in a worker thread so synchronous TextIO streams do not
                # block the event loop that also owns run events and approvals.
                line = await asyncio.to_thread(
                    readline,
                    max_command_line_chars + 1,
                )
            except Exception:
                _logger.exception("bridge_read_failed")
                return

            if line == "":
                # EOF means the client process is gone. The backend should stop
                # local work without trying to emit more protocol traffic.
                return

            if len(line) > max_command_line_chars:
                # Drain the rest of the oversized NDJSON record before reading
                # the next command, otherwise the remainder would be parsed as a
                # second malformed command.
                await _discard_oversized_line_remainder(
                    readline,
                    line=line,
                    chunk_size=max_command_line_chars + 1,
                )

                if not await _report_command_error(
                    backend,
                    "Command line exceeds bridge size limit.",
                ):
                    return

                continue

            parsed = parse_command_line(line)

            if isinstance(parsed, ProtocolError):
                # Parse failures are user-fixable command problems; keep the
                # bridge alive as long as the client can still receive errors.
                if not await _report_command_error(backend, parsed.message):
                    return

                continue

            command = parsed

            try:
                await backend.handle_command(command)
            except ContractPayloadError:
                # Contract payload failures mean backend state could not be
                # announced truthfully. Exiting loudly is safer than keeping a
                # live client with silently divergent config state.
                raise
            except Exception as exc:
                # Command handling is the boundary where validation, backend
                # state, and transport errors become structured client feedback.
                _logger.exception("bridge_command_failed", command_type=command.type)

                if not await _report_command_error(backend, f"Command failed: {exc}"):
                    return

                continue

            if isinstance(command, ShutdownCommand):
                # The shutdown handler owns writer/backend close. Running the
                # generic finally close as well would double-close the transport.
                should_close = False
                return
    finally:
        if should_close:
            await _close_after_dead_client(backend)


async def run_stdio(
    *,
    agent: AgentRuntime,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    ready_metadata: ReadyMetadataProvider | None = None,
    approvals: ToolApprovalBroker | None = None,
    config: RuntimeConfigStore | None = None,
) -> None:
    """Run a bridge backend against process stdin/stdout.

    Pass ``approvals`` to share the broker an app wired into its agent's tool
    approval callbacks, so pending requests and ``approve`` commands resolve
    against the same broker instance.
    """
    configure_stderr_logging()
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    backend = BridgeBackend(
        agent=agent,
        events=EventWriter(output_stream),
        ready_metadata=ready_metadata,
        approvals=approvals,
        config=config,
    )
    await backend.start()

    if stdout is None:
        with redirect_stdout(sys.stderr):
            await serve_stdio(backend, readline=input_stream.readline)
    else:
        await serve_stdio(backend, readline=input_stream.readline)


def configure_stderr_logging() -> None:
    """Route diagnostics to stderr so stdout remains valid NDJSON."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


async def _report_command_error(backend: BridgeBackend, message: str) -> bool:
    try:
        await backend.events.emit(
            BridgeEventType.ERROR,
            message=message,
            scope=ERROR_SCOPE_COMMAND,
        )
    except ContractPayloadError:
        raise
    except Exception:
        # If the error itself cannot be written, the client pipe is no longer a
        # reliable recovery channel and the command loop should stop.
        _logger.exception("bridge_error_report_failed")
        return False

    return True


async def _close_after_dead_client(backend: BridgeBackend) -> None:
    try:
        await backend.close(emit_terminal=False)
    except (BrokenPipeError, OSError):
        # Dead-client cleanup intentionally avoids terminal run events, but the
        # low-level writer may still observe a closed pipe while flushing.
        _logger.exception("bridge_close_after_dead_client_failed")


async def _discard_oversized_line_remainder(
    readline: Callable[[int], str],
    *,
    line: str,
    chunk_size: int,
) -> None:
    if line.endswith("\n"):
        return

    while True:
        try:
            chunk = await asyncio.to_thread(readline, chunk_size)
        except Exception:
            return

        if chunk == "" or chunk.endswith("\n"):
            # Either EOF or the record terminator has been reached; the next
            # serve loop iteration can resume at a command boundary.
            return
