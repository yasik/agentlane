"""Versioned NDJSON protocol helpers for local AgentLane process bridges."""

import asyncio
import json
import math
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass, field
from itertools import islice
from typing import Literal, TextIO, cast

from strenum import LowercaseStrEnum

from agentlane.harness import HarnessEventType, RunEventKind

PROTOCOL_VERSION = "1.0"
"""Current process-bridge protocol version written on every outbound event."""

PROTOCOL_MAJOR = 1
"""Major protocol version accepted from inbound commands."""

MAX_EVENT_TEXT_CHARS = 5000
"""Maximum string length preserved in one outbound event payload field."""

MAX_EVENT_ITEMS = 50
"""Maximum list or mapping items preserved in one outbound event payload field."""

MAX_TOOL_RESULT_PREVIEW_CHARS = 1800
"""Maximum tool-result characters sent in compact run-event previews."""

MAX_CONTRACT_PAYLOAD_BYTES = 32_768
"""Maximum serialized size for one authoritative protocol payload field."""

RESERVED_EVENT_FIELDS = frozenset({"protocol_version", "type", "ts"})
"""Event envelope keys payloads must not overwrite."""

type CommandType = Literal[
    "approve",
    "cancel",
    "configure",
    "prompt",
    "reset",
    "shutdown",
]
type ErrorScope = Literal["command", "run"]

COMMAND_APPROVE: CommandType = "approve"
"""Inbound command type for resolving a pending tool approval request."""

COMMAND_CANCEL: CommandType = "cancel"
"""Inbound command type for requesting active-run cancellation."""

COMMAND_CONFIGURE: CommandType = "configure"
"""Inbound command type for applying runtime configuration state."""

COMMAND_PROMPT: CommandType = "prompt"
"""Inbound command type for starting a prompt run."""

COMMAND_RESET: CommandType = "reset"
"""Inbound command type for cancelling active work and resetting conversation state."""

COMMAND_SHUTDOWN: CommandType = "shutdown"
"""Inbound command type for graceful process-bridge shutdown."""

COMMAND_TYPES: frozenset[CommandType] = frozenset(
    {
        COMMAND_APPROVE,
        COMMAND_CANCEL,
        COMMAND_CONFIGURE,
        COMMAND_PROMPT,
        COMMAND_RESET,
        COMMAND_SHUTDOWN,
    }
)
"""Supported inbound command vocabulary for this bridge protocol version."""

ERROR_SCOPE_COMMAND: ErrorScope = "command"
"""Error scope for malformed commands or command-handler failures."""

ERROR_SCOPE_RUN: ErrorScope = "run"
"""Error scope for agent/runtime failures while a prompt run is active."""


class BridgeEventType(LowercaseStrEnum):
    """Stable wire event names emitted by the process bridge."""

    # Bridge backend is ready to accept commands.
    READY = "ready"
    # A user prompt run has started.
    RUN_START = HarnessEventType.RUN_START.value
    # A user prompt run completed with a result.
    RUN_COMPLETE = HarnessEventType.RUN_COMPLETE.value
    # A user prompt run was cancelled.
    RUN_CANCELLED = HarnessEventType.RUN_CANCELLED.value
    # A cancellation request was accepted.
    CANCEL_REQUESTED = "cancel_requested"
    # A cancellation request had no active target.
    CANCEL_IGNORED = "cancel_ignored"
    # A runtime configuration patch settled.
    CONFIG = "config"
    # Conversation state was reset.
    RESET = "reset"
    # The bridge is shutting down.
    SHUTDOWN = "shutdown"
    # A command, run, or model error occurred.
    ERROR = HarnessEventType.ERROR.value

    # An agent task started.
    AGENT_START = RunEventKind.AGENT_START.value
    # An agent task ended.
    AGENT_END = RunEventKind.AGENT_END.value
    # A model request started.
    LLM_START = RunEventKind.LLM_START.value
    # A model request ended.
    LLM_END = RunEventKind.LLM_END.value
    # A tool call started.
    TOOL_START = RunEventKind.TOOL_START.value
    # A tool call ended.
    TOOL_END = RunEventKind.TOOL_END.value
    # A handoff transfer started.
    HANDOFF_START = RunEventKind.HANDOFF_START.value
    # A handoff transfer ended.
    HANDOFF_END = RunEventKind.HANDOFF_END.value
    # A compact run state snapshot was emitted.
    STATE_SNAPSHOT = RunEventKind.STATE_SNAPSHOT.value
    # A structured plan update was emitted.
    PLAN_UPDATED = RunEventKind.PLAN_UPDATED.value

    # Assistant-visible text streamed from the model.
    ASSISTANT_DELTA = HarnessEventType.ASSISTANT_DELTA.value
    # Reasoning text or metadata streamed from the model.
    REASONING_DELTA = HarnessEventType.REASONING_DELTA.value
    # Tool-call arguments streamed from the model.
    TOOL_ARGUMENTS_DELTA = HarnessEventType.TOOL_ARGUMENTS_DELTA.value
    # Provider-native stream metadata was observed.
    PROVIDER_EVENT = HarnessEventType.PROVIDER_EVENT.value
    # A tool approval request is waiting for a decision.
    APPROVAL_REQUEST = HarnessEventType.APPROVAL_REQUEST.value
    # A tool approval request was resolved.
    APPROVAL_RESOLVED = HarnessEventType.APPROVAL_RESOLVED.value
    # Fallback wrapper for an unknown future run event.
    RUN_EVENT = HarnessEventType.RUN_EVENT.value


BRIDGE_EVENT_TYPES: frozenset[BridgeEventType] = frozenset(BridgeEventType)
"""Complete set of bridge event names this package may emit."""

_STREAMING_EVENT_TYPES: frozenset[BridgeEventType] = frozenset(
    {
        BridgeEventType.ASSISTANT_DELTA,
        BridgeEventType.PROVIDER_EVENT,
        BridgeEventType.REASONING_DELTA,
        BridgeEventType.TOOL_ARGUMENTS_DELTA,
    }
)
"""High-volume events that can be batched without forcing an immediate drain."""

_WRITE_BATCH_SIZE = 64
"""Maximum number of queued NDJSON lines written by one worker batch."""


class ContractPayloadError(ValueError):
    """Authoritative protocol payload could not be emitted without corruption."""


@dataclass(frozen=True, slots=True)
class ProtocolError:
    """Parse failure for one inbound protocol line."""

    message: str


@dataclass(frozen=True, slots=True)
class PromptCommand:
    """Prompt command from the TypeScript host."""

    text: str
    type: Literal["prompt"] = field(default="prompt", init=False)


@dataclass(frozen=True, slots=True)
class ApproveCommand:
    """Approval-resolution command from the TypeScript host."""

    request_id: str
    allowed: bool
    reason: str | None = None
    type: Literal["approve"] = field(default="approve", init=False)


@dataclass(frozen=True, slots=True)
class ConfigureCommand:
    """Runtime configuration patch from the TypeScript host."""

    patch: dict[str, object] | None
    type: Literal["configure"] = field(default="configure", init=False)


@dataclass(frozen=True, slots=True)
class CancelCommand:
    """Request cancellation of the active run."""

    type: Literal["cancel"] = field(default="cancel", init=False)


@dataclass(frozen=True, slots=True)
class ResetCommand:
    """Reset conversation state after cancelling active work."""

    type: Literal["reset"] = field(default="reset", init=False)


@dataclass(frozen=True, slots=True)
class ShutdownCommand:
    """Request graceful bridge shutdown."""

    type: Literal["shutdown"] = field(default="shutdown", init=False)


@dataclass(frozen=True, slots=True)
class UnknownCommand:
    """Well-formed command whose type this backend does not know."""

    type: str


type BridgeCommand = (
    PromptCommand
    | ApproveCommand
    | ConfigureCommand
    | CancelCommand
    | ResetCommand
    | ShutdownCommand
    | UnknownCommand
)


@dataclass(frozen=True, slots=True)
class BridgeCommandParser(ABC):
    """One parser for a bridge command payload."""

    type: CommandType

    @abstractmethod
    def parse(self, payload: dict[str, object]) -> BridgeCommand:
        """Convert a parsed JSON payload into a typed bridge command.

        Parsers receive the raw object after protocol-version validation. They
        should coerce only their own command fields and leave command-level
        validation to the backend handler.
        """


class PromptCommandParser(BridgeCommandParser):
    """Parse prompt commands."""

    def __init__(self) -> None:
        super().__init__(COMMAND_PROMPT)

    def parse(self, payload: dict[str, object]) -> PromptCommand:
        return PromptCommand(text=_coerce_text(payload, "text", default=""))


class ApproveCommandParser(BridgeCommandParser):
    """Parse tool approval commands."""

    def __init__(self) -> None:
        super().__init__(COMMAND_APPROVE)

    def parse(self, payload: dict[str, object]) -> ApproveCommand:
        return ApproveCommand(
            request_id=_coerce_text(payload, "id", default=""),
            allowed=_coerce_allowed(payload),
            reason=_optional_text(payload, "reason"),
        )


class ConfigureCommandParser(BridgeCommandParser):
    """Parse runtime configuration commands."""

    def __init__(self) -> None:
        super().__init__(COMMAND_CONFIGURE)

    def parse(self, payload: dict[str, object]) -> ConfigureCommand:
        patch = payload["patch"] if "patch" in payload else None

        if not isinstance(patch, dict):
            return ConfigureCommand(patch=None)

        # The bridge validates only the top-level JSON shape. Key meaning and
        # deeper structure belong to the app-owned RuntimeConfigStore.
        return ConfigureCommand(patch=cast(dict[str, object], patch))


class CancelCommandParser(BridgeCommandParser):
    """Parse cancellation commands."""

    def __init__(self) -> None:
        super().__init__(COMMAND_CANCEL)

    def parse(self, payload: dict[str, object]) -> CancelCommand:
        del payload

        return CancelCommand()


class ResetCommandParser(BridgeCommandParser):
    """Parse reset commands."""

    def __init__(self) -> None:
        super().__init__(COMMAND_RESET)

    def parse(self, payload: dict[str, object]) -> ResetCommand:
        del payload

        return ResetCommand()


class ShutdownCommandParser(BridgeCommandParser):
    """Parse shutdown commands."""

    def __init__(self) -> None:
        super().__init__(COMMAND_SHUTDOWN)

    def parse(self, payload: dict[str, object]) -> ShutdownCommand:
        del payload

        return ShutdownCommand()


COMMAND_PARSERS: tuple[BridgeCommandParser, ...] = (
    PromptCommandParser(),
    ApproveCommandParser(),
    ConfigureCommandParser(),
    CancelCommandParser(),
    ResetCommandParser(),
    ShutdownCommandParser(),
)
"""Default ordered parsers for known inbound command types."""

COMMAND_PARSERS_BY_TYPE: dict[str, BridgeCommandParser] = {
    parser.type: parser for parser in COMMAND_PARSERS
}
"""Lookup table used to route a parsed command payload to its parser."""


@dataclass(slots=True)
class EventWriter:
    """Async-safe NDJSON writer for bridge events.

    Events are queued and drained by one background worker. Lifecycle/control
    events wait for the queue to flush; high-volume streaming deltas can batch
    behind them without blocking the event loop on a slow local client pipe.
    """

    stream: TextIO
    write_timeout_seconds: float | None = 30.0
    max_queue_size: int = 1024
    _queue: asyncio.Queue[str] | None = field(default=None, init=False)
    _worker: asyncio.Task[None] | None = field(default=None, init=False)
    _failed: BaseException | None = field(default=None, init=False)
    _closed: bool = field(default=False, init=False)

    async def emit(
        self,
        event_type: BridgeEventType,
        *,
        verbatim_payload: dict[str, object] | None = None,
        **payload: object,
    ) -> None:
        """Write one versioned protocol event as a newline-terminated JSON object."""
        await self.emit_payload(
            event_type,
            payload,
            verbatim_payload=verbatim_payload,
        )

    async def emit_payload(
        self,
        event_type: BridgeEventType,
        payload: dict[str, object],
        *,
        verbatim_payload: dict[str, object] | None = None,
    ) -> None:
        """Write one event from an already assembled payload dictionary."""
        self._raise_if_failed()
        event = build_event(event_type, payload, verbatim_payload=verbatim_payload)
        line = json.dumps(
            event,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            default=str,
        )
        await self._enqueue(line)

        # Lifecycle and control events should be observable before the caller
        # continues. Streaming deltas may batch to avoid blocking the event loop
        # on every token from a slow local pipe.
        if event_type not in _STREAMING_EVENT_TYPES:
            await self.drain()

    async def drain(self) -> None:
        """Wait until all queued lines have been written."""
        queue = self._queue
        if queue is None:
            return

        try:
            if self.write_timeout_seconds is None:
                await queue.join()
            else:
                await asyncio.wait_for(
                    queue.join(),
                    timeout=self.write_timeout_seconds,
                )
        except TimeoutError:
            # Once the writer times out, every later emit should fail fast with
            # the same broken-pipe semantics as a closed client.
            self._failed = BrokenPipeError("Bridge event writer timed out.")
            raise

        self._raise_if_failed()

    async def aclose(self) -> None:
        """Flush queued events and stop the background writer."""
        if self._closed:
            return

        self._closed = True
        try:
            await self.drain()
        finally:
            if self._worker is not None:
                self._worker.cancel()
                with suppress(asyncio.CancelledError):
                    await self._worker

    async def _enqueue(self, line: str) -> None:
        if self._closed:
            raise BrokenPipeError("Bridge event writer is closed.")

        self._raise_if_failed()
        queue = self._ensure_queue()

        try:
            if self.write_timeout_seconds is None:
                await queue.put(line)
            else:
                await asyncio.wait_for(
                    queue.put(line),
                    timeout=self.write_timeout_seconds,
                )
        except TimeoutError:
            # Backpressure past the configured timeout means the downstream
            # process is no longer draining bridge output reliably.
            self._failed = BrokenPipeError("Bridge event writer queue is full.")
            raise

    def _ensure_queue(self) -> asyncio.Queue[str]:
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.max_queue_size)

        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._write_worker())

        return self._queue

    async def _write_worker(self) -> None:
        queue = self._ensure_queue()
        while True:
            first = await queue.get()
            batch = [first]
            await asyncio.sleep(0)

            # Give other tasks one scheduling point, then coalesce whatever
            # arrived into a bounded write batch.
            while len(batch) < _WRITE_BATCH_SIZE:
                try:
                    batch.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            try:
                await self._write_batch(batch)
            except TimeoutError as exc:
                self._failed = BrokenPipeError("Bridge event writer timed out.")
                self._failed.__cause__ = exc
            except (BrokenPipeError, OSError) as exc:
                self._failed = exc
            finally:
                for _ in batch:
                    queue.task_done()

            if self._failed is not None:
                # Join waiters must be released after a write failure; later
                # calls observe the stored exception through _raise_if_failed().
                self._discard_queued_lines(queue)

    async def _write_batch(self, lines: list[str]) -> None:
        write = asyncio.ensure_future(asyncio.to_thread(self._write_lines, lines))
        try:
            if self.write_timeout_seconds is None:
                await asyncio.shield(write)
            else:
                await asyncio.wait_for(
                    asyncio.shield(write),
                    timeout=self.write_timeout_seconds,
                )
        except asyncio.CancelledError:
            await write
            raise

    def _write_lines(self, lines: list[str]) -> None:
        for line in lines:
            self.stream.write(line + "\n")
        self.stream.flush()

    def _discard_queued_lines(self, queue: asyncio.Queue[str]) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._failed is not None:
            raise self._failed


def build_event(
    event_type: BridgeEventType,
    payload: dict[str, object],
    *,
    timestamp: float | None = None,
    verbatim_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one flat versioned event object."""
    ts = time.time() if timestamp is None else timestamp
    verbatim_fields = _validated_verbatim_payload(verbatim_payload)
    reserved_fields = (set(payload) | set(verbatim_fields)) & RESERVED_EVENT_FIELDS
    if reserved_fields:
        field_names = ", ".join(sorted(reserved_fields))
        raise ContractPayloadError(
            f"Contract payload fields overlap event envelope fields: {field_names}.",
        )

    overlapping_fields = set(payload) & set(verbatim_fields)
    if overlapping_fields:
        field_names = ", ".join(sorted(overlapping_fields))
        raise ContractPayloadError(
            f"Contract payload fields overlap regular payload fields: {field_names}.",
        )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": event_type.value,
        "ts": round(ts, 3),
        **_truncate_payload(payload),
        **verbatim_fields,
    }


def parse_command_line(line: str) -> BridgeCommand | ProtocolError:
    """Parse one inbound NDJSON command line.

    A supported ``protocol_version`` is required before command handling.
    """
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return ProtocolError("Invalid JSON command.")

    # The bridge protocol is intentionally flat NDJSON. Arrays, strings, and
    # other JSON values cannot carry versioned command fields safely.
    if not isinstance(value, dict):
        return ProtocolError("Command must be a JSON object.")

    payload = cast(dict[str, object], value)
    command_type = payload["type"] if "type" in payload else None

    if not isinstance(command_type, str):
        return ProtocolError("Command is missing a string type.")

    version = payload["protocol_version"] if "protocol_version" in payload else None

    # Version checks happen before dispatch so newer clients fail with an
    # explicit protocol error instead of partially executing an unknown command.
    if not _supports_protocol_version(version):
        return ProtocolError(f"Unsupported protocol version: {version}")

    return _parse_command_payload(command_type, payload)


def _parse_command_payload(
    command_type: str,
    payload: dict[str, object],
) -> BridgeCommand:
    if command_type in COMMAND_PARSERS_BY_TYPE:
        return COMMAND_PARSERS_BY_TYPE[command_type].parse(payload)

    # Preserve the command type for a structured backend error. The parser's job
    # is not to decide whether unknown future command names should terminate the
    # bridge.
    return UnknownCommand(type=command_type)


def _coerce_text(
    payload: dict[str, object],
    key: str,
    *,
    default: str,
) -> str:
    if key not in payload:
        return default

    return str(payload[key])


def _optional_text(payload: dict[str, object], key: str) -> str | None:
    if key not in payload:
        return None

    value = payload[key]

    if value is None:
        return None

    return str(value)


def _coerce_allowed(payload: dict[str, object]) -> bool:
    if "allowed" not in payload:
        return False

    return payload["allowed"] is True


def _supports_protocol_version(value: object) -> bool:
    if not isinstance(value, str):
        return False

    major_text = value.split(".", maxsplit=1)[0]

    try:
        major = int(major_text)
    except ValueError:
        return False

    # Minor versions are expected to be additive, but a major version change may
    # rename or reinterpret command fields.
    return major == PROTOCOL_MAJOR


def _truncate_payload(payload: dict[str, object]) -> dict[str, object]:
    return {key: _truncate_value(value) for key, value in payload.items()}


def _validated_verbatim_payload(
    payload: dict[str, object] | None,
) -> dict[str, object]:
    if payload is None:
        return {}

    for key, value in payload.items():
        _validate_contract_payload_field(key, value)

    return payload


def _validate_contract_payload_field(key: str, value: object) -> None:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ContractPayloadError(
            f"Contract payload field {key!r} is not JSON-serializable.",
        ) from exc

    byte_count = len(serialized.encode())
    if byte_count > MAX_CONTRACT_PAYLOAD_BYTES:
        raise ContractPayloadError(
            f"Contract payload field {key!r} exceeds "
            f"{MAX_CONTRACT_PAYLOAD_BYTES} bytes.",
        )


def _truncate_value(value: object) -> object:
    """Bound payload sizes while leaving explicit truncation markers."""
    if value is None or isinstance(value, (int, bool)):
        return value

    # JSON forbids NaN and Infinity when allow_nan=False, so preserve them as
    # readable strings instead of failing an entire event.
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)

    if isinstance(value, str) and len(value) > MAX_EVENT_TEXT_CHARS:
        omitted = len(value) - MAX_EVENT_TEXT_CHARS
        return (
            value[:MAX_EVENT_TEXT_CHARS].rstrip()
            + f"\n[truncated, +{omitted} more chars]"
        )

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        items = cast(list[object], value)
        truncated_items = [_truncate_value(item) for item in items[:MAX_EVENT_ITEMS]]

        if len(items) > MAX_EVENT_ITEMS:
            truncated_items.append(f"... (+{len(items) - MAX_EVENT_ITEMS} more)")

        return truncated_items

    if isinstance(value, tuple):
        # Tuples are not a JSON shape; normalize them through the list branch so
        # the same item limits and recursive truncation apply.
        return _truncate_value(list(cast(tuple[object, ...], value)))

    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        truncated_mapping = {
            str(key): _truncate_value(item)
            for key, item in islice(mapping.items(), MAX_EVENT_ITEMS)
        }

        if len(mapping) > MAX_EVENT_ITEMS:
            truncated_mapping["..."] = f"+{len(mapping) - MAX_EVENT_ITEMS} more"

        return truncated_mapping

    # Last-resort values still need to be protocol-safe and bounded.
    return _truncate_value(str(value))
