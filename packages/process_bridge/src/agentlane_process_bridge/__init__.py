"""Local process bridge for TypeScript apps hosting Python AgentLane agents."""

from ._backend import (
    BRIDGE_COMMAND_HANDLERS,
    AgentRuntime,
    BridgeBackend,
    BridgeCommandBackend,
    BridgeCommandHandler,
    ReadyMetadataProvider,
)
from ._events import (
    RUN_EVENT_BRIDGE_HANDLERS,
    RUN_EVENT_KIND_BRIDGE_EVENT_TYPES,
    BridgeRunEvent,
    RunEventBridgeHandler,
    RunEventEncoder,
    RunEventEncodingContext,
    encode_run_event,
)
from ._protocol import (
    BRIDGE_EVENT_TYPES,
    COMMAND_TYPES,
    ApproveCommand,
    BridgeCommand,
    BridgeEventType,
    CancelCommand,
    EventWriter,
    PromptCommand,
    ProtocolError,
    ResetCommand,
    ShutdownCommand,
    UnknownCommand,
    parse_command_line,
)
from ._stdio import configure_stderr_logging, run_stdio, serve_stdio

__all__ = [
    "AgentRuntime",
    "ApproveCommand",
    "BRIDGE_EVENT_TYPES",
    "BRIDGE_COMMAND_HANDLERS",
    "BridgeBackend",
    "BridgeCommand",
    "BridgeCommandBackend",
    "BridgeCommandHandler",
    "BridgeEventType",
    "BridgeRunEvent",
    "CancelCommand",
    "COMMAND_TYPES",
    "EventWriter",
    "PromptCommand",
    "ProtocolError",
    "ReadyMetadataProvider",
    "ResetCommand",
    "RUN_EVENT_BRIDGE_HANDLERS",
    "RUN_EVENT_KIND_BRIDGE_EVENT_TYPES",
    "RunEventBridgeHandler",
    "RunEventEncodingContext",
    "RunEventEncoder",
    "ShutdownCommand",
    "UnknownCommand",
    "configure_stderr_logging",
    "encode_run_event",
    "parse_command_line",
    "run_stdio",
    "serve_stdio",
]
"""Public API exported by the process bridge package."""
