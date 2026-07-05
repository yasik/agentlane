/**
 * Public entrypoint for the TypeScript process bridge package.
 *
 * Exports stay explicit so app consumers see the supported surface and internal
 * helpers do not become accidental API.
 */

export type { BackendSpec, PythonBackendSpec } from "./backend-spec.ts";
export { resolveBackendSpec } from "./backend-spec.ts";
export type {
  BridgeChannel,
  BridgeChannelOptions,
  BridgeChildLike,
  ChannelScheduler,
} from "./channel.ts";
export { createBridgeChannel } from "./channel.ts";
export { createAgentSession } from "./create-agent-session.ts";
export {
  BridgeDecodeError,
  decodeBridgeEventLine,
  isKnownEventType,
  KNOWN_EVENT_TYPES,
  tryDecodeBridgeEventLine,
} from "./decoders.ts";
export type {
  BridgeProcessCallbacks,
  BridgeProcessOptions,
  BridgeProcessWiring,
  BridgeReadableProcess,
} from "./process.ts";
export { spawnBridgeProcess, wireBridgeProcess } from "./process.ts";
export type {
  AgentEndEvent,
  AgentStartEvent,
  ApprovalRequestEvent,
  ApprovalRequestPayload,
  ApprovalResolvedEvent,
  AssistantDeltaEvent,
  BridgeCommand,
  BridgeCommandType,
  BridgeEnvelope,
  BridgeEvent,
  CancelIgnoredEvent,
  CancelRequestedEvent,
  ConfigErrorCode,
  ConfigErrorPayload,
  ConfigEvent,
  DecodedBridgeEvent,
  ErrorEvent,
  ErrorScope,
  HandoffEndEvent,
  HandoffStartEvent,
  KnownBridgeEvent,
  LineageFields,
  LlmEndEvent,
  LlmStartEvent,
  PlanStep,
  PlanUpdatedEvent,
  ProviderEvent,
  ReadyEvent,
  ReasoningDeltaEvent,
  ResetEvent,
  RunCancelledEvent,
  RunCompleteEvent,
  RunEventEvent,
  RunStartEvent,
  ShutdownEvent,
  StateSnapshotEvent,
  TokenUsage,
  ToolArgumentsDeltaEvent,
  ToolEndEvent,
  ToolErrorPayload,
  ToolStartEvent,
  VersionedBridgeCommand,
} from "./protocol.ts";
export {
  encodeBridgeCommand,
  isSupportedProtocolVersion,
  KNOWN_COMMAND_TYPES,
  PROTOCOL_MAJOR,
  PROTOCOL_VERSION,
} from "./protocol.ts";
export type {
  AgentActivity,
  AgentInfo,
  AgentSession,
  AgentSessionOptions,
  ApprovalDecision,
  ApprovalPolicy,
  ApprovalRequest,
  ApprovalResolution,
  ConfigureErrorCode,
  PlanStepStatus,
  PlanUpdate,
  ReadyInfo,
  RunResult,
  SessionClose,
  SessionDiagnostic,
  TextChunk,
  ToolActivity,
  ToolCallInfo,
} from "./session-types.ts";
export {
  ConfigureError,
  RunError,
  SessionClosedError,
  SessionStartError,
  SessionStateError,
} from "./session-types.ts";
