export const PROTOCOL_VERSION = "1.0";
export const PROTOCOL_MAJOR = 1;

// Commands are app-to-backend only. Events are backend-to-app only; keeping the
// unions separate prevents accidental reuse of event payloads as control input.
export const KNOWN_COMMAND_TYPES = [
  "prompt",
  "approve",
  "cancel",
  "reset",
  "shutdown",
] as const;

export type BridgeCommandType = (typeof KNOWN_COMMAND_TYPES)[number];

export type BridgeCommand =
  | { type: "prompt"; text: string }
  | { type: "approve"; id: string; allowed: boolean; reason?: string }
  | { type: "cancel" }
  | { type: "reset" }
  | { type: "shutdown" };

export type VersionedBridgeCommand = BridgeCommand & {
  protocol_version: string;
};

export type ErrorScope = "command" | "run";

export type BridgeEnvelope = {
  protocol_version: string;
  ts: number;
};

type WithMetadata = { metadata?: Record<string, unknown> };

export type ReadyEvent = BridgeEnvelope &
  WithMetadata & {
    type: "ready";
    version: string;
    package: string;
  };

export type RunStartEvent = BridgeEnvelope & {
  type: "run_start";
  prompt: string;
};

export type RunCompleteEvent = BridgeEnvelope & {
  type: "run_complete";
  final_output: string;
  turn_count: number;
  response_count: number;
  shim_state: Record<string, unknown>;
};

export type RunCancelledEvent = BridgeEnvelope & { type: "run_cancelled" };

export type ErrorEvent = BridgeEnvelope & {
  type: "error";
  message: string;
  scope: ErrorScope;
};

export type AssistantDeltaEvent = BridgeEnvelope & {
  type: "assistant_delta";
  text: string;
};

export type ReasoningDeltaEvent = BridgeEnvelope & {
  type: "reasoning_delta";
  text: string;
  provider_event_type: string | null;
  reasoning_signature: string | null;
};

export type ToolArgumentsDeltaEvent = BridgeEnvelope & {
  type: "tool_arguments_delta";
  tool_call_id: string;
  tool_call_index: number | null;
  delta: string;
};

export type ProviderEvent = BridgeEnvelope & {
  type: "provider_event";
  provider_event_type: string | null;
  item_index: number | null;
  item_type: string | null;
  phase: string | null;
};

export type LineageFields = {
  task_id: string;
  parent_task_id: string | null;
  is_root: boolean;
  is_subagent: boolean;
};

export type AgentStartEvent = BridgeEnvelope &
  LineageFields & {
    type: "agent_start";
    agent: string;
    next_turn: number | null;
  };

export type AgentEndEvent = BridgeEnvelope &
  LineageFields & {
    type: "agent_end";
    agent: string;
    final_preview: string | null;
  };

export type LlmStartEvent = BridgeEnvelope &
  LineageFields & {
    type: "llm_start";
    agent: string;
    message_count: number;
  };

export type TokenUsage = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
};

export type LlmEndEvent = BridgeEnvelope &
  LineageFields & {
    type: "llm_end";
    agent: string;
    output_preview: string | null;
    // null when the provider omits usage on this response; treat as missing
    // data rather than zero.
    usage: TokenUsage | null;
  };

export type ToolStartEvent = BridgeEnvelope &
  LineageFields & {
    type: "tool_start";
    agent: string;
    tool: string;
    tool_call_id: string;
    arguments: unknown;
    is_plan: boolean;
    is_delegation: boolean;
  };

export type ToolErrorPayload = {
  message: string;
  kind: string | null;
};

export type PlanStep = {
  status: string;
  step: string;
};

export type ToolEndEvent = BridgeEnvelope &
  LineageFields & {
    type: "tool_end";
    agent: string;
    tool: string;
    tool_call_id: string;
    result: unknown;
    ok: boolean;
    error: ToolErrorPayload | null;
    is_plan: boolean;
    is_delegation: boolean;
  };

export type PlanUpdatedEvent = BridgeEnvelope &
  LineageFields & {
    type: "plan_updated";
    agent: string;
    tool_call_id: string;
    explanation: string | null;
    raw: unknown | null;
    steps: PlanStep[];
    title: string | null;
  };

export type ApprovalRequestPayload = {
  tool_name: string;
  operation: string;
  cwd: string;
  path: string | null;
  command: string | null;
  skill_name: string | null;
  reason: string | null;
  run_id: string | null;
  agent_name: string | null;
  tool_call_id: string | null;
  metadata: Record<string, unknown>;
};

export type ApprovalRequestEvent = BridgeEnvelope & {
  type: "approval_request";
  id: string;
  request: ApprovalRequestPayload;
  reason: string | null;
};

export type ApprovalResolvedEvent = BridgeEnvelope & {
  type: "approval_resolved";
  id: string;
  allowed: boolean;
  request: ApprovalRequestPayload;
  reason: string | null;
};

export type StateSnapshotEvent = BridgeEnvelope & {
  type: "state_snapshot";
  boundary: string;
  turn_count: number;
  history_length: number;
  response_count: number;
  shim_state: Record<string, unknown>;
};

export type HandoffStartEvent = BridgeEnvelope &
  LineageFields & {
    type: "handoff_start";
    agent: string;
    target: string;
    tool: string;
    tool_call_id: string;
  };

export type HandoffEndEvent = BridgeEnvelope &
  LineageFields & {
    type: "handoff_end";
    agent: string;
    target: string;
    tool: string;
    tool_call_id: string;
    final_preview: string | null;
  };

export type ResetEvent = BridgeEnvelope & { type: "reset" };

export type CancelRequestedEvent = BridgeEnvelope & {
  type: "cancel_requested";
};

export type CancelIgnoredEvent = BridgeEnvelope & {
  type: "cancel_ignored";
  reason: string;
};

export type ShutdownEvent = BridgeEnvelope & { type: "shutdown" };

export type RunEventEvent = BridgeEnvelope & {
  type: "run_event";
  run_event_type: string;
};

export type UnknownBridgeEvent = BridgeEnvelope & {
  type: "unknown_event";
  event_type: string;
  payload: Record<string, unknown>;
};

export type BridgeEvent =
  | ReadyEvent
  | RunStartEvent
  | RunCompleteEvent
  | RunCancelledEvent
  | ErrorEvent
  | AssistantDeltaEvent
  | ReasoningDeltaEvent
  | ToolArgumentsDeltaEvent
  | ProviderEvent
  | AgentStartEvent
  | AgentEndEvent
  | LlmStartEvent
  | LlmEndEvent
  | ToolStartEvent
  | ToolEndEvent
  | PlanUpdatedEvent
  | ApprovalRequestEvent
  | ApprovalResolvedEvent
  | StateSnapshotEvent
  | HandoffStartEvent
  | HandoffEndEvent
  | ResetEvent
  | CancelRequestedEvent
  | CancelIgnoredEvent
  | ShutdownEvent
  | RunEventEvent
  | UnknownBridgeEvent;

export type KnownBridgeEvent = Exclude<BridgeEvent, UnknownBridgeEvent>;

export type DecodedBridgeEvent = {
  event: BridgeEvent;
  fallbacks: string[];
};

export function encodeBridgeCommand(command: BridgeCommand): string {
  const payload: VersionedBridgeCommand = {
    ...command,
    protocol_version: PROTOCOL_VERSION,
  };
  return `${JSON.stringify(payload)}\n`;
}

export function isSupportedProtocolVersion(value: string): boolean {
  // Minor versions must stay additive. Older TypeScript hosts can then accept
  // newer Python bridge payloads while fallback diagnostics expose drift.
  const [major] = value.split(".");
  return Number(major) === PROTOCOL_MAJOR;
}
