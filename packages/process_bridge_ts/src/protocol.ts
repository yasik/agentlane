import { z } from "zod";

/** Current bridge protocol version emitted on app-to-backend commands. */
export const PROTOCOL_VERSION = "1.0";

/** Major version accepted by this TypeScript package for backend events. */
export const PROTOCOL_MAJOR = 1;

// Commands are app-to-backend only. Events are backend-to-app only; keeping the
// unions separate prevents accidental reuse of event payloads as control input.
/** Command names the TypeScript app may send to the Python backend. */
export const KNOWN_COMMAND_TYPES = [
  "prompt",
  "approve",
  "cancel",
  "reset",
  "shutdown",
] as const;

/** Literal command-name union derived from `KNOWN_COMMAND_TYPES`. */
export type BridgeCommandType = (typeof KNOWN_COMMAND_TYPES)[number];

/** App-to-backend command payload before protocol metadata is attached. */
export type BridgeCommand =
  | { type: "prompt"; text: string }
  | { type: "approve"; id: string; allowed: boolean; reason?: string }
  | { type: "cancel" }
  | { type: "reset" }
  | { type: "shutdown" };

/** Command payload written to stdin after `encodeBridgeCommand` adds metadata. */
export type VersionedBridgeCommand = BridgeCommand & {
  protocol_version: string;
};

/** Shared envelope present on every backend-to-app event. */
export type BridgeEnvelope = {
  protocol_version: string;
  ts: number;
};

/** Error origin used by bridge error events. */
export type ErrorScope = "command" | "run";

/** Task lineage metadata carried by agent, model, tool, and handoff events. */
export type LineageFields = {
  task_id: string;
  parent_task_id: string | null;
  is_root: boolean;
  is_subagent: boolean;
};

/** Provider token counts when the model backend reports usage metadata. */
export type TokenUsage = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
};

/** Structured tool error payload emitted on failed tool completions. */
export type ToolErrorPayload = {
  message: string;
  kind: string | null;
};

/** One plan row emitted by the AgentLane planning tool. */
export type PlanStep = {
  status: string;
  step: string;
};

/** Approval request metadata displayed by downstream app permission UI. */
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

type BridgeEnvelopeShape = {
  protocol_version: z.ZodString;
  ts: z.ZodNumber;
};

type LineageFieldsShape = {
  task_id: z.ZodString;
  parent_task_id: z.ZodNullable<z.ZodString>;
  is_root: z.ZodBoolean;
  is_subagent: z.ZodBoolean;
};

/** JSON object payload with string keys and bridge-owned unknown values. */
const recordSchema: z.ZodRecord<z.ZodString, z.ZodUnknown> = z.record(
  z.string(),
  z.unknown(),
);

/** Reused nullable string schema for optional provider and tool metadata. */
const nullableStringSchema: z.ZodNullable<z.ZodString> = z.string().nullable();

/** Common schema for the event envelope each backend line must include. */
const bridgeEnvelopeSchema: z.ZodObject<BridgeEnvelopeShape> = z.object({
  protocol_version: z.string(),
  ts: z.number(),
});

/** Schema for the closed set of error scopes emitted by Python. */
const errorScopeSchema: z.ZodType<ErrorScope> = z.enum(["command", "run"]);

/** Shared schema for task lineage fields on task-scoped events. */
const lineageFieldsSchema: z.ZodObject<LineageFieldsShape> = z.object({
  task_id: z.string(),
  parent_task_id: nullableStringSchema,
  is_root: z.boolean(),
  is_subagent: z.boolean(),
});

/** Schema for provider token usage when usage data is available. */
const tokenUsageSchema: z.ZodType<TokenUsage> = z.object({
  prompt_tokens: z.number(),
  completion_tokens: z.number(),
  total_tokens: z.number(),
});

/** Schema for tool failure details nested under `tool_end.error`. */
const toolErrorPayloadSchema: z.ZodType<ToolErrorPayload> = z.object({
  message: z.string(),
  kind: nullableStringSchema,
});

/** Schema for a single planning step emitted by the plan tool. */
const planStepSchema: z.ZodType<PlanStep> = z.object({
  status: z.string(),
  step: z.string(),
});

/** Schema for approval request details nested under approval events. */
const approvalRequestPayloadSchema: z.ZodType<ApprovalRequestPayload> =
  z.object({
    tool_name: z.string(),
    operation: z.string(),
    cwd: z.string(),
    path: nullableStringSchema,
    command: nullableStringSchema,
    skill_name: nullableStringSchema,
    reason: nullableStringSchema,
    run_id: nullableStringSchema,
    agent_name: nullableStringSchema,
    tool_call_id: nullableStringSchema,
    metadata: recordSchema,
  });

/**
 * Strict backend-to-app event schema registry.
 *
 * This map is the TypeScript mirror of the Python bridge event surface. A new
 * event type should add one schema entry here and one representative fixture so
 * parity tests can catch missing or stale protocol updates.
 */
export const BRIDGE_EVENT_SCHEMAS = {
  ready: bridgeEnvelopeSchema.extend({
    type: z.literal("ready"),
    version: z.string(),
    package: z.string(),
    metadata: recordSchema.optional(),
  }),
  run_start: bridgeEnvelopeSchema.extend({
    type: z.literal("run_start"),
    prompt: z.string(),
  }),
  run_complete: bridgeEnvelopeSchema.extend({
    type: z.literal("run_complete"),
    final_output: z.string(),
    turn_count: z.number(),
    response_count: z.number(),
    shim_state: recordSchema,
  }),
  run_cancelled: bridgeEnvelopeSchema.extend({
    type: z.literal("run_cancelled"),
  }),
  error: bridgeEnvelopeSchema.extend({
    type: z.literal("error"),
    message: z.string(),
    scope: errorScopeSchema,
  }),
  assistant_delta: bridgeEnvelopeSchema.extend({
    type: z.literal("assistant_delta"),
    text: z.string(),
  }),
  reasoning_delta: bridgeEnvelopeSchema.extend({
    type: z.literal("reasoning_delta"),
    text: z.string(),
    provider_event_type: nullableStringSchema,
    reasoning_signature: nullableStringSchema,
  }),
  tool_arguments_delta: bridgeEnvelopeSchema.extend({
    type: z.literal("tool_arguments_delta"),
    tool_call_id: z.string(),
    tool_call_index: z.number().nullable(),
    delta: z.string(),
  }),
  provider_event: bridgeEnvelopeSchema.extend({
    type: z.literal("provider_event"),
    provider_event_type: nullableStringSchema,
    item_index: z.number().nullable(),
    item_type: nullableStringSchema,
    phase: nullableStringSchema,
  }),
  agent_start: bridgeEnvelopeSchema.extend({
    type: z.literal("agent_start"),
    ...lineageFieldsSchema.shape,
    agent: z.string(),
    next_turn: z.number().nullable(),
  }),
  agent_end: bridgeEnvelopeSchema.extend({
    type: z.literal("agent_end"),
    ...lineageFieldsSchema.shape,
    agent: z.string(),
    final_preview: nullableStringSchema,
  }),
  llm_start: bridgeEnvelopeSchema.extend({
    type: z.literal("llm_start"),
    ...lineageFieldsSchema.shape,
    agent: z.string(),
    message_count: z.number(),
  }),
  llm_end: bridgeEnvelopeSchema.extend({
    type: z.literal("llm_end"),
    ...lineageFieldsSchema.shape,
    agent: z.string(),
    output_preview: nullableStringSchema,
    // null when the provider omits usage on this response; treat as missing
    // data rather than zero.
    usage: tokenUsageSchema.nullable(),
  }),
  tool_start: bridgeEnvelopeSchema.extend({
    type: z.literal("tool_start"),
    ...lineageFieldsSchema.shape,
    agent: z.string(),
    tool: z.string(),
    tool_call_id: z.string(),
    arguments: z.unknown(),
    is_plan: z.boolean(),
    is_delegation: z.boolean(),
  }),
  tool_end: bridgeEnvelopeSchema.extend({
    type: z.literal("tool_end"),
    ...lineageFieldsSchema.shape,
    agent: z.string(),
    tool: z.string(),
    tool_call_id: z.string(),
    result: z.unknown(),
    ok: z.boolean(),
    error: toolErrorPayloadSchema.nullable(),
    is_plan: z.boolean(),
    is_delegation: z.boolean(),
  }),
  plan_updated: bridgeEnvelopeSchema.extend({
    type: z.literal("plan_updated"),
    ...lineageFieldsSchema.shape,
    agent: z.string(),
    tool_call_id: z.string(),
    explanation: nullableStringSchema,
    raw: z.unknown(),
    steps: z.array(planStepSchema),
    title: nullableStringSchema,
  }),
  approval_request: bridgeEnvelopeSchema.extend({
    type: z.literal("approval_request"),
    id: z.string(),
    request: approvalRequestPayloadSchema,
    reason: nullableStringSchema,
  }),
  approval_resolved: bridgeEnvelopeSchema.extend({
    type: z.literal("approval_resolved"),
    id: z.string(),
    allowed: z.boolean(),
    request: approvalRequestPayloadSchema,
    reason: nullableStringSchema,
  }),
  state_snapshot: bridgeEnvelopeSchema.extend({
    type: z.literal("state_snapshot"),
    boundary: z.string(),
    turn_count: z.number(),
    history_length: z.number(),
    response_count: z.number(),
    shim_state: recordSchema,
  }),
  handoff_start: bridgeEnvelopeSchema.extend({
    type: z.literal("handoff_start"),
    ...lineageFieldsSchema.shape,
    agent: z.string(),
    target: z.string(),
    tool: z.string(),
    tool_call_id: z.string(),
  }),
  handoff_end: bridgeEnvelopeSchema.extend({
    type: z.literal("handoff_end"),
    ...lineageFieldsSchema.shape,
    agent: z.string(),
    target: z.string(),
    tool: z.string(),
    tool_call_id: z.string(),
    final_preview: nullableStringSchema,
  }),
  reset: bridgeEnvelopeSchema.extend({
    type: z.literal("reset"),
  }),
  cancel_requested: bridgeEnvelopeSchema.extend({
    type: z.literal("cancel_requested"),
  }),
  cancel_ignored: bridgeEnvelopeSchema.extend({
    type: z.literal("cancel_ignored"),
    reason: z.string(),
  }),
  shutdown: bridgeEnvelopeSchema.extend({
    type: z.literal("shutdown"),
  }),
  run_event: bridgeEnvelopeSchema.extend({
    type: z.literal("run_event"),
    run_event_type: z.string(),
  }),
} as const;

type BridgeEventSchema =
  (typeof BRIDGE_EVENT_SCHEMAS)[keyof typeof BRIDGE_EVENT_SCHEMAS];

/** Strictly decoded backend-to-app event union. */
export type BridgeEvent = z.infer<BridgeEventSchema>;

/** Backwards-compatible alias for the supported event union. */
export type KnownBridgeEvent = BridgeEvent;

/** Backwards-compatible alias for the strict decoder return type. */
export type DecodedBridgeEvent = BridgeEvent;

type EventOf<T extends BridgeEvent["type"]> = Extract<BridgeEvent, { type: T }>;

/** Backend readiness event, including bridge package metadata. */
export type ReadyEvent = EventOf<"ready">;

/** Run accepted event emitted before AgentLane starts processing a prompt. */
export type RunStartEvent = EventOf<"run_start">;

/** Successful run completion event with final output and shim state. */
export type RunCompleteEvent = EventOf<"run_complete">;

/** Run cancellation event emitted when active work was cancelled. */
export type RunCancelledEvent = EventOf<"run_cancelled">;

/** Backend error event scoped to a command or active run. */
export type ErrorEvent = EventOf<"error">;

/** Streaming assistant text delta. */
export type AssistantDeltaEvent = EventOf<"assistant_delta">;

/** Streaming reasoning text delta and provider reasoning metadata. */
export type ReasoningDeltaEvent = EventOf<"reasoning_delta">;

/** Streaming tool-call argument delta from the model provider. */
export type ToolArgumentsDeltaEvent = EventOf<"tool_arguments_delta">;

/** Provider lifecycle event without bridge-specific side effects. */
export type ProviderEvent = EventOf<"provider_event">;

/** Agent task start event with task lineage. */
export type AgentStartEvent = EventOf<"agent_start">;

/** Agent task completion event with a final-output preview. */
export type AgentEndEvent = EventOf<"agent_end">;

/** Model request start event for an agent turn. */
export type LlmStartEvent = EventOf<"llm_start">;

/** Model response completion event with optional token usage. */
export type LlmEndEvent = EventOf<"llm_end">;

/** Tool invocation start event with raw tool arguments. */
export type ToolStartEvent = EventOf<"tool_start">;

/** Tool invocation completion event with result or error details. */
export type ToolEndEvent = EventOf<"tool_end">;

/** Plan-tool update event carrying current plan rows. */
export type PlanUpdatedEvent = EventOf<"plan_updated">;

/** Permission approval request event sent to the app UI. */
export type ApprovalRequestEvent = EventOf<"approval_request">;

/** Approval resolution event confirming backend receipt of a decision. */
export type ApprovalResolvedEvent = EventOf<"approval_resolved">;

/** Snapshot of backend conversation state at a lifecycle boundary. */
export type StateSnapshotEvent = EventOf<"state_snapshot">;

/** Handoff tool start event. */
export type HandoffStartEvent = EventOf<"handoff_start">;

/** Handoff tool completion event. */
export type HandoffEndEvent = EventOf<"handoff_end">;

/** Backend reset completion event. */
export type ResetEvent = EventOf<"reset">;

/** Confirmation that the backend accepted a cancel request. */
export type CancelRequestedEvent = EventOf<"cancel_requested">;

/** Notice that a cancel request had no active run to cancel. */
export type CancelIgnoredEvent = EventOf<"cancel_ignored">;

/** Backend shutdown event emitted before process exit. */
export type ShutdownEvent = EventOf<"shutdown">;

/** Diagnostic wrapper for AgentLane run events without dedicated UI semantics. */
export type RunEventEvent = EventOf<"run_event">;

/** Encode one app command as a single NDJSON frame for Python stdin. */
export function encodeBridgeCommand(command: BridgeCommand): string {
  const payload: VersionedBridgeCommand = {
    ...command,
    protocol_version: PROTOCOL_VERSION,
  };
  return `${JSON.stringify(payload)}\n`;
}

/** Return whether an event protocol version is accepted by this package. */
export function isSupportedProtocolVersion(value: string): boolean {
  // Minor versions must stay additive. A major version change may reinterpret
  // command or event fields and must be handled explicitly by both packages.
  const [major] = value.split(".");
  return Number(major) === PROTOCOL_MAJOR;
}
