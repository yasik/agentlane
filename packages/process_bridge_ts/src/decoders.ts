import {
  type ApprovalRequestPayload,
  type BridgeEnvelope,
  type DecodedBridgeEvent,
  isSupportedProtocolVersion,
  type KnownBridgeEvent,
  type LineageFields,
  type PlanStep,
  PROTOCOL_VERSION,
  type TokenUsage,
  type ToolErrorPayload,
  type UnknownBridgeEvent,
} from "./protocol.ts";

type Raw = Record<string, unknown>;
type KnownEventType = KnownBridgeEvent["type"];
type KnownEventPayload<K extends KnownEventType> = Omit<
  Extract<KnownBridgeEvent, { type: K }>,
  keyof BridgeEnvelope
>;
type EventDecoder<K extends KnownEventType> = (
  fields: FieldReader,
) => KnownEventPayload<K>;
type ApprovalRequiredStringKey = "cwd" | "operation" | "tool_name";
type ApprovalOptionalStringKey =
  | "agent_name"
  | "command"
  | "path"
  | "reason"
  | "run_id"
  | "skill_name"
  | "tool_call_id";
type ApprovalRecordKey = "metadata";
type FallbackRecorder = (field: string) => void;

type FieldReader = {
  bool: (key: string, fallback?: boolean) => boolean;
  envelope: () => BridgeEnvelope;
  fallback: (field: string) => void;
  num: (key: string, fallback?: number) => number;
  numOrNull: (key: string) => number | null;
  raw: (key: string, fallback?: unknown) => unknown;
  record: (key: string) => Record<string, unknown>;
  str: (key: string, fallback?: string) => string;
  strOrNull: (key: string) => string | null;
};

type DecoderMap = {
  [K in KnownEventType]: EventDecoder<K>;
};

// This is the hand-authored TypeScript side of the wire contract. The map type
// forces one decoder row per known event, and cross-package fixture tests ensure
// the Python bridge emits the same event set.
const DECODERS: DecoderMap = {
  ready: (f: FieldReader) => ({
    type: "ready",
    version: f.str("version", "0.0.0"),
    package: f.str("package", "agentlane-process-bridge"),
    metadata: f.record("metadata"),
  }),
  run_start: (f: FieldReader) => ({
    type: "run_start",
    prompt: f.str("prompt"),
  }),
  run_complete: (f: FieldReader) => ({
    type: "run_complete",
    final_output: f.str("final_output"),
    turn_count: f.num("turn_count"),
    response_count: f.num("response_count"),
    shim_state: f.record("shim_state"),
  }),
  run_cancelled: () => ({
    type: "run_cancelled",
  }),
  error: (f: FieldReader) => ({
    type: "error",
    message: f.str("message", "unknown error"),
    scope: f.str("scope", "run") === "command" ? "command" : "run",
  }),
  assistant_delta: (f: FieldReader) => ({
    type: "assistant_delta",
    text: f.str("text"),
  }),
  reasoning_delta: (f: FieldReader) => ({
    type: "reasoning_delta",
    text: f.str("text"),
    provider_event_type: f.strOrNull("provider_event_type"),
    reasoning_signature: f.strOrNull("reasoning_signature"),
  }),
  tool_arguments_delta: (f: FieldReader) => ({
    type: "tool_arguments_delta",
    tool_call_id: f.str("tool_call_id"),
    tool_call_index: f.numOrNull("tool_call_index"),
    delta: f.str("delta"),
  }),
  provider_event: (f: FieldReader) => ({
    type: "provider_event",
    provider_event_type: f.strOrNull("provider_event_type"),
    item_index: f.numOrNull("item_index"),
    item_type: f.strOrNull("item_type"),
    phase: f.strOrNull("phase"),
  }),
  agent_start: (f: FieldReader) => ({
    type: "agent_start",
    ...lineage(f),
    agent: f.str("agent", "unknown"),
    next_turn: f.numOrNull("next_turn"),
  }),
  agent_end: (f: FieldReader) => ({
    type: "agent_end",
    ...lineage(f),
    agent: f.str("agent", "unknown"),
    final_preview: f.strOrNull("final_preview"),
  }),
  llm_start: (f: FieldReader) => ({
    type: "llm_start",
    ...lineage(f),
    agent: f.str("agent", "unknown"),
    message_count: f.num("message_count"),
  }),
  llm_end: (f: FieldReader) => ({
    type: "llm_end",
    ...lineage(f),
    agent: f.str("agent", "unknown"),
    output_preview: f.strOrNull("output_preview"),
    usage: tokenUsage(f.raw("usage")),
  }),
  tool_start: (f: FieldReader) => ({
    type: "tool_start",
    ...lineage(f),
    agent: f.str("agent", "unknown"),
    tool: f.str("tool", "unknown"),
    tool_call_id: f.str("tool_call_id"),
    arguments: f.raw("arguments"),
    is_plan: f.bool("is_plan"),
    is_delegation: f.bool("is_delegation"),
  }),
  tool_end: (f: FieldReader) => ({
    type: "tool_end",
    ...lineage(f),
    agent: f.str("agent", "unknown"),
    tool: f.str("tool", "unknown"),
    tool_call_id: f.str("tool_call_id"),
    result: f.raw("result"),
    ok: f.bool("ok", true),
    error: toolError(f.raw("error")),
    is_plan: f.bool("is_plan"),
    is_delegation: f.bool("is_delegation"),
  }),
  plan_updated: (f: FieldReader) => ({
    type: "plan_updated",
    ...lineage(f),
    agent: f.str("agent", "unknown"),
    tool_call_id: f.str("tool_call_id"),
    explanation: f.strOrNull("explanation"),
    steps: planSteps(f.raw("steps", []), f.fallback),
    title: f.strOrNull("title"),
    raw: f.raw("raw", null),
  }),
  approval_request: (f: FieldReader) => ({
    type: "approval_request",
    id: f.str("id"),
    request: approvalRequest(f.record("request"), f.fallback),
    reason: f.strOrNull("reason"),
  }),
  approval_resolved: (f: FieldReader) => ({
    type: "approval_resolved",
    id: f.str("id"),
    allowed: f.bool("allowed"),
    request: approvalRequest(f.record("request"), f.fallback),
    reason: f.strOrNull("reason"),
  }),
  state_snapshot: (f: FieldReader) => ({
    type: "state_snapshot",
    boundary: f.str("boundary"),
    turn_count: f.num("turn_count"),
    history_length: f.num("history_length"),
    response_count: f.num("response_count"),
    shim_state: f.record("shim_state"),
  }),
  handoff_start: (f: FieldReader) => ({
    type: "handoff_start",
    ...lineage(f),
    agent: f.str("agent", "unknown"),
    target: f.str("target", "unknown"),
    tool: f.str("tool", "unknown"),
    tool_call_id: f.str("tool_call_id"),
  }),
  handoff_end: (f: FieldReader) => ({
    type: "handoff_end",
    ...lineage(f),
    agent: f.str("agent", "unknown"),
    target: f.str("target", "unknown"),
    tool: f.str("tool", "unknown"),
    tool_call_id: f.str("tool_call_id"),
    final_preview: f.strOrNull("final_preview"),
  }),
  reset: () => ({ type: "reset" }),
  cancel_requested: () => ({
    type: "cancel_requested",
  }),
  cancel_ignored: (f: FieldReader) => ({
    type: "cancel_ignored",
    reason: f.str("reason", "unknown"),
  }),
  shutdown: () => ({
    type: "shutdown",
  }),
  run_event: (f: FieldReader) => ({
    type: "run_event",
    run_event_type: f.str("run_event_type", "unknown"),
  }),
};

export const KNOWN_EVENT_TYPES: readonly KnownEventType[] = Object.keys(
  DECODERS,
) as KnownEventType[];

export function isKnownEventType(type: string): type is KnownEventType {
  return Object.hasOwn(DECODERS, type);
}

export function decodeBridgeEventLine(line: string): DecodedBridgeEvent | null {
  let value: unknown;
  try {
    value = JSON.parse(line);
  } catch {
    return null;
  }

  // Decode tolerantly and report every repair. Process wiring decides whether a
  // repaired known event is safe to deliver to app reducers.
  const raw = asRecord(value);
  if (raw === null || typeof raw.type !== "string") return null;

  const fallbacks: string[] = [];
  const recordFallback = (field: string): void => {
    fallbacks.push(field);
  };

  if (typeof raw.protocol_version !== "string") {
    // Keep malformed frames inspectable, but mark the envelope repair so strict
    // consumers can reject synthesized protocol fields.
    raw.protocol_version = PROTOCOL_VERSION;
    recordFallback("protocol_version");
  } else if (!isSupportedProtocolVersion(raw.protocol_version)) {
    return null;
  }

  const f = fieldReader(raw, recordFallback);
  const event = isKnownEventType(raw.type)
    ? decodeKnownEvent(raw.type, f)
    : unknownEvent(raw.type, raw, f.envelope());
  return { event, fallbacks };
}

function decodeKnownEvent<K extends KnownEventType>(
  type: K,
  fields: FieldReader,
): Extract<KnownBridgeEvent, { type: K }> {
  return {
    ...DECODERS[type](fields),
    ...fields.envelope(),
  } as Extract<KnownBridgeEvent, { type: K }>;
}

function fieldReader(
  payload: Raw,
  recordFallback: FallbackRecorder,
): FieldReader {
  // All unsafe field reads go through this helper so fallback field names stay
  // consistent across simple fields, nested payloads, and fixture assertions.
  const miss = <T>(key: string, fallback: T): T => {
    recordFallback(key);
    return fallback;
  };

  return {
    bool: (key: string, fallback: boolean = false): boolean => {
      const value = payload[key];
      return typeof value === "boolean" ? value : miss(key, fallback);
    },
    envelope: (): BridgeEnvelope => ({
      protocol_version:
        typeof payload.protocol_version === "string"
          ? payload.protocol_version
          : miss("protocol_version", PROTOCOL_VERSION),
      ts:
        typeof payload.ts === "number" && Number.isFinite(payload.ts)
          ? payload.ts
          : miss("ts", 0),
    }),
    fallback: (field: string): void => {
      recordFallback(field);
    },
    num: (key: string, fallback: number = 0): number => {
      const value = payload[key];
      return typeof value === "number" && Number.isFinite(value)
        ? value
        : miss(key, fallback);
    },
    numOrNull: (key: string): number | null => {
      const value = payload[key];
      if (typeof value === "number" && Number.isFinite(value)) return value;
      if (value === null) return null;
      return miss(key, null);
    },
    raw: (key: string, fallback: unknown = null): unknown =>
      Object.hasOwn(payload, key) ? payload[key] : miss(key, fallback),
    record: (key: string): Record<string, unknown> =>
      asRecord(payload[key]) ?? miss(key, {}),
    str: (key: string, fallback: string = ""): string => {
      const value = payload[key];
      return typeof value === "string" ? value : miss(key, fallback);
    },
    strOrNull: (key: string): string | null => {
      const value = payload[key];
      if (typeof value === "string" || value === null) return value;
      return miss(key, null);
    },
  };
}

function unknownEvent(
  type: string,
  payload: Record<string, unknown>,
  envelope: BridgeEnvelope,
): UnknownBridgeEvent {
  // Preserve unknown payloads verbatim so older apps can surface newer bridge
  // events instead of dropping them or pretending they decoded successfully.
  return { ...envelope, type: "unknown_event", event_type: type, payload };
}

function approvalRequest(
  record: Record<string, unknown>,
  fallback: (field: string) => void,
): ApprovalRequestPayload {
  // Approval fields are nested under `request`; these helpers keep diagnostic
  // paths stable as `request.<field>` instead of scattering strings by hand.
  return {
    tool_name: requestString(record, fallback, "tool_name"),
    operation: requestString(record, fallback, "operation"),
    cwd: requestString(record, fallback, "cwd"),
    path: requestStringOrNull(record, fallback, "path"),
    command: requestStringOrNull(record, fallback, "command"),
    skill_name: requestStringOrNull(record, fallback, "skill_name"),
    reason: requestStringOrNull(record, fallback, "reason"),
    run_id: requestStringOrNull(record, fallback, "run_id"),
    agent_name: requestStringOrNull(record, fallback, "agent_name"),
    tool_call_id: requestStringOrNull(record, fallback, "tool_call_id"),
    metadata: requestRecord(record, fallback, "metadata"),
  };
}

function planSteps(
  value: unknown,
  fallback: (field: string) => void,
): PlanStep[] {
  if (!Array.isArray(value)) {
    fallback("steps");
    return [];
  }

  return value.map((item, index) => {
    const record = asRecord(item);
    if (record === null) {
      fallback(`steps.${index}`);
      return { status: "", step: "" };
    }
    return {
      status: nestedString(record, `steps.${index}.status`, fallback, "status"),
      step: nestedString(record, `steps.${index}.step`, fallback, "step"),
    };
  });
}

function lineage(f: FieldReader): LineageFields {
  return {
    task_id: f.str("task_id"),
    parent_task_id: f.strOrNull("parent_task_id"),
    is_root: f.bool("is_root", true),
    is_subagent: f.bool("is_subagent"),
  };
}

function nestedString(
  record: Record<string, unknown>,
  field: string,
  fallback: (field: string) => void,
  key: string = field,
): string {
  const value = record[key];
  if (typeof value === "string") return value;
  fallback(field);
  return "";
}

function nestedStringOrNull(
  record: Record<string, unknown>,
  field: string,
  fallback: (field: string) => void,
  key: string = field,
): string | null {
  const value = record[key];
  if (typeof value === "string" || value === null) return value;
  fallback(field);
  return null;
}

function requestString(
  record: Record<string, unknown>,
  fallback: (field: string) => void,
  key: ApprovalRequiredStringKey,
): string {
  return nestedString(record, `request.${key}`, fallback, key);
}

function requestStringOrNull(
  record: Record<string, unknown>,
  fallback: (field: string) => void,
  key: ApprovalOptionalStringKey,
): string | null {
  return nestedStringOrNull(record, `request.${key}`, fallback, key);
}

function requestRecord(
  record: Record<string, unknown>,
  fallback: (field: string) => void,
  key: ApprovalRecordKey,
): Record<string, unknown> {
  const value = asRecord(record[key]);
  if (value !== null) return value;

  fallback(`request.${key}`);
  return {};
}

function toolError(value: unknown): ToolErrorPayload | null {
  if (value === null || value === undefined) return null;
  const record = asRecord(value);
  if (record === null || typeof record.message !== "string") return null;
  return {
    message: record.message,
    kind: typeof record.kind === "string" ? record.kind : null,
  };
}

function tokenUsage(value: unknown): TokenUsage | null {
  // null is the on-wire value when the provider omits usage; only a
  // well-formed object with three numeric fields decodes to a usage payload.
  const record = asRecord(value);
  if (record === null) return null;
  const { prompt_tokens, completion_tokens, total_tokens } = record;
  if (
    typeof prompt_tokens !== "number" ||
    typeof completion_tokens !== "number" ||
    typeof total_tokens !== "number"
  ) {
    return null;
  }
  return { prompt_tokens, completion_tokens, total_tokens };
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}
