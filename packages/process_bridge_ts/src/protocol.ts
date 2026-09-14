import { z } from "zod";
import { configEventSchema } from "./protocol-config.ts";

export type { ConfigErrorCode, ConfigErrorPayload } from "./protocol-config.ts";

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
  "configure",
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
  | { type: "configure"; patch: Record<string, unknown> }
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

export type {
  ApprovalRequestPayload,
  TokenUsage,
  ToolErrorPayload,
} from "./protocol-native.ts";

import { NATIVE_RUN_EVENT_SCHEMA } from "./protocol-native.ts";

type BridgeEnvelopeShape = {
  protocol_version: z.ZodString;
  ts: z.ZodNumber;
};

/** JSON object payload with string keys and bridge-owned unknown values. */
const recordSchema: z.ZodRecord<z.ZodString, z.ZodUnknown> = z.record(
  z.string(),
  z.unknown(),
);

/** Common schema for the event envelope each backend line must include. */
const bridgeEnvelopeSchema: z.ZodObject<BridgeEnvelopeShape> = z
  .object({
    protocol_version: z.string(),
    ts: z.number(),
  })
  .strict();

/** Build one strict event schema while preserving open app-owned record fields. */
const bridgeEventSchema = <TShape extends z.ZodRawShape>(
  shape: TShape,
): z.ZodObject<BridgeEnvelopeShape & TShape> =>
  bridgeEnvelopeSchema.extend(shape).strict();

/** Schema for the closed set of error scopes emitted by Python. */
const errorScopeSchema: z.ZodType<ErrorScope> = z.enum(["command", "run"]);

/**
 * Strict backend-to-app event schema registry.
 *
 * This map is the TypeScript mirror of the Python bridge event surface. A new
 * event type should add one schema entry here and one representative fixture so
 * parity tests can catch missing or stale protocol updates.
 */
export const BRIDGE_EVENT_SCHEMAS = {
  ready: bridgeEventSchema({
    type: z.literal("ready"),
    version: z.string(),
    package: z.string(),
    metadata: recordSchema.optional(),
    config: recordSchema.optional(),
  }),
  config: configEventSchema,
  run_start: bridgeEventSchema({
    type: z.literal("run_start"),
    prompt: z.string(),
  }),
  run_complete: bridgeEventSchema({
    type: z.literal("run_complete"),
    final_output: z.unknown(),
    turn_count: z.number(),
    response_count: z.number(),
    shim_state: recordSchema,
  }),
  run_cancelled: bridgeEventSchema({
    type: z.literal("run_cancelled"),
  }),
  error: bridgeEventSchema({
    type: z.literal("error"),
    message: z.string(),
    scope: errorScopeSchema,
  }),
  reset: bridgeEventSchema({
    type: z.literal("reset"),
    config: recordSchema.optional(),
  }),
  cancel_requested: bridgeEventSchema({
    type: z.literal("cancel_requested"),
  }),
  cancel_ignored: bridgeEventSchema({
    type: z.literal("cancel_ignored"),
    reason: z.string(),
  }),
  shutdown: bridgeEventSchema({
    type: z.literal("shutdown"),
  }),
  run_event: bridgeEventSchema({
    type: z.literal("run_event"),
    event: NATIVE_RUN_EVENT_SCHEMA,
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

/** Configure settlement event carrying the authoritative config document. */
export type ConfigEvent = EventOf<"config">;

/** Run accepted event emitted before AgentLane starts processing a prompt. */
export type RunStartEvent = EventOf<"run_start">;

/** Successful run completion event with final output and shim state. */
export type RunCompleteEvent = EventOf<"run_complete">;

/** Run cancellation event emitted when active work was cancelled. */
export type RunCancelledEvent = EventOf<"run_cancelled">;

/** Backend error event scoped to a command or active run. */
export type ErrorEvent = EventOf<"error">;

/** Backend reset completion event. */
export type ResetEvent = EventOf<"reset">;

/** Confirmation that the backend accepted a cancel request. */
export type CancelRequestedEvent = EventOf<"cancel_requested">;

/** Notice that a cancel request had no active run to cancel. */
export type CancelIgnoredEvent = EventOf<"cancel_ignored">;

/** Backend shutdown event emitted before process exit. */
export type ShutdownEvent = EventOf<"shutdown">;

/** Complete native AgentLane event inside the bridge transport envelope. */
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
