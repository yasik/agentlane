import { z } from "zod";
import type { BridgeEnvelope } from "./protocol.ts";

/**
 * Closed transport-level failure taxonomy for configure settlements.
 *
 * App-specific rejection reasons live in `message`; the code stays limited so
 * session code can make stable promise-settlement decisions.
 */
export type ConfigErrorCode =
  | "invalid"
  | "unsupported"
  | "rejected"
  | "internal";

/** Failure details carried by a failed config settlement event. */
export type ConfigErrorPayload = {
  /** Transport-level reason class understood by the session API. */
  code: ConfigErrorCode;

  /** User-presentable text for `rejected`, fixed diagnostic text otherwise. */
  message: string;
};

/** Strict config settlement payload before it joins the wider BridgeEvent union. */
type ConfigEventPayload = BridgeEnvelope & {
  /** Dedicated settlement event type for accepted configure commands. */
  type: "config";

  /** True when Python applied the patch; false when it rejected or failed it. */
  ok: boolean;

  /** Full authoritative document, or null only on failure without a snapshot. */
  config: Record<string, unknown> | null;

  /** Closed failure details, present only for failed settlements. */
  error: ConfigErrorPayload | null;
};

/** Opaque config document shape; key meaning belongs to the app store. */
const configDocumentSchema: z.ZodRecord<z.ZodString, z.ZodUnknown> = z.record(
  z.string(),
  z.unknown(),
);

/** Schema for the closed set of configure failure codes emitted by Python. */
const configErrorCodeSchema: z.ZodType<ConfigErrorCode> = z.enum([
  "invalid",
  "unsupported",
  "rejected",
  "internal",
]);

/** Schema for backend-reported configure failure details. */
const configErrorPayloadSchema: z.ZodType<ConfigErrorPayload> = z
  .object({
    code: configErrorCodeSchema,
    message: z.string(),
  })
  .strict();

/**
 * Config settlement schema.
 *
 * Config contents are opaque, but the envelope is strict: success must announce
 * the applied document, and failure must announce a closed error code. This
 * keeps `configure()` settlement deterministic while leaving app config shape
 * to `decodeConfig` and the Python `RuntimeConfigStore`.
 */
export const configEventSchema: z.ZodType<ConfigEventPayload> = z
  .object({
    protocol_version: z.string(),
    ts: z.number(),
    type: z.literal("config"),
    ok: z.boolean(),
    config: configDocumentSchema.nullable(),
    error: configErrorPayloadSchema.nullable(),
  })
  .strict()
  .superRefine((event, context) => {
    // A successful configure without a document would resolve the promise with
    // no backend truth. Treat that as protocol drift, not an empty config.
    if (event.ok && event.config === null) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: "successful config events require config",
        path: ["config"],
      });
    }

    // Error details on success are ambiguous for callers; Python should emit
    // exactly one of applied config or failure details.
    if (event.ok && event.error !== null) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: "successful config events must not include error",
        path: ["error"],
      });
    }

    // Failures may omit `config` only when no snapshot is obtainable, but the
    // error object is mandatory so configure() can reject with a stable code.
    if (!event.ok && event.error === null) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: "failed config events require error",
        path: ["error"],
      });
    }
  });
