import type { z } from "zod";
import {
  BRIDGE_EVENT_SCHEMAS,
  type BridgeEvent,
  isSupportedProtocolVersion,
} from "./protocol.ts";

type Raw = Record<string, unknown>;
type KnownEventType = keyof typeof BRIDGE_EVENT_SCHEMAS;

/** Event names that this TypeScript package can decode from Python stdout. */
export const KNOWN_EVENT_TYPES: readonly KnownEventType[] = Object.keys(
  BRIDGE_EVENT_SCHEMAS,
) as KnownEventType[];

/**
 * Strict protocol decode failure.
 *
 * `fields` contains the smallest known payload paths that failed validation so
 * app shells can surface actionable drift diagnostics without receiving a
 * partially repaired event.
 */
export class BridgeDecodeError extends Error {
  /** Payload field paths involved in the decode failure. */
  readonly fields: readonly string[];

  constructor(message: string, fields: readonly string[] = []) {
    super(message);
    this.name = "BridgeDecodeError";
    this.fields = fields;
  }
}

/** Return whether a raw string is one of the supported bridge event names. */
export function isKnownEventType(type: string): type is KnownEventType {
  return Object.hasOwn(BRIDGE_EVENT_SCHEMAS, type);
}

/**
 * Decode one NDJSON stdout line into a typed bridge event.
 *
 * This is intentionally strict. Unknown event names, unsupported protocol major
 * versions, and missing required fields throw `BridgeDecodeError`; callers must
 * not synthesize defaults because Python and TypeScript are released together.
 */
export function decodeBridgeEventLine(line: string): BridgeEvent {
  let value: unknown;
  try {
    value = JSON.parse(line);
  } catch (error) {
    throw new BridgeDecodeError(`Invalid JSON bridge event: ${error}`);
  }

  const raw = asRecord(value);
  if (raw === null) {
    throw new BridgeDecodeError("Bridge event must be a JSON object.");
  }

  // Validate the envelope before selecting an event schema. Without this guard
  // an incompatible protocol could be parsed as a current-version event.
  const version = raw.protocol_version;
  if (typeof version !== "string") {
    throw new BridgeDecodeError(
      "Bridge event protocol_version must be a string.",
      ["protocol_version"],
    );
  }

  if (!isSupportedProtocolVersion(version)) {
    throw new BridgeDecodeError(
      `Unsupported bridge protocol version: ${version}.`,
      ["protocol_version"],
    );
  }

  // Event name validation happens before Zod parsing so unknown bridge changes
  // produce an explicit protocol error rather than a generic union mismatch.
  const eventType = raw.type;
  if (typeof eventType !== "string") {
    throw new BridgeDecodeError("Bridge event type must be a string.", [
      "type",
    ]);
  }

  if (!isKnownEventType(eventType)) {
    throw new BridgeDecodeError(`Unknown bridge event type: ${eventType}.`, [
      "type",
    ]);
  }

  // The schema map is the single TypeScript mirror of the Python event surface.
  // Adding a bridge event should mean adding exactly one schema entry here.
  const parsed = BRIDGE_EVENT_SCHEMAS[eventType].safeParse(raw);
  if (!parsed.success) {
    throw decodeError(parsed.error);
  }

  return parsed.data as BridgeEvent;
}

/**
 * Decode a line when the caller only needs success/failure.
 *
 * Non-bridge exceptions are rethrown so unexpected runtime defects do not get
 * hidden behind the protocol-error path.
 */
export function tryDecodeBridgeEventLine(line: string): BridgeEvent | null {
  try {
    return decodeBridgeEventLine(line);
  } catch (error) {
    if (error instanceof BridgeDecodeError) {
      return null;
    }

    throw error;
  }
}

function decodeError(error: z.ZodError): BridgeDecodeError {
  // Collapse duplicate issue paths so a single invalid nested object creates a
  // concise diagnostic for the app trace and tests.
  const fields = [...new Set(error.issues.map(issuePath))];
  return new BridgeDecodeError(
    `Invalid bridge event payload: ${fields.join(", ") || "event"}.`,
    fields,
  );
}

function issuePath(issue: z.ZodIssue): string {
  return issue.path.length === 0 ? "event" : issue.path.map(String).join(".");
}

function asRecord(value: unknown): Raw | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;

  return value as Raw;
}
