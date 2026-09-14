import { describe, expect, test } from "bun:test";
import {
  BridgeDecodeError,
  decodeBridgeEventLine,
  KNOWN_EVENT_TYPES,
  tryDecodeBridgeEventLine,
} from "../src/decoders.ts";
import { encodeBridgeCommand, PROTOCOL_VERSION } from "../src/protocol.ts";

describe("protocol commands", () => {
  test("encodes exactly one versioned NDJSON command", () => {
    const line = encodeBridgeCommand({ type: "prompt", text: "hello" });
    expect(line.endsWith("\n")).toBe(true);
    expect(line.split("\n")).toHaveLength(2);
    expect(JSON.parse(line)).toEqual({
      protocol_version: PROTOCOL_VERSION,
      type: "prompt",
      text: "hello",
    });
  });

  test("encodes configure patches as opaque JSON objects", () => {
    const line = encodeBridgeCommand({
      type: "configure",
      patch: { model: "anthropic/claude-opus-4-8" },
    });

    expect(JSON.parse(line)).toEqual({
      protocol_version: PROTOCOL_VERSION,
      type: "configure",
      patch: { model: "anthropic/claude-opus-4-8" },
    });
  });
});

describe("event decoding", () => {
  test("decodes known event payloads", () => {
    const decoded = decodeBridgeEventLine(
      JSON.stringify({
        protocol_version: "1.0",
        type: "error",
        ts: 1,
        message: "bad",
        scope: "command",
      }),
    );

    expect(decoded).toMatchObject({
      protocol_version: "1.0",
      type: "error",
      ts: 1,
      message: "bad",
      scope: "command",
    });
  });

  test("rejects missing typed fields", () => {
    const error = decodeErrorFor(
      JSON.stringify({ protocol_version: "1.0", type: "run_start", ts: 1 }),
    );

    expect(error.fields).toEqual(["prompt"]);
  });

  test("rejects extra fields on known event payloads", () => {
    const error = decodeErrorFor(
      JSON.stringify({
        protocol_version: "1.0",
        type: "run_start",
        ts: 1,
        prompt: "go",
        unexpected: true,
      }),
    );

    expect(error.fields).toEqual(["event"]);
  });

  test("decodes ready, reset, and config documents", () => {
    const ready = decodeBridgeEventLine(
      JSON.stringify({
        protocol_version: "1.0",
        type: "ready",
        ts: 1,
        version: "0.1.0",
        package: "agentlane-process-bridge",
        config: { model: "openai/gpt-5.5" },
      }),
    );
    const config = decodeBridgeEventLine(
      JSON.stringify({
        protocol_version: "1.0",
        type: "config",
        ts: 2,
        ok: true,
        config: { model: "anthropic/claude-opus-4-8" },
        error: null,
      }),
    );
    const reset = decodeBridgeEventLine(
      JSON.stringify({
        protocol_version: "1.0",
        type: "reset",
        ts: 3,
        config: { model: "anthropic/claude-opus-4-8" },
      }),
    );

    expect(ready).toMatchObject({
      type: "ready",
      config: { model: "openai/gpt-5.5" },
    });
    expect(config).toMatchObject({ type: "config", ok: true });
    expect(reset).toMatchObject({
      type: "reset",
      config: { model: "anthropic/claude-opus-4-8" },
    });
  });

  test("decodes every configure failure code", () => {
    for (const code of ["invalid", "unsupported", "rejected", "internal"]) {
      const decoded = decodeBridgeEventLine(
        JSON.stringify({
          protocol_version: "1.0",
          type: "config",
          ts: 1,
          ok: false,
          config: { model: "openai/gpt-5.5" },
          error: { code, message: `${code} failure` },
        }),
      );

      expect(decoded).toMatchObject({
        type: "config",
        ok: false,
        error: { code, message: `${code} failure` },
      });
    }
  });

  test("rejects invalid config settlement invariants", () => {
    const successWithoutConfig = decodeErrorFor(
      JSON.stringify({
        protocol_version: "1.0",
        type: "config",
        ts: 1,
        ok: true,
        config: null,
        error: null,
      }),
    );
    const failureWithoutError = decodeErrorFor(
      JSON.stringify({
        protocol_version: "1.0",
        type: "config",
        ts: 1,
        ok: false,
        config: null,
        error: null,
      }),
    );

    expect(successWithoutConfig.fields).toContain("config");
    expect(failureWithoutError.fields).toContain("error");
  });

  test("rejects unknown events", () => {
    const error = decodeErrorFor(
      JSON.stringify({
        protocol_version: "1.0",
        type: "new_event",
        ts: 1,
        value: true,
      }),
    );

    expect(error.fields).toEqual(["type"]);
    expect(error.message).toContain("Unknown bridge event type");
  });

  test("rejects unsupported protocol major versions", () => {
    expect(
      tryDecodeBridgeEventLine(
        JSON.stringify({ protocol_version: "2.0", type: "ready", ts: 1 }),
      ),
    ).toBeNull();
  });

  test("known event list includes lifecycle and approval events", () => {
    expect(KNOWN_EVENT_TYPES).toContain("run_event");
    expect(KNOWN_EVENT_TYPES).toContain("run_complete");
  });
});

function decodeErrorFor(line: string): BridgeDecodeError {
  try {
    decodeBridgeEventLine(line);
  } catch (error) {
    if (error instanceof BridgeDecodeError) return error;

    throw error;
  }

  throw new Error("Expected bridge decode to fail.");
}

test("retains unknown native records and all nested fields", () => {
  const event = {
    protocol_version: "1.0",
    ts: 1,
    type: "run_event" as const,
    event: {
      type: "future_native",
      source_extra: { nested: [null, "界\n"] },
      payload: { kind: "future_native", data: { complete: true } },
    },
  };
  expect(decodeBridgeEventLine(JSON.stringify(event))).toEqual(event);
});

test("rejects mismatched native kinds and missing source fields", () => {
  const frame = (type: string, payload: unknown): string =>
    JSON.stringify({
      protocol_version: "1.0",
      ts: 1,
      type: "run_event",
      event: { type, payload },
    });
  expect(
    decodeErrorFor(frame("agent_start", { kind: "agent_end" })).fields,
  ).toContain("event.payload.kind");
  expect(
    decodeErrorFor(frame("tool_start", { kind: "tool_start" })).fields,
  ).toContain("event.payload.tool_call");
  expect(
    decodeErrorFor(
      frame("model_stream", {
        kind: "model_stream",
        event: { kind: "text_delta" },
      }),
    ).fields,
  ).toContain("event.payload.event.text");
  expect(
    decodeErrorFor(
      frame("tool_approval", { kind: "tool_approval", event: { record: {} } }),
    ).fields,
  ).toContain("event.payload.event.record.request");
});

test("accepts unknown model kinds and preserves their fields", () => {
  const event = {
    protocol_version: "1.0",
    ts: 1,
    type: "run_event" as const,
    event: {
      type: "model_stream",
      payload: {
        kind: "model_stream",
        event: { kind: "future_model", details: [null, { signature: "abc" }] },
      },
    },
  };
  expect(decodeBridgeEventLine(JSON.stringify(event))).toEqual(event);
});

test("retains source keys that object-schema parsing can omit", () => {
  const line =
    '{"protocol_version":"1.0","ts":1,"type":"run_event","event":{"type":"future_native","__proto__":{"source":true},"payload":{"kind":"future_native","__proto__":{"nested":true}}}}';
  expect(JSON.stringify(decodeBridgeEventLine(line))).toBe(line);
});
